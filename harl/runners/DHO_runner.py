import torch
import numpy as np
from harl.common.buffers.DHO_rollout_buffer import RolloutBuffer, WM_RolloutBuffer
from harl.algorithms.representation.tdd import TDDModel, mrn_distance
from harl.algorithms.representation.wm import DreamerWorldModel
import matplotlib
matplotlib.use('Agg')  # Headless backend 설정
import matplotlib.pyplot as plt
import os
import time
import logging
logger = logging.getLogger(__name__)

class TddRunner:  # tdd_args가 none이 아닐때만 호출 됨
    def __init__(self, n_rollout_threads, num_agents, observation_space, tdd_args=None, env_args=None, save_dir=None):
        if tdd_args is None:
            print("TDD is disabled")
            return
        
        print("TDD is enabled")
        self.tdd_args = tdd_args
        self.save_dir = save_dir
        
        self.model = None
        self.optimizer = None
        # GPU 사용 강제
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"GPU 사용: {torch.cuda.get_device_name(self.device)}")
        else:
            print("경고: GPU를 사용할 수 없습니다. CPU를 사용하지만 성능이 저하될 수 있습니다.")
            self.device = torch.device("cpu")
        
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents
        self.observation_space = observation_space
        
        if self.tdd_args["network"]["use_full_p_obs"] and not self.tdd_args["network"]["use_intra_obs"]:
            self.tdd_model = TDDModel(self.tdd_args, self.num_agents, 2 + 2 + 2 * self.num_agents + 4 * (self.num_agents - 1), self.device, run_dir=save_dir)
        elif self.tdd_args["network"]["use_intra_obs"]:
            self.tdd_model = TDDModel(self.tdd_args, self.num_agents, 4, self.device, run_dir=save_dir)
        elif self.tdd_args["network"]["use_p_obs_without_others"]:
            self.tdd_model = TDDModel(self.tdd_args, self.num_agents, 2 + 2 + 2 * (self.num_agents), self.device, run_dir=save_dir)
        else:
            self.tdd_model = TDDModel(self.tdd_args, self.num_agents, 2, self.device, run_dir=save_dir)

        self.max_historical_samples = self.tdd_args["train"]["max_historical_samples"]
        self.max_batch_size = self.tdd_args["train"]["max_batch_size"]
        self.default_k_value = self.tdd_args["train"]["default_k_value"]
        
        # 보상 계산 관련 변수 초기화
        self.reward_update_counter = 0
        self.last_reward_update = None
        self.prev_obs = None
        
        self.rollout_buffer = RolloutBuffer(
                {**self.tdd_args, **env_args},
                self.observation_space,
                self.num_agents,
                self.n_rollout_threads
        )
    
    def update_tdd_model(self, is_warm_up=False):
        if len(self.rollout_buffer.rollout_history[0]) == 0:
            raise ValueError("rollout_buffer.rollout_history가 비어있습니다.")
        self.tdd_model.update(self.rollout_buffer.rollout_history, is_warm_up=is_warm_up)

    def _compute_mrn_distance(self, current_state, prev_states, agent_id, n_rollout_threads):
        """현재 상태와 이전 상태들 간의 MRN 거리를 계산합니다."""
        current_state = torch.tensor(current_state, device=self.device).float()
        if self.tdd_args["network"]["use_independent_nets"]:
            phi_y = self.tdd_model.s_encoder[agent_id](current_state)
        else:
            phi_y = self.tdd_model.s_encoder(current_state)
        
        prev_states = torch.tensor(np.array(prev_states), device=self.device).float()
        prev_states = prev_states.view(-1, prev_states.shape[-1])
        
        # 안전한 나눗셈을 위한 체크
        if prev_states.shape[0] < n_rollout_threads:
            # 데이터가 부족한 경우 기본값 반환
            return torch.zeros(n_rollout_threads, device=self.device)
        
        if self.tdd_args["network"]["use_independent_nets"]:
            phi_x = self.tdd_model.s_encoder[agent_id](prev_states)
        else:
            phi_x = self.tdd_model.s_encoder(prev_states)
        
        phi_y_repeated = phi_y.repeat(prev_states.shape[0] // n_rollout_threads, 1)
        dists = mrn_distance(phi_x, phi_y_repeated)
        dists = dists.view(prev_states.shape[0] // n_rollout_threads, n_rollout_threads)
        return torch.min(dists, dim=0)[0]

    def _get_prev_states(self, current_rollout_states, agent_id, pos, is_eval=False):
        """이전 상태들을 가져옵니다."""
        prev_states = []
        for i, state_dict in enumerate(current_rollout_states[agent_id]):
            if i == len(current_rollout_states[agent_id]) - 1:
                prev_states.append(state_dict['obs'])
                prev_states.append(pos[agent_id])
                if not np.allclose(pos[agent_id], state_dict['next_obs'], rtol=1e-5, atol=1e-5):
                    raise AssertionError(f"rollout_states의 마지막 상태의 next_obs({state_dict['next_obs']})가 현재 상태({pos[agent_id]})와 다른 경우입니다.")
            else:
                prev_states.append(state_dict['obs'])
        return prev_states

    def compute_intrinsic_reward(self, pos, new_pos, is_eval=False, temp_rollout_buffer=None, n_rollout_threads=None):
        int_rew = [[] for _ in range(self.num_agents)]
        if temp_rollout_buffer is None:
            current_rollout_states = self.rollout_buffer.current_rollout_states[:]
        else:
            current_rollout_states = temp_rollout_buffer    # (n_agents, {'obs': (2,), 'share_obs': (54,), 'next_obs': (2,), 'next_share_obs': (54,), 'dones': (1,)}) 여야함 (에이전트 수 3개 기준)
        
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                # 현재 상태에 대한 내재적 보상 계산
                if any(current_rollout_states):
                    prev_states = self._get_prev_states(current_rollout_states, agent_id, pos, is_eval)
                    min_dists = self._compute_mrn_distance(new_pos[agent_id], prev_states, agent_id, n_rollout_threads)
                    int_rew[agent_id].append(min_dists.cpu().numpy())   # 당장 이 때의 shape: list of (n_rollout_threads,)
                    
                    # 에이전트 간 상호작용 고려
                    if self.tdd_args["train"]["coeff_inter_agent_int_rew"] != 0:
                        other_min_dists_ls = []
                        for other_agent_id in range(self.num_agents):
                            if other_agent_id != agent_id:
                                other_prev_states = self._get_prev_states(current_rollout_states, other_agent_id, pos, is_eval)
                                other_min_dists = self._compute_mrn_distance(new_pos[agent_id], other_prev_states, other_agent_id, n_rollout_threads)
                                other_min_dists_ls.append(other_min_dists.cpu().numpy())    # n_agents - 1개의 이웃에 대해 n_rollout_threads 각각의 최소 temporal distance들을 저장해둠둠
                        
                        if self.tdd_args["train"]["use_updated_inter"]:
                            if len(self.rollout_buffer.current_rollout_states[agent_id]) > 0:
                                vals, _ = torch.topk(torch.tensor(np.array(other_min_dists_ls)), k=self.num_agents // 2, dim=0, largest=False)
                                other_min_dists_final = np.array(torch.log(1 + 1 / (self.num_agents // 2) * (torch.sum(vals, axis=0) ** (self.num_agents - 1))).cpu().numpy())  # other_min_dists_final의 shape는 (n_rollout_threads,)이다.
                                int_rew[agent_id] += other_min_dists_final * self.tdd_args["train"]["coeff_inter_agent_int_rew"]    # (1, n_rollout_threads)로 변신한다.
                            else:
                                print(f"calculate_inter_int_rew 함수에서 current_rollout_states가 비어있습니다. agent_id: {agent_id}, current_rollout_states: {self.rollout_buffer.current_rollout_states}")
                                int_rew[agent_id] += np.zeros(n_rollout_threads).reshape(1, n_rollout_threads) # np.zeros(n_rollout_threads).reshape(1, n_rollout_threads)의 shape는 (1, n_rollout_threads)이다.
                                # raise ValueError(f"calculate_inter_int_rew 함수에서 current_rollout_states가 비어있습니다. agent_id: {agent_id}, current_rollout_states: {self.rollout_buffer.current_rollout_states}")
                        else:
                            other_min_dists_final = np.min(np.array(other_min_dists_ls), axis=0)
                            int_rew[agent_id] += other_min_dists_final * self.tdd_args["train"]["coeff_inter_agent_int_rew"]    # (n_rollout_threads,)
                else:
                    int_rew[agent_id].append(np.zeros(n_rollout_threads))
        
        return np.array(int_rew).transpose(2, 0, 1)  # 저렇게 바꾸면 -> (n_rollout_threads, n_agents, 1)
    
    def calculate_central_state_entropy(self, new_pos, agent_id, step=None):
        batch_size = new_pos.shape[0]
        # 보통 1024, obs의 차원만큼 인풋이 들어올거다.
        rollout_buffer_all = self.rollout_buffer.rollout_history[:]    # (n_agents, n_trajs, max_cycles, (n_rollout_threads, {'obs': (2,), 'next_obs': (2,)})) ex/ (3, 29, 800, (dict...))
        
        # 빈 버퍼 체크
        if len(rollout_buffer_all) == 0 or len(rollout_buffer_all[0]) == 0:
            raise ValueError("calculate_central_state_entropy 함수에서 rollout_buffer가 비어있습니다.")
            # 기본 엔트로피 값으로 작은 양수 값 반환 (0보다는 크지만 매우 작은 값)
            # return torch.ones(batch_size, device=self.device) * 0.1
        
        # 우선 new_pos의 절대좌표만 잘라내자
        if new_pos.shape[1] >= 4:  # 최소 4차원 이상인지 확인
            new_pos_abs = new_pos[:, 2:4]
        else:
            # 차원이 부족한 경우 전체를 사용
            raise ValueError("new_pos의 차원이 부족합니다.")
        new_pos_abs = torch.tensor(new_pos_abs, device=self.device).float()
        
        # 가장 최근 스텝만 가져오기 위한 전초전
        n_agents = len(rollout_buffer_all)
        n_trajs = len(rollout_buffer_all[0])
        n_cycles = len(rollout_buffer_all[0][0])
        
        
        # numpy array로 변환하고 reshape
        rollout_array = np.array(rollout_buffer_all, dtype=object)
        # 아무튼간에 지금 rollout_array는 (n_agents, n_trajs, n_cycles) 형태이다.
        # n_trajs에서 뒤에 x 개를 뽑고, 그리고 n_cycles에서 뒤에 y 개를 뽑아야하는데 중요한 점은
        # x * y * n_rollout_threads * n_agents = batch_size 를 넘어야 한다.
        # factor = n_rollout_threads * n_agents라고 치고, x * y * factor = batch_size 를 만족하는 x, y를 찾아야 한다.
        # 그러면 x = batch_size // (y * factor) 가 되고, y = batch_size // (x * factor) 가 된다.
        factor = rollout_array[0][0][0]["obs"].shape[0] * n_agents
        x_times_y = batch_size // factor + 1
        safe_batch_size_1 = int(np.ceil(x_times_y / n_trajs))
        safe_batch_size_2 = int(np.ceil(x_times_y / n_cycles))
        safe_batch_size = int(max(max(safe_batch_size_1, safe_batch_size_2), np.ceil(np.sqrt(x_times_y))))
        
        rollout_array = rollout_array[:, -safe_batch_size:, -safe_batch_size:]
        flattened = rollout_array.reshape(-1)   # (x * y * factor,)차원정도 될 것 같은데
        
        # 빈 flattened 배열 체크
        if len(flattened) == 0:
            raise ValueError("calculate_central_state_entropy 함수에서 flattened가 비어있습니다.")
            # return torch.ones(batch_size, device=self.device) * 0.1
        
        all_obs_temp = np.array([buffer['obs'] for buffer in flattened])    # (엄청여러개, 2)
        all_obs_temp = all_obs_temp.reshape(-1, 2)
        all_obs = all_obs_temp
        
        # 최적화: Historical data 샘플링 제한
        if all_obs.shape[0] > self.max_historical_samples:
            # 가장 최근 데이터부터 MAX_HISTORICAL_SAMPLES만큼 선택
            all_obs = all_obs[-self.max_historical_samples:]
        elif batch_size <= all_obs.shape[0]:
            indices = np.random.choice(all_obs.shape[0], batch_size, replace=False)
            all_obs = all_obs[indices]
        else:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_graph_logging"]:
                logger.warning(f"batch_size({batch_size})가 all_obs.shape[0]({all_obs.shape[0]})보다 큽니다. new_pos_abs를 all_obs.shape[0]만큼 잘라냅니다.")
            new_pos_abs = new_pos_abs[:all_obs.shape[0]]
            if batch_size != all_obs.shape[0]:
                raise ValueError(f"batch_size({batch_size})와 all_obs.shape[0]({all_obs.shape[0]})가 다릅니다. 버퍼가 충분하지 않습니다. n_tajs: {n_trajs}, safe_batch_size: {safe_batch_size}, flattened.shape: {flattened.shape}")
        
        all_obs = torch.tensor(all_obs, device=self.device).float()
        if self.tdd_args["network"]["use_independent_nets"]:
            phi_y = self.tdd_model.s_encoder[agent_id](new_pos_abs)  # (batch_size, hidden_dim)가 결과다.
            phi_x = self.tdd_model.s_encoder[agent_id](all_obs)  # (batch_size, hidden_dim)
        else:
            phi_y = self.tdd_model.s_encoder(new_pos_abs)  # (batch_size, hidden_dim)
            phi_x = self.tdd_model.s_encoder(all_obs)  # (batch_size, hidden_dim)
        
        # 최적화: 배치 크기 제한으로 메모리 사용량 감소
        if batch_size > self.max_batch_size:
            entropy_terms = []
            for i in range(0, batch_size, self.max_batch_size):
                end_idx = min(i + self.max_batch_size, batch_size)
                phi_y_batch = phi_y[i:end_idx]
                
                # Historical data도 배치 크기에 맞게 조정
                phi_x_batch = phi_x[:min(self.max_batch_size, len(phi_x))]
                
                dists_batch = mrn_distance(phi_x_batch[:, None], phi_y_batch[None, :])
                
                # k 값 동적 조정
                k_value = min(self.default_k_value, phi_x_batch.shape[0] // 10)
                if k_value < 2:
                    k_value = 2
                
                _, indices_batch = torch.topk(dists_batch, k=k_value, largest=False, dim=0)
                nearest_neighbors_batch = indices_batch[1:, :]  # 첫번째 행은 아마 자기 자신일 가능성이 꽤 높다.
                
                neighbor_distances_batch = torch.gather(dists_batch, dim=0, index=nearest_neighbors_batch)
                neighbor_distances_batch = neighbor_distances_batch.T
                
                normalized_distances_batch = neighbor_distances_batch / (neighbor_distances_batch.max() + 1e-8)
                sum_distances_batch = torch.sum(normalized_distances_batch, dim=1)
                sum_distances_batch = torch.clamp(sum_distances_batch, max=1e6)
                entropy_term_batch = torch.log(1 + (1/k_value) * sum_distances_batch)
                
                entropy_terms.append(entropy_term_batch)
            
            entropy_term = torch.cat(entropy_terms)
        else:
            # 기존 로직 (작은 배치 크기)
            dists = mrn_distance(phi_x[:, None], phi_y[None, :])
            
            # k 값 동적 조정
            k_value = min(self.default_k_value, phi_x.shape[0] // 10)
            if k_value < 2:
                k_value = 2
            
            _, indices = torch.topk(dists, k=k_value, largest=False, dim=0)
            nearest_neighbors = indices[1:, :]
            
            neighbor_distances = torch.gather(dists, dim=0, index=nearest_neighbors)
            neighbor_distances = neighbor_distances.T
            normalized_distances = neighbor_distances / (neighbor_distances.max() + 1e-8)
            sum_distances = torch.sum(normalized_distances, dim=1)
            sum_distances = torch.clamp(sum_distances, max=1e6)
            entropy_term = torch.log(1 + (1/k_value) * (sum_distances ** self.num_agents))
        
        # 성능 모니터링을 위한 로깅 추가 (파일에만 기록, step 기준)
        if step is not None and step % 1000 == 0:  # 1000 스텝마다만 로깅
            # 로깅이 활성화된 경우에만 출력
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_logger_logging"]:
                logger.info(f"Step {step}: Processing batch_size: {batch_size}, historical_samples: {len(all_obs)}")
        
        return entropy_term # 여기서 540이 찍히네 왜와이?
        
    def calculate_decentral_state_entropy(self, new_pos, agent_id, step=None):
        batch_size = new_pos.shape[0]
        # 보통 1000, obs의 차원만큼 인풋이 들어올거다.
        rollout_buffer_agent = self.rollout_buffer.rollout_history[agent_id]    # (n_timesteps, n_cycles, (n_rollout_threads, {'obs': (2,), 'next_obs': (2,)})) ex/ (29, 800, (dict...))
        
        # 빈 버퍼 체크
        if len(rollout_buffer_agent) == 0 or len(rollout_buffer_agent[0]) == 0:
            raise ValueError("calculate_decentral_state_entropy 함수에서 rollout_buffer가 비어있습니다.")
        
        # 우선 new_pos의 절대좌표만 잘라내자
        if new_pos.shape[1] >= 4:  # 최소 4차원 이상인지 확인
            new_pos_abs = new_pos[:, 2:4]
        else:
            raise ValueError("new_pos의 차원이 부족합니다.")
        new_pos_abs = torch.tensor(new_pos_abs, device=self.device).float()
        
        # 가장 최근 스텝만 가져오기
        n_trajs = len(rollout_buffer_agent)
        n_cycles = len(rollout_buffer_agent[0])
        
        rollout_array = np.array(rollout_buffer_agent, dtype=object)
        factor = rollout_array[0][0]["obs"].shape[0]
        x_times_y = batch_size // factor + 1
        safe_batch_size_1 = int(np.ceil(x_times_y / n_trajs))
        safe_batch_size_2 = int(np.ceil(x_times_y / n_cycles))
        safe_batch_size = int(max(max(safe_batch_size_1, safe_batch_size_2), np.ceil(np.sqrt(x_times_y))))
        rollout_array = rollout_array[-safe_batch_size:, -safe_batch_size:]
        flattened = rollout_array.reshape(-1)
        
        # 빈 flattened 배열 체크
        if len(flattened) == 0:
            raise ValueError("calculate_decentral_state_entropy 함수에서 flattened가 비어있습니다.")
        
        all_obs_temp = np.array([buffer['obs'] for buffer in flattened])    # (엄청여러개, 2)
        all_obs_temp = all_obs_temp.reshape(-1, 2)
        all_obs = all_obs_temp
        
        # 최적화: Historical data 샘플링 제한
        if all_obs.shape[0] > self.max_historical_samples:
            # 가장 최근 데이터부터 MAX_HISTORICAL_SAMPLES만큼 선택
            all_obs = all_obs[-self.max_historical_samples:]
        elif batch_size <= all_obs.shape[0]:
            indices = np.random.choice(all_obs.shape[0], batch_size, replace=False)
            all_obs = all_obs[indices]
        else:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_logger_logging"]:
                logger.warning(f"batch_size({batch_size})가 all_obs.shape[0]({all_obs.shape[0]})보다 큽니다. new_pos_abs를 all_obs.shape[0]만큼 잘라냅니다.")
            new_pos_abs = new_pos_abs[:all_obs.shape[0]]
            if batch_size != all_obs.shape[0]:
                raise ValueError(f"batch_size({batch_size})와 all_obs.shape[0]({all_obs.shape[0]})가 다릅니다. 버퍼가 충분하지 않습니다.")
        
        all_obs = torch.tensor(all_obs, device=self.device).float()
        if self.tdd_args["network"]["use_independent_nets"]:
            phi_y = self.tdd_model.s_encoder[agent_id](new_pos_abs)  # (batch_size, hidden_dim)가 결과다.
            phi_x = self.tdd_model.s_encoder[agent_id](all_obs)  # (batch_size, hidden_dim)
        else:
            phi_y = self.tdd_model.s_encoder(new_pos_abs)  # (batch_size, hidden_dim)
            phi_x = self.tdd_model.s_encoder(all_obs)  # (batch_size, hidden_dim)
        
        # 최적화: 배치 크기 제한으로 메모리 사용량 감소
        if batch_size > self.max_batch_size:
            entropy_terms = []
            for i in range(0, batch_size, self.max_batch_size):
                end_idx = min(i + self.max_batch_size, batch_size)
                phi_y_batch = phi_y[i:end_idx]
                
                # Historical data도 배치 크기에 맞게 조정
                phi_x_batch = phi_x[:min(self.max_batch_size, len(phi_x))]
                
                dists_batch = mrn_distance(phi_x_batch[:, None], phi_y_batch[None, :])
                
                # k 값 동적 조정
                k_value = min(self.default_k_value, phi_x_batch.shape[0] // 10)
                if k_value < 2:
                    k_value = 2
                
                _, indices_batch = torch.topk(dists_batch, k=k_value, largest=False, dim=0)
                nearest_neighbors_batch = indices_batch[1:, :]  # 첫번째 행은 아마 자기 자신일 가능성이 꽤 높다.
                
                neighbor_distances_batch = torch.gather(dists_batch, dim=0, index=nearest_neighbors_batch)
                neighbor_distances_batch = neighbor_distances_batch.T
                
                normalized_distances_batch = neighbor_distances_batch / (neighbor_distances_batch.max() + 1e-8)
                sum_distances_batch = torch.sum(normalized_distances_batch, dim=1)
                sum_distances_batch = torch.clamp(sum_distances_batch, max=1e6)
                entropy_term_batch = torch.log(1 + (1/k_value) * (sum_distances_batch ** self.num_agents))
                
                entropy_terms.append(entropy_term_batch)
            
            entropy_term = torch.cat(entropy_terms)
        else:
            # 기존 로직 (작은 배치 크기)
            dists = mrn_distance(phi_x[:, None], phi_y[None, :])
            
            # k 값 동적 조정
            k_value = min(self.default_k_value, phi_x.shape[0] // 10)
            if k_value < 2:
                k_value = 2
            
            _, indices = torch.topk(dists, k=k_value, largest=False, dim=0)
            nearest_neighbors = indices[1:, :]
            
            neighbor_distances = torch.gather(dists, dim=0, index=nearest_neighbors)
            neighbor_distances = neighbor_distances.T
            normalized_distances = neighbor_distances / (neighbor_distances.max() + 1e-8)
            sum_distances = torch.sum(normalized_distances, dim=1)
            sum_distances = torch.clamp(sum_distances, max=1e6)
            entropy_term = torch.log(1 + (1/k_value) * (sum_distances ** self.num_agents))
        
        # 성능 모니터링을 위한 로깅 추가 (파일에만 기록, step 기준)
        if step is not None and step % 1000 == 0:  # 1000 스텝마다만 로깅
            # 로깅이 활성화된 경우에만 출력
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_logger_logging"]:
                logger.info(f"Step {step}: Processing batch_size: {batch_size}, historical_samples: {len(all_obs)}")
        
        return entropy_term
    
    def plot_distance_map(self, start_pos, map_size, agent_id, landmarks, obstacles, agents_input=None, suffix=None, step=None):
        """목표 지점으로부터의 거리를 시각화합니다.
        Args:
            start_pos: (tuple) 시작 위치 (x, y)
            landmarks: (list) 랜드마크 정보 리스트, 각 요소는 {'position': (x, y), 'size': size} 형태
            obstacles: (list) 장애물 정보 리스트, 각 요소는 {'position': (x, y), 'size': size} 형태
            map_size: (int) 맵의 크기
            agent_id: (int) 에이전트 ID
            suffix: (str, optional) 파일명에 추가할 접미사
            step: (int, optional) 현재 학습 스텝
        """
        # 모든 가능한 위치 생성
        x = np.linspace(-map_size, map_size, 100)   # -map_size ~ map_size 사이의 100개의 점
        y = np.linspace(-map_size, map_size, 100)
        X, Y = np.meshgrid(x, y)    # 100x100 크기의 그리드 생성. x와 y의 모든 조합을 포함
        positions = np.stack([X.flatten(), Y.flatten()], axis=1)    # 100x2 크기의 행렬로 변환
        
        # 목표 위치를 텐서로 변환
        if agents_input is None:
            start_input_tensor = torch.tensor(start_pos, device=self.device).float()  # torch.Size([2])
        else:
            start_input_tensor = torch.tensor(agents_input, device=self.device).float()  # torch.Size([obs_dim])
        
        # 배치 크기 설정 (더 작게 조정)
        batch_size = 100  # 한 번에 처리할 점의 수
        n_positions = len(positions)
        dists = np.zeros(n_positions)
        
        with torch.no_grad():
            # 배치 단위로 처리
            for i in range(0, n_positions, batch_size): # n_positions = 10000, batch_size = 100
                end_idx = min(i + batch_size, n_positions)
                batch_positions = positions[i:end_idx]  # 100개의 위치 (100, 2)
                
                # 현재 배치의 위치와 목표 위치를 인코딩
                positions_tensor = torch.from_numpy(batch_positions).to(self.device).float()
                start_input_tensor_batch = start_input_tensor.unsqueeze(0).repeat(len(batch_positions), 1)
                
                # 인코딩 수행
                if self.tdd_args["network"]["use_independent_nets"]:
                    phi_g = self.tdd_model.s_encoder[agent_id](positions_tensor)    # (100, 32)
                    phi_start = self.tdd_model.s_encoder[agent_id](start_input_tensor_batch)    # (100, 32)
                else:
                    phi_g = self.tdd_model.s_encoder(positions_tensor)    # (100, 32)
                    phi_start = self.tdd_model.s_encoder(start_input_tensor_batch)    # (100, 32)
                
                batch_dists = mrn_distance(phi_start[:, None], phi_g[None, :])  #(100, 10)
                dists[i:end_idx] = torch.diag(batch_dists).cpu().numpy().squeeze()
                
                # 메모리 해제
                del phi_g, phi_start, batch_dists
                torch.cuda.empty_cache()
            
            # 거리 맵 생성
            dist_map = dists.reshape(X.shape)
            
            # 시각화
            plt.figure(figsize=(10, 8))
            im = plt.imshow(dist_map, extent=[-map_size, map_size, -map_size, map_size], 
                          origin='lower', cmap='viridis')
            plt.colorbar(im, label='Distance from Start')
            
            # 목표 위치 표시
            plt.plot(start_pos[0], start_pos[1], 'r*', markersize=15, label='Start')
            
            # 랜드마크 표시
            for landmark in landmarks:
                plt.plot(landmark['position'][0], landmark['position'][1], 'g*', markersize=15, label='Landmark')
            
            # 장애물 표시
            for obstacle in obstacles:
                obstacle_circle = plt.Circle(
                    (obstacle['position'][0], obstacle['position'][1]),
                    obstacle['size'],
                    facecolor='red', alpha=0.3,
                    label='Wall'
                )
                plt.gca().add_patch(obstacle_circle)
            
            plt.title(f'Distance Map from Start (with Landmarks and Wall) - {suffix}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            
            # 저장
            if step is not None:
                save_dir = os.path.join(self.save_dir, 'distance_map', f'step_{step}')
                os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
                save_path = os.path.join(save_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{suffix}.png')
            else:
                save_path = os.path.join(self.save_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{suffix}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

class WM_Runner:
    def __init__(self, obs_dim, action_spaces, algo_args, env_args, tdd_args):
        self.algo_args = algo_args
        self.env_args = env_args
        self.tdd_args = tdd_args
        self.num_agents = env_args["N"]
        self.max_episode_len = env_args["max_cycles"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.wm_buffer = WM_RolloutBuffer(
                {**self.tdd_args, **env_args},
                obs_dim,
                action_spaces,
                self.env_args["N"],
                self.algo_args["train"]["n_rollout_threads"],
                self.device
        )
        self.wm_ls = [
            DreamerWorldModel(obs_dim, action_spaces, self.tdd_args["wm"]).to(self.device) for _ in range(self.num_agents)
        ]
        self.target_wm_ls = [
            DreamerWorldModel(obs_dim, action_spaces, self.tdd_args["wm"]).to(self.device) for _ in range(self.num_agents)
        ]
        
        # 상태 저장을 위한 변수들 추가
        self.prev_states = [None for _ in range(self.num_agents)]  # 각 에이전트별 이전 상태 저장
        self.episode_step = 0  # 현재 에피소드 스텝
        
    def train_wm(self, rollout_buffer, soft_update=True):
        batch, max_episode_len = rollout_buffer.sample(self.tdd_args["wm"]["batch_size"])
        
        batch_o = batch['obs_n'].to(self.device)    # shape: (batch_size, n_timesteps + 1, n_agents, obs_dim)
        batch_a_before = batch['a_n_before'].to(self.device)
        batch_active = batch['active'].to(self.device)  # shape: (batch_size, n_timesteps, 1)
        
        batch_size, n_timesteps_plus_1, n_agents, obs_dim = batch_o.shape
        
        is_first = torch.zeros(batch_size, n_timesteps_plus_1, n_agents, device=self.device)  # shape: (batch_size, n_timesteps + 1, n_agents)
        is_first[:, 0, :] = 1.0
        
        batch_o = batch_o.permute(0, 2, 1, 3) # shape: (batch_size, n_agents, n_timesteps + 1, obs_dim)
        batch_a_before = batch_a_before.permute(0, 2, 1, 3) # shape: (batch_size, n_agents, n_timesteps + 1, action_dim)
        is_first = is_first.permute(0, 2, 1)   # shape: (batch_size, n_agents, n_timesteps + 1)
        
        batch_active = batch_active.expand(-1, -1, n_agents)  # shape: (batch_size, n_timesteps, n_agents)
        batch_active = torch.cat([batch_active, batch_active[:, -1:, :]], dim=1)  # shape: (batch_size, n_timesteps + 1, n_agents)
        
        metrics_ls = []
        for agent_id in range(self.num_agents):
            wm_data_dict = {
                'vector_obs': batch_o[:, agent_id, :, :],
                'action': batch_a_before[:, agent_id, :, :],
                'is_first': is_first[:, agent_id, :],
                'mask': batch_active[:, :, agent_id] # shape: (batch_size, n_timesteps + 1)
            }
            _, _, metrics = self.wm_ls[agent_id]._train(wm_data_dict)
            metrics = {f'wm/{k}':np.mean(v) for k,v in metrics.items()}
            metrics_ls.append(metrics)
        
        if soft_update:
            for agent_id in range(self.num_agents):
                self.soft_update_params(self.wm_ls[agent_id], self.target_wm_ls[agent_id], self.tdd_args["wm"]["dyna_tau"])
        else:
            for agent_id in range(self.num_agents):
                self.target_wm_ls[agent_id].load_state_dict(self.wm_ls[agent_id].state_dict())
            
        return metrics_ls

    def soft_update_params(self, net, target_net, tau):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def reset_episode_states(self):
        """에피소드 종료 시 상태 초기화"""
        self.prev_states = [None for _ in range(self.num_agents)]
        self.episode_step = 0
    
    def compute_wm_int_rew(self, obs, new_obs, actions, is_eval=False, temp_wm_buffer=None, n_rollout_threads=None, step=None):
        """World Model 기반 intrinsic reward 계산 - 단일 스텝 처리 방식
        
        Args:
            obs: 현재 관찰 (n_rollout_threads, n_agents, obs_dim)
            new_obs: 새로운 관찰 (n_rollout_threads, n_agents, obs_dim)
            is_eval: 평가 모드 여부
            temp_wm_buffer: 임시 WM 버퍼
            n_rollout_threads: 롤아웃 스레드 수
            step: 현재 스텝 (로깅용)
            
        Returns:
            intrinsic_rewards: (n_rollout_threads, n_agents, 1) 형태의 intrinsic rewards
        """
        step %= self.max_episode_len
        
        # WM 버퍼에서 현재까지의 시퀀스 가져오기
        if temp_wm_buffer is None:
            # WM_RolloutBuffer에서 현재 에피소드 데이터 가져오기
            current_episode_data = self.wm_buffer.current_buffer
        else:
            current_episode_data = temp_wm_buffer
        
        if current_episode_data is None:
            # 버퍼가 비어있으면 기본값 반환
            return np.zeros((n_rollout_threads, self.num_agents, 1))
        
        with torch.no_grad():
            agent_obs = current_episode_data['obs_n'][:, :(step + 2), :, :]  # (n_rollout_threads, n_timesteps, n_agents, obs_dim)
            agent_actions_before = current_episode_data['a_n_before'][:, :(step + 2), :, :]  # (n_rollout_threads, n_timesteps, n_agents, action_dim)
            
            # 현재 스텝의 데이터 업데이트
            if step == 0:
                agent_obs[:, 0, :, :] = obs
                agent_obs[:, 1, :, :] = new_obs
                agent_actions_before[:, 1, :, :] = actions
            elif step > 0:
                if not (agent_obs[:, step, :, :] == obs).all():
                    raise ValueError(f"step({step})에서 obs와 agent_obs[:, step, :, :]가 다릅니다.")
                agent_obs[:, step + 1, :, :] = new_obs
                agent_actions_before[:, step + 1, :, :] = actions
            else:
                raise ValueError(f"step({step})이 0 미만입니다.")
            
            # 기존 방식과 새로운 방식의 결과를 비교하기 위한 변수들
            # old_post_results = []
            # old_prior_results = []
            # new_post_results = []
            # new_prior_results = []
            # old_kl_losses = []
            # new_kl_losses = []
            
            # 기존 방식 (전체 시퀀스 처리)
            # threads_o = torch.tensor(agent_obs, device=self.device, dtype=torch.float32).permute(0, 2, 1, 3) # shape: (n_rollout_threads, n_agents, n_timesteps + 1, obs_dim)
            # threads_a_before = torch.tensor(agent_actions_before, device=self.device, dtype=torch.float32).permute(0, 2, 1, 3) # shape: (n_rollout_threads, n_agents, n_timesteps + 1, action_dim)
            
            # threads_is_first = torch.zeros((n_rollout_threads, (step + 2), self.num_agents), device=self.device, dtype=torch.float32)
            # threads_is_first[:, 0, :] = 1.0
            # threads_is_first = threads_is_first.permute(0, 2, 1) # shape: (n_rollout_threads, n_agents, n_timesteps + 1)
            
            # 새로운 방식 (단일 스텝 처리)
            threads_o_step = torch.tensor(agent_obs[:, step:(step + 2), :, :], device=self.device, dtype=torch.float32).permute(0, 2, 1, 3) # shape: (n_rollout_threads, n_agents, 2, obs_dim)
            threads_a_before_step = torch.tensor(agent_actions_before[:, step:(step + 2), :, :], device=self.device, dtype=torch.float32).permute(0, 2, 1, 3)  # (n_rollout_threads, n_agents, 2, action_dim)
            threads_is_first_step = torch.zeros((n_rollout_threads, 2, self.num_agents), device=self.device, dtype=torch.float32)
            if step == 0:
                threads_is_first_step[:, 0, :] = 1.0
            threads_is_first_step = threads_is_first_step.permute(0, 2, 1) # shape: (n_rollout_threads, n_agents, 2)
            
            int_rew = []
            for agent_id in range(self.num_agents):
                # === 기존 방식 (전체 시퀀스 처리) ===
                # wm_data_dict_old = {
                #     'vector_obs': threads_o[:, agent_id, :, :],  # shape: (n_rollout_threads, n_timesteps + 1, obs_dim)
                #     'action': threads_a_before[:, agent_id, :, :],  # shape: (n_rollout_threads, n_timesteps + 1, action_dim)
                #     'is_first': threads_is_first[:, agent_id, :]  # shape: (n_rollout_threads, n_timesteps + 1)
                # }
                
                # embed_old = self.wm_ls[agent_id].encoder(wm_data_dict_old)  # embed_old의 shape: (n_rollout_threads, 2, embed_size)
                # # RSSM observe를 통한 prior와 posterior 계산 (현재까지의 시퀀스)
                # post_old, prior_old = self.wm_ls[agent_id].dynamics.observe(
                #     embed_old, 
                #     threads_a_before[:, agent_id, :, :], 
                #     threads_is_first[:, agent_id, :]
                # )
                
                # KL loss 계산 (새로운 방식 사용)
                kl_free = self.tdd_args["wm"]["kl_free"]
                dyn_scale = self.tdd_args["wm"]["dyn_scale"]
                rep_scale = self.tdd_args["wm"]["rep_scale"]
                
                # 기존 방식의 KL loss 계산 (비교용)
                # kl_loss_old, kl_value_old, dyn_loss_old, rep_loss_old = self.wm_ls[agent_id].dynamics.kl_loss(
                #     post_old, prior_old, kl_free, dyn_scale, rep_scale
                # )
                
                # === 새로운 방식 (단일 스텝 처리) ===
                # agent_obs_step = threads_o_step[:, agent_id, :, :]  # (n_rollout_threads, 2, obs_dim)
                # agent_obs_next_step = threads_o_step[:, agent_id, 1, :]  # (n_rollout_threads, obs_dim)
                # agent_action_before_step = threads_a_before_step[:, agent_id, 1, :]  # (n_rollout_threads, action_dim)
                # agent_is_first = is_first[:, agent_id]  # (n_rollout_threads,)
                
                # 인코더를 통한 임베딩 생성
                wm_data_dict_new = {
                    'vector_obs': threads_o_step[:, agent_id, :, :],  # (n_rollout_threads, 2, obs_dim)
                    'action': threads_a_before_step[:, agent_id, :, :],    # (n_rollout_threads, 2, action_dim)
                    'is_first': threads_is_first_step[:, agent_id]  # (n_rollout_threads,)
                }
                
                embed_step = self.wm_ls[agent_id].encoder(wm_data_dict_new)  # (n_rollout_threads, 2, embed_size)
                
                # 단일 스텝 observe 처리
                post_step, prior_step = self.wm_ls[agent_id].dynamics.observe_step(
                    self.prev_states[agent_id],  # 이전 상태
                    embed_step,                   # 현재 임베딩
                    threads_a_before_step[:, agent_id, :, :],           # 현재 액션
                    threads_is_first_step[:, agent_id]              # is_first 플래그
                )
                
                kl_loss, kl_value, dyn_loss, rep_loss = self.wm_ls[agent_id].dynamics.kl_loss(
                    post_step, prior_step, kl_free, dyn_scale, rep_scale
                )
                
                # intrinsic reward는 각 스레드별 KL loss (평균 내지 않음)
                # kl_loss shape: (n_rollout_threads,) - 각 스레드별 개별 KL loss
                if agent_id == 0:
                    int_rew = kl_loss.unsqueeze(1)  # (n_rollout_threads, 1)
                else:
                    int_rew = torch.cat([int_rew, kl_loss.unsqueeze(1)], dim=1)  # (n_rollout_threads, n_agents)
                
                # 결과 비교 및 저장
                # old_post_results.append(post_old)
                # old_prior_results.append(prior_old)
                # new_post_results.append(post_step)
                # new_prior_results.append(prior_step)
                # old_kl_losses.append(kl_loss_old)
                # new_kl_losses.append(kl_loss)
                
                # 다음 스텝을 위해 현재 상태 저장
                self.prev_states[agent_id] = post_step
            
            # 최종 차원을 (n_rollout_threads, n_agents, 1)로 맞춤
            int_rew = int_rew.unsqueeze(-1)  # (n_rollout_threads, n_agents, 1)
            
            # === 결과 비교 ===
            # if step % 1 == 0:  # 100 스텝마다 비교 (성능상의 이유로)
            #     self._compare_results(old_post_results, old_prior_results, 
            #                        new_post_results, new_prior_results, 
            #                        old_kl_losses, new_kl_losses, step)
        
        # 로깅 (선택적)
        if step is not None and step % 1000 == 0:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_logger_logging"]:
                avg_reward = np.mean(int_rew.cpu().numpy())
                logger.info(f"Step {step}: WM Intrinsic Reward = {avg_reward:.4f}, "
                          f"Agents = {self.num_agents}, Threads = {n_rollout_threads}")
        
        return np.array(int_rew.cpu().numpy())
    
    def _compare_results(self, old_post_results, old_prior_results, new_post_results, new_prior_results, old_kl_losses, new_kl_losses, step):
        """기존 방식과 새로운 방식의 결과를 비교합니다. 분포 파라미터 중심으로 비교합니다."""
        try:
            for agent_id in range(self.num_agents):
                old_post = old_post_results[agent_id]
                old_prior = old_prior_results[agent_id]
                new_post = new_post_results[agent_id]
                new_prior = new_prior_results[agent_id]
                old_kl_loss = old_kl_losses[agent_id]
                new_kl_loss = new_kl_losses[agent_id]
                
                # === 1. 분포 파라미터 비교 (같아야 함) ===
                logger.info(f"Step {step}, Agent {agent_id}: 분포 파라미터 비교")
                
                # Posterior 분포 파라미터 비교
                for param_key in ["mean", "std", "logit"]:
                    if param_key in old_post and param_key in new_post:
                        old_param = old_post[param_key]
                        new_param = new_post[param_key]
                        
                        # 마지막 타임스텝만 비교 (새로운 방식은 단일 스텝이므로)
                        if len(old_param.shape) > 1:
                            old_param_last = old_param[:, -1]  # 마지막 타임스텝
                        else:
                            old_param_last = old_param
                        
                        # 분포 파라미터 차이 계산
                        param_diff = torch.abs(old_param_last - new_param).max().item()
                        
                        if param_diff > 1e-6:
                            logger.warning(f"  Posterior '{param_key}': 차이 = {param_diff:.8f}")
                            logger.warning(f"    Old shape: {old_param.shape}, New shape: {new_param.shape}")
                            logger.warning(f"    Old values: {old_param_last[:3]}")
                            logger.warning(f"    New values: {new_param[:3]}")
                        else:
                            logger.info(f"  Posterior '{param_key}': 일치 (차이 = {param_diff:.8f})")
                
                # Prior 분포 파라미터 비교
                for param_key in ["mean", "std", "logit"]:
                    if param_key in old_prior and param_key in new_prior:
                        old_param = old_prior[param_key]
                        new_param = new_prior[param_key]
                        
                        if len(old_param.shape) > 1:
                            old_param_last = old_param[:, -1]
                        else:
                            old_param_last = old_param
                        
                        param_diff = torch.abs(old_param_last - new_param).max().item()
                        
                        if param_diff > 1e-6:
                            logger.warning(f"  Prior '{param_key}': 차이 = {param_diff:.8f}")
                        else:
                            logger.info(f"  Prior '{param_key}': 일치 (차이 = {param_diff:.8f})")
                
                # === 2. 결정적 상태 비교 (같아야 함) ===
                logger.info(f"Step {step}, Agent {agent_id}: 결정적 상태 비교")
                
                # Posterior deter 비교
                if "deter" in old_post and "deter" in new_post:
                    old_deter = old_post["deter"]
                    new_deter = new_post["deter"]
                    
                    if len(old_deter.shape) > 1:
                        old_deter_last = old_deter[:, -1]
                    else:
                        old_deter_last = old_deter
                    
                    deter_diff = torch.abs(old_deter_last - new_deter).max().item()
                    
                    if deter_diff > 1e-6:
                        logger.warning(f"  Posterior deter: 차이 = {deter_diff:.8f}")
                    else:
                        logger.info(f"  Posterior deter: 일치 (차이 = {deter_diff:.8f})")
                
                # Prior deter 비교
                if "deter" in old_prior and "deter" in new_prior:
                    old_deter = old_prior["deter"]
                    new_deter = new_prior["deter"]
                    
                    if len(old_deter.shape) > 1:
                        old_deter_last = old_deter[:, -1]
                    else:
                        old_deter_last = old_deter
                    
                    deter_diff = torch.abs(old_deter_last - new_deter).max().item()
                    
                    if deter_diff > 1e-6:
                        logger.warning(f"  Prior deter: 차이 = {deter_diff:.8f}")
                    else:
                        logger.info(f"  Prior deter: 일치 (차이 = {deter_diff:.8f})")
                
                # === 3. 확률적 상태 비교 (다를 수 있음 - 정상) ===
                logger.info(f"Step {step}, Agent {agent_id}: 확률적 상태 비교 (샘플링으로 인해 다를 수 있음)")
                
                if "stoch" in old_post and "stoch" in new_post:
                    old_stoch = old_post["stoch"]
                    new_stoch = new_post["stoch"]
                    
                    if len(old_stoch.shape) > 1:
                        old_stoch_last = old_stoch[:, -1]
                    else:
                        old_stoch_last = old_stoch
                    
                    stoch_diff = torch.abs(old_stoch_last - new_stoch).max().item()
                    
                    # 확률적 샘플링이므로 차이가 있어도 정상
                    logger.info(f"  Posterior stoch: 차이 = {stoch_diff:.8f} (샘플링으로 인한 차이)")
                
                if "stoch" in old_prior and "stoch" in new_prior:
                    old_stoch = old_prior["stoch"]
                    new_stoch = new_prior["stoch"]
                    
                    if len(old_stoch.shape) > 1:
                        old_stoch_last = old_stoch[:, -1]
                    else:
                        old_stoch_last = old_stoch
                    
                    stoch_diff = torch.abs(old_stoch_last - new_stoch).max().item()
                    
                    logger.info(f"  Prior stoch: 차이 = {stoch_diff:.8f} (샘플링으로 인한 차이)")
                
                # === 4. KL Loss 비교 (분포 파라미터 기반) ===
                logger.info(f"Step {step}, Agent {agent_id}: KL Loss 비교")
                
                if len(old_kl_loss.shape) > 1:
                    old_kl_last = old_kl_loss[:, -1]  # 마지막 타임스텝
                else:
                    old_kl_last = old_kl_loss
                
                kl_diff = torch.abs(old_kl_last - new_kl_loss).max().item()
                
                # KL Loss는 분포 파라미터에 기반하므로 비슷해야 함
                if kl_diff > 1e-4:  # KL Loss는 약간의 차이 허용
                    logger.warning(f"  KL Loss: 차이 = {kl_diff:.8f}")
                    logger.warning(f"    Old KL shape: {old_kl_loss.shape}, New KL shape: {new_kl_loss.shape}")
                    logger.warning(f"    Old KL last step: {old_kl_last[:3]}")
                    logger.warning(f"    New KL values: {new_kl_loss[:3]}")
                else:
                    logger.info(f"  KL Loss: 일치 (차이 = {kl_diff:.8f})")
                
                # === 5. 분포 통계 비교 ===
                logger.info(f"Step {step}, Agent {agent_id}: 분포 통계 비교")
                
                try:
                    # Posterior 분포의 통계적 특성 비교
                    if "mean" in old_post and "std" in old_post and "mean" in new_post and "std" in new_post:
                        old_mean = old_post["mean"]
                        old_std = old_post["std"]
                        new_mean = new_post["mean"]
                        new_std = new_post["std"]
                        
                        if len(old_mean.shape) > 1:
                            old_mean_last = old_mean[:, -1]
                            old_std_last = old_std[:, -1]
                        else:
                            old_mean_last = old_mean
                            old_std_last = old_std
                        
                        # 분포의 통계적 특성 비교
                        mean_diff = torch.abs(old_mean_last - new_mean).max().item()
                        std_diff = torch.abs(old_std_last - new_std).max().item()
                        
                        if mean_diff > 1e-6:
                            logger.warning(f"  Posterior mean: 차이 = {mean_diff:.8f}")
                        else:
                            logger.info(f"  Posterior mean: 일치 (차이 = {mean_diff:.8f})")
                        
                        if std_diff > 1e-6:
                            logger.warning(f"  Posterior std: 차이 = {std_diff:.8f}")
                        else:
                            logger.info(f"  Posterior std: 일치 (차이 = {std_diff:.8f})")
                
                except Exception as e:
                    logger.warning(f"  분포 통계 비교 중 오류: {e}")
                
                logger.info(f"Step {step}, Agent {agent_id}: 비교 완료\n")
                            
        except Exception as e:
            logger.error(f"Error during result comparison: {e}")
            import traceback
            traceback.print_exc()