import torch
import numpy as np
from harl.common.buffers.tdd_rollout_buffer import RolloutBuffer
from harl.algorithms.representation.tdd import TDDModel, mrn_distance
import matplotlib.pyplot as plt
import os
import time
import logging
logger = logging.getLogger(__name__)

# class RunningMeanStd:
#     def __init__(self, epsilon=1e-4, shape=(), momentum=None):
#         self.mean = np.zeros(shape, dtype=np.float64)
#         self.var = np.ones(shape, dtype=np.float64)
#         self.count = epsilon
#         self.momentum = momentum

#     def update(self, x):
#         batch_mean = np.mean(x, axis=0)
#         batch_var = np.var(x, axis=0)
#         batch_count = len(x)
        
#         if self.momentum is None:
#             delta = batch_mean - self.mean
#             tot_count = self.count + batch_count
#             new_mean = self.mean + delta * batch_count / tot_count
#             m_a = self.var * self.count
#             m_b = batch_var * batch_count
#             M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
#             new_var = M2 / tot_count
#             new_count = tot_count
#         else:
#             new_mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
#             new_var = self.momentum * self.var + (1 - self.momentum) * batch_var
#             new_count = self.count + batch_count

#         self.mean = new_mean
#         self.var = new_var
#         self.count = new_count

#     @property
#     def std(self):
#         return np.sqrt(self.var + 1e-8)


class TddRunner:  # tdd_args가 none이 아닐때만 호출 됨
    def __init__(self, n_rollout_threads, num_agents, observation_space, tdd_args=None, save_dir=None):
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
        
        if self.tdd_args["network"]["use_central_SD"]:
            self.tdd_model = TDDModel(self.tdd_args, self.num_agents, self.num_agents * (2 + 2 + 2 * self.num_agents + 4 * (self.num_agents - 1)), self.device, run_dir=save_dir)
        else:
            self.tdd_model = TDDModel(self.tdd_args, self.num_agents, 2, self.device, run_dir=save_dir)
        
        self.max_historical_samples = self.tdd_args["train"]["max_historical_samples"]
        self.max_batch_size = self.tdd_args["train"]["max_batch_size"]
        self.default_k_value = self.tdd_args["train"]["default_k_value"]
        
        # 보상 계산 관련 변수 초기화
        self.reward_update_counter = 0
        self.last_reward_update = None
        self.prev_obs = None
        
        # 리워드 정규화를 위한 변수들
        # self.int_rew_norm = self.tdd_args.get("int_rew_norm", 0)  # 0: 정규화 없음, 1: 정규화
        # self.int_rew_clip = self.tdd_args.get("int_rew_clip", 0.0)  # 클리핑 값
        # self.int_rew_eps = self.tdd_args.get("int_rew_eps", 1e-8)  # 수치 안정성을 위한 작은 값
        # self.int_rew_momentum = self.tdd_args.get("int_rew_momentum", None)  # 모멘텀 값
        # self.int_rew_stats = RunningMeanStd(momentum=self.int_rew_momentum)
        
        self.rollout_buffer = RolloutBuffer(
                {**self.tdd_args["network"], **self.tdd_args["train"], **self.tdd_args["tdd"]},
                self.observation_space,
                self.num_agents,
                self.n_rollout_threads
        )
    
    def update_tdd_model(self, is_warm_up=False):
        if len(self.rollout_buffer.rollout_history[0]) == 0:
            raise ValueError("rollout_buffer.rollout_history가 비어있습니다.")
        self.tdd_model.update(self.rollout_buffer.rollout_history, is_warm_up=is_warm_up)
        
    # def normalize_rewards(self, rewards):
    #     """리워드를 정규화합니다."""
    #     if self.int_rew_norm == 0:
    #         return rewards
            
    #     # 리워드 통계 업데이트
    #     self.int_rew_stats.update(rewards.reshape(-1))
        
    #     # 정규화
    #     normalized_rewards = (rewards - self.int_rew_stats.mean) / (self.int_rew_stats.std + self.int_rew_eps)
        
    #     # 클리핑
    #     if self.int_rew_clip > 0:
    #         normalized_rewards = np.clip(normalized_rewards, -self.int_rew_clip, self.int_rew_clip)
            
    #     return normalized_rewards

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
                if self.tdd_args["network"]["use_central_SD"]:
                    prev_states.append(state_dict['share_obs'])
                else:
                    prev_states.append(state_dict['obs'])
                prev_states.append(pos[agent_id])
                if self.tdd_args["network"]["use_central_SD"]:
                    if not np.allclose(pos[agent_id], state_dict['next_share_obs'], rtol=1e-5, atol=1e-5):
                        raise AssertionError(f"rollout_states의 마지막 상태의 next_share_obs({state_dict['next_share_obs']})가 현재 상태({pos[agent_id]})와 다른 경우입니다.")
                else:
                    if not np.allclose(pos[agent_id], state_dict['next_obs'], rtol=1e-5, atol=1e-5):
                        raise AssertionError(f"rollout_states의 마지막 상태의 next_obs({state_dict['next_obs']})가 현재 상태({pos[agent_id]})와 다른 경우입니다.")
            else:
                if self.tdd_args["network"]["use_central_SD"]:
                    prev_states.append(state_dict['share_obs'])
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
                    int_rew[agent_id].append(min_dists.cpu().numpy())
                    
                    # 에이전트 간 상호작용 고려
                    if self.tdd_args["train"]["coeff_inter_agent_int_rew"] != 0:
                        other_min_dists_ls = []
                        for other_agent_id in range(self.num_agents):
                            if other_agent_id != agent_id:
                                other_prev_states = self._get_prev_states(current_rollout_states, other_agent_id, pos, is_eval)
                                other_min_dists = self._compute_mrn_distance(new_pos[agent_id], other_prev_states, other_agent_id, n_rollout_threads)
                                other_min_dists_ls.append(other_min_dists.cpu().numpy())
                        
                        other_min_dists_final = np.min(np.array(other_min_dists_ls), axis=0)
                        int_rew[agent_id] += other_min_dists_final * self.tdd_args["train"]["coeff_inter_agent_int_rew"]
                else:
                    int_rew[agent_id].append(np.zeros(n_rollout_threads))
        
        return np.array(int_rew).transpose(2, 0, 1)  # 저렇게 바꾸면 -> (n_rollout_threads, n_agents, 1)
    
    def calculate_central_state_entropy(self, new_pos, agent_id, step=None):
        batch_size = new_pos.shape[0]
        # 보통 1000, obs의 차원만큼 인풋이 들어올거다.
        rollout_buffer_all = self.rollout_buffer.rollout_history[:]    # (n_agents, n_trajs, max_cycles, (n_rollout_threads, {'obs': (2,), 'next_obs': (2,)})) ex/ (3, 29, 800, (dict...))
        
        # 빈 버퍼 체크
        if len(rollout_buffer_all) == 0 or len(rollout_buffer_all[0]) == 0:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.warning("rollout_buffer가 비어있습니다. 기본 엔트로피 값을 반환합니다.")
            # 기본 엔트로피 값으로 작은 양수 값 반환 (0보다는 크지만 매우 작은 값)
            return torch.ones(batch_size, device=self.device) * 0.1
        
        # 우선 new_pos의 절대좌표만 잘라내자
        if new_pos.shape[1] >= 4:  # 최소 4차원 이상인지 확인
            new_pos_abs = new_pos[:, 2:4]
        else:
            # 차원이 부족한 경우 전체를 사용
            raise ValueError("new_pos의 차원이 부족합니다.")
        new_pos_abs = torch.tensor(new_pos_abs, device=self.device).float()
        
        # 가장 최근 스텝만 가져오기
        n_agents = len(rollout_buffer_all)
        n_trajs = len(rollout_buffer_all[0])
        n_cycles = len(rollout_buffer_all[0][0])
        
        total_steps = n_agents * n_trajs * n_cycles
        start_idx = max(0, total_steps - 10 * batch_size)
        
        # numpy array로 변환하고 reshape
        rollout_array = np.array(rollout_buffer_all, dtype=object)
        # rollout_array[0]에서 batch_size // n_agents만큼 뒤에서부터 잘라내기
        # 안전한 인덱싱을 위해 최소값 사용
        safe_batch_size = min(batch_size // n_agents, len(rollout_array[0]) if len(rollout_array) > 0 else 0)
        if safe_batch_size > 0:
            rollout_array = rollout_array[:, -safe_batch_size:, -safe_batch_size:]
        flattened = rollout_array.reshape(-1)   # 위 작업 진행 안 했다면 0~799번째까지 연속된 궤적이고 800번째부터 다시 초기화된다. 그렇게 3200이 rollout_array의 [0][5][0]이 됩니다만... 그렇다는건 flattened[3200]까진 전부 0번째 에이전트의 궤적이란 말이다;;
        
        # 빈 flattened 배열 체크
        if len(flattened) == 0:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.warning("flattened 배열이 비어있습니다. 기본 엔트로피 값을 반환합니다.")
            # 기본 엔트로피 값으로 작은 양수 값 반환 (0보다는 크지만 매우 작은 값)
            return torch.ones(batch_size, device=self.device) * 0.1
        
        all_obs_temp = np.array([buffer['obs'] for buffer in flattened])    # (엄청여러개, 2)
        all_obs_temp = all_obs_temp.reshape(-1, 2)
        all_obs = all_obs_temp[start_idx:]
        
        # 최적화: Historical data 샘플링 제한
        if all_obs.shape[0] > self.max_historical_samples:
            # 가장 최근 데이터부터 MAX_HISTORICAL_SAMPLES만큼 선택
            all_obs = all_obs[-self.max_historical_samples:]
        elif batch_size <= all_obs.shape[0]:
            indices = np.random.choice(all_obs.shape[0], batch_size, replace=False)
            all_obs = all_obs[indices]
        else:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.warning(f"batch_size({batch_size})가 all_obs.shape[0]({all_obs.shape[0]})보다 큽니다. new_pos_abs를 all_obs.shape[0]만큼 잘라냅니다.")
            new_pos_abs = new_pos_abs[:all_obs.shape[0]]
            batch_size = all_obs.shape[0]  # batch_size도 조정
        
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
            entropy_term = torch.log(1 + (1/k_value) * sum_distances)
        
        # 성능 모니터링을 위한 로깅 추가 (파일에만 기록, step 기준)
        if step is not None and step % 1000 == 0:  # 1000 스텝마다만 로깅
            # 로깅이 활성화된 경우에만 출력
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.info(f"Step {step}: Processing batch_size: {batch_size}, historical_samples: {len(all_obs)}")
        
        return entropy_term
        
    def calculate_decentral_state_entropy(self, new_pos, agent_id, step=None):
        batch_size = new_pos.shape[0]
        # 보통 1000, obs의 차원만큼 인풋이 들어올거다.
        rollout_buffer_agent = self.rollout_buffer.rollout_history[agent_id]    # (n_timesteps, n_cycles, (n_rollout_threads, {'obs': (2,), 'next_obs': (2,)})) ex/ (29, 800, (dict...))
        
        # 빈 버퍼 체크
        if len(rollout_buffer_agent) == 0 or len(rollout_buffer_agent[0]) == 0:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.warning(f"agent {agent_id}의 rollout_buffer가 비어있습니다. 기본 엔트로피 값을 반환합니다.")
            # 기본 엔트로피 값으로 작은 양수 값 반환 (0보다는 크지만 매우 작은 값)
            return torch.ones(batch_size, device=self.device) * 0.1
        
        # 우선 new_pos의 절대좌표만 잘라내자
        if new_pos.shape[1] >= 4:  # 최소 4차원 이상인지 확인
            new_pos_abs = new_pos[:, 2:4]
        else:
            raise ValueError("new_pos의 차원이 부족합니다.")
        new_pos_abs = torch.tensor(new_pos_abs, device=self.device).float()
        
        # 가장 최근 스텝만 가져오기
        n_trajs = len(rollout_buffer_agent)
        n_cycles = len(rollout_buffer_agent[0])
        
        total_steps = n_trajs * n_cycles
        start_idx = max(0, total_steps - 10 * batch_size)
        
        rollout_array = np.array(rollout_buffer_agent, dtype=object)
        rollout_array = rollout_array[-batch_size:, -batch_size:]
        flattened = rollout_array.reshape(-1)
        
        # 빈 flattened 배열 체크
        if len(flattened) == 0:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.warning(f"agent {agent_id}의 flattened 배열이 비어있습니다. 기본 엔트로피 값을 반환합니다.")
            # 기본 엔트로피 값으로 작은 양수 값 반환 (0보다는 크지만 매우 작은 값)
            return torch.ones(batch_size, device=self.device) * 0.1
        
        all_obs_temp = np.array([buffer['obs'] for buffer in flattened])    # (엄청여러개, 2)
        all_obs_temp = all_obs_temp.reshape(-1, 2)
        all_obs = all_obs_temp[start_idx:]
        
        # 최적화: Historical data 샘플링 제한
        if all_obs.shape[0] > self.max_historical_samples:
            # 가장 최근 데이터부터 MAX_HISTORICAL_SAMPLES만큼 선택
            all_obs = all_obs[-self.max_historical_samples:]
        elif batch_size <= all_obs.shape[0]:
            indices = np.random.choice(all_obs.shape[0], batch_size, replace=False)
            all_obs = all_obs[indices]
        else:
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.warning(f"batch_size({batch_size})가 all_obs.shape[0]({all_obs.shape[0]})보다 큽니다. new_pos_abs를 all_obs.shape[0]만큼 잘라냅니다.")
            new_pos_abs = new_pos_abs[:all_obs.shape[0]]
            batch_size = all_obs.shape[0]  # batch_size도 조정
        
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
            entropy_term = torch.log(1 + (1/k_value) * sum_distances)
        
        # 성능 모니터링을 위한 로깅 추가 (파일에만 기록, step 기준)
        if step is not None and step % 1000 == 0:  # 1000 스텝마다만 로깅
            # 로깅이 활성화된 경우에만 출력
            if self.tdd_args is not None and "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                logger.info(f"Step {step}: Processing batch_size: {batch_size}, historical_samples: {len(all_obs)}")
        
        return entropy_term
    
    def plot_distance_map(self, start_pos, map_size, agent_id, landmarks, obstacles, suffix=None, step=None):
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
        start_pos_tensor = torch.tensor(start_pos, device=self.device).float()  # (2,)
        
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
                start_pos_tensor_batch = start_pos_tensor.unsqueeze(0).repeat(len(batch_positions), 1)
                
                # 인코딩 수행
                if self.tdd_args["network"]["use_independent_nets"]:
                    phi_g = self.tdd_model.s_encoder[agent_id](positions_tensor)    # (100, 32)
                    phi_start = self.tdd_model.s_encoder[agent_id](start_pos_tensor_batch)    # (100, 32)
                else:
                    phi_g = self.tdd_model.s_encoder(positions_tensor)    # (100, 32)
                    phi_start = self.tdd_model.s_encoder(start_pos_tensor_batch)    # (100, 32)
                
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
                save_dir = os.path.join(self.save_dir, f'step_{step}')
                os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
                save_path = os.path.join(save_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{suffix}.png')
            else:
                save_path = os.path.join(self.save_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{suffix}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()