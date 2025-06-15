import torch
import numpy as np
from harl.common.buffers.tdd_rollout_buffer import RolloutBuffer
from harl.algorithms.representation.tdd import TDDModel, mrn_distance
import matplotlib.pyplot as plt
import os
import time


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), momentum=None):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.momentum = momentum

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        
        if self.momentum is None:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
            new_var = M2 / tot_count
            new_count = tot_count
        else:
            new_mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            new_var = self.momentum * self.var + (1 - self.momentum) * batch_var
            new_count = self.count + batch_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class TddRunner:  # tdd_args가 none이 아닐때만 호출 됨
    def __init__(self, n_rollout_threads, num_agents, observation_space, tdd_args=None, log_dir=None):
        if tdd_args is None:
            print("TDD is disabled")
            return
        
        print("TDD is enabled")
        self.tdd_args = tdd_args
        self.log_dir = log_dir
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
        
        self.tdd_model = TDDModel(self.tdd_args, self.num_agents, 2, self.device, run_dir=log_dir)
        
        # 보상 계산 관련 변수 초기화
        self.reward_update_counter = 0
        self.last_reward_update = None
        self.prev_obs = None
        
        # 리워드 정규화를 위한 변수들
        self.int_rew_norm = self.tdd_args.get("int_rew_norm", 0)  # 0: 정규화 없음, 1: 정규화
        self.int_rew_clip = self.tdd_args.get("int_rew_clip", 0.0)  # 클리핑 값
        self.int_rew_eps = self.tdd_args.get("int_rew_eps", 1e-8)  # 수치 안정성을 위한 작은 값
        self.int_rew_momentum = self.tdd_args.get("int_rew_momentum", None)  # 모멘텀 값
        self.int_rew_stats = RunningMeanStd(momentum=self.int_rew_momentum)
        
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
        
    def normalize_rewards(self, rewards):
        """리워드를 정규화합니다."""
        if self.int_rew_norm == 0:
            return rewards
            
        # 리워드 통계 업데이트
        self.int_rew_stats.update(rewards.reshape(-1))
        
        # 정규화
        normalized_rewards = (rewards - self.int_rew_stats.mean) / (self.int_rew_stats.std + self.int_rew_eps)
        
        # 클리핑
        if self.int_rew_clip > 0:
            normalized_rewards = np.clip(normalized_rewards, -self.int_rew_clip, self.int_rew_clip)
            
        return normalized_rewards

    def _compute_mrn_distance(self, current_state, prev_states, agent_id, n_rollout_threads):
        """현재 상태와 이전 상태들 간의 MRN 거리를 계산합니다."""
        current_state = torch.tensor(current_state, device=self.device).float()
        phi_y = self.tdd_model.s_encoder[agent_id](current_state)
        
        prev_states = torch.tensor(np.array(prev_states), device=self.device).float()
        prev_states = prev_states.view(-1, prev_states.shape[-1])
        phi_x = self.tdd_model.s_encoder[agent_id](prev_states)
        
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
            current_rollout_states = temp_rollout_buffer    # (n_agents, {'obs': (2,), 'next_obs': (2,), 'dones': (1,)}) 여야함
        
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                # 현재 상태에 대한 내재적 보상 계산
                if any(current_rollout_states):
                    prev_states = self._get_prev_states(current_rollout_states, agent_id, pos, is_eval)
                    min_dists = self._compute_mrn_distance(new_pos[agent_id], prev_states, agent_id, n_rollout_threads)
                    int_rew[agent_id].append(min_dists.cpu().numpy())
                    
                    # 에이전트 간 상호작용 고려
                    if self.tdd_args["train"]["use_inter_agent_int_rew"]:
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
    
    def calculate_state_entropy(self, new_pos):
        batch_size = new_pos.shape[0]
        # 보통 1000, obs의 차원만큼 인풋이 들어올거다.
        rollout_buffer_all = self.rollout_buffer.rollout_history[:]    # (n_agents, n_timesteps, max_cycles, (n_rollout_threads, {'obs': (2,), 'next_obs': (2,)})) ex/ (3, 29, 800, (dict...))
        
        # 지금 저 new_pos는 agent_wise로 들어오긴 했다. 하지만 나는 모든 에이전트에 쌓인 rollout_buffer_all에 대해 mrn_distance를 계산해야겠다.
        
        """ 가장 최근에 쌓인 rollout_buffer에서 batch_size만큼 obs 가져오는게 좋아 보인다. """
        
        # 우선 new_pos의 절대좌표만 잘라내자
        new_pos_abs = new_pos[:, 2:4]
        new_pos_abs = torch.tensor(new_pos_abs, device=self.device).float()
        phi_y = self.tdd_model.s_encoder(new_pos_abs)  # (n_rollout_threads, hidden_dim)
        
        # 가장 최근 3000 스텝만 가져오기
        n_agents = len(rollout_buffer_all)
        n_timesteps = len(rollout_buffer_all[0])
        n_cycles = len(rollout_buffer_all[0][0])
        
        total_steps = n_agents * n_timesteps * n_cycles
        start_idx = max(0, total_steps - 10 * batch_size)
        
        # numpy array로 변환하고 reshape
        rollout_array = np.array(rollout_buffer_all, dtype=object)
        # rollout_array[0]에서 batch_size // n_agents만큼 뒤에서부터 잘라내기
        rollout_array = rollout_array[:, -batch_size // n_agents:, -batch_size // n_agents:]
        flattened = rollout_array.reshape(-1)   # 위 작업 진행 안 했다면 0~799번째까지 연속된 궤적이고 800번째부터 다시 초기화된다. 그렇게 3200이 rollout_array의 [0][5][0]이 됩니다만... 그렇다는건 flattened[3200]까진 전부 0번째 에이전트의 궤적이란 말이다;;
        
        all_obs_temp = np.array([buffer['obs'] for buffer in flattened])
        all_obs_temp = all_obs_temp.reshape(-1, 2)
        all_obs = all_obs_temp[start_idx:]
        
        if batch_size <= all_obs.shape[0]:
            indices = np.random.choice(all_obs.shape[0], batch_size, replace=False)
            all_obs = all_obs[indices]
        else:
            print(f"batch_size({batch_size})가 all_obs.shape[0]({all_obs.shape[0]})보다 큽니다. new_pos_abs를 all_obs.shape[0]만큼 잘라냅니다.")
            new_pos_abs = new_pos_abs[:all_obs.shape[0]]
            
        all_obs = torch.tensor(all_obs, device=self.device).float()
        phi_x = self.tdd_model.s_encoder(all_obs)  # (batch_size, hidden_dim)
        
        dists = mrn_distance(phi_x[:, None], phi_y[None, :])    # (batch_size, bathch_size) 여기서는 일단 행이 나타내는게 각 phi_x이며, 열이 바로 각 phi_y이다. 그래서 행별로 각 phi_x부터 모든 phi_y와의 거리를 계산한다.
        _, indices = torch.topk(dists, k=11, largest=False, dim=0)  # dim=0을 수행함으로써, 각 phi_y에 대해 가장 가까운 10개의 phi_x를 찾는다.
        nearest_neighbors = indices[1:, :]  # 첫번째 행은 아마 자기 자신일 가능성이 꽤 높다.
        
        neighbor_distances = torch.gather(dists, dim=0, index=nearest_neighbors)  # (10, batch_size)가 될 것이다.
        neighbor_distances = neighbor_distances.T   # (batch_size, 10)이 될 것이다.
        entropy_term = torch.log(1 + (1/10) * torch.sum(neighbor_distances ** batch_size, dim=1))  # 각 phi_y에 대해 10개의 가장 가까운 phi_x와의 거리를 모두 더한 후 10으로 나누고 1을 더한 후 로그를 취한다.
        # 그나저나 초기에는 entropy_term이 0이 뜨는데 괜찮으려나
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
                phi_g = self.tdd_model.s_encoder[agent_id](positions_tensor)    # (100, 32)
                phi_start = self.tdd_model.s_encoder[agent_id](start_pos_tensor_batch)    # (100, 32)
                
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
                save_dir = os.path.join(self.log_dir, f'step_{step}')
                os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
                save_path = os.path.join(save_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{suffix}.png')
            else:
                save_path = os.path.join(self.log_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{suffix}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()