import torch
import numpy as np
from harl.common.buffers.tdd_rollout_buffer import RolloutBuffer
from harl.algorithms.representation.tdd import TDDModel, mrn_distance


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
        
        self.tdd_model = TDDModel(self.tdd_args, 2, self.device, run_dir=log_dir)
        
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
    
    def update_tdd_model(self):
        self.tdd_model.update(self.rollout_buffer.rollout_history)
        
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

    def compute_intrinsic_reward(self, pos, new_pos):
        int_rew = [[] for _ in range(self.num_agents)]
        current_rollout_states = self.rollout_buffer.current_rollout_states[:]  # (n_agents, n_timesteps, n_rollout_threads, {'obs': (2,), 'next_obs': (2,)})
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                current_state = new_pos[agent_id]  # (n_rollout_threads, 2)
                current_state = torch.tensor(current_state, device=self.device).float()
                phi_y = self.tdd_model.s_encoder(current_state)  # (n_rollout_threads, hidden_dim)
                
                # 현재 에이전트의 모든 이전 상태들에 대해 MRN 거리 계산
                rollout_states = current_rollout_states[agent_id]
                    
                if rollout_states:  # 이전 상태가 있는 경우에만
                    if np.any(rollout_states[-1]['dones'], axis=-1): # rollout_states[-1]['dones']는 (n_rollout_threads,) 형태의 배열
                        raise AssertionError("rollout_states의 마지막 상태 중 하나 이상의 스레드가 완료된 경우입니다.")    
                    # 모든 이전 상태들을 하나의 텐서로 모음
                    prev_states = []
                    for i, state_dict in enumerate(rollout_states):
                        if i == len(rollout_states) - 1:  # 마지막 항목인 경우
                            prev_states.append(state_dict['obs'])  # (n_rollout_threads, 2)
                            prev_states.append(pos[agent_id])  # (n_rollout_threads, 2)
                            if not np.allclose(pos[agent_id], state_dict['next_obs'], rtol=1e-5, atol=1e-5):
                                raise AssertionError(f"rollout_states의 마지막 상태의 next_obs({state_dict['next_obs']})가 현재 상태({pos[agent_id]})와 다른 경우입니다.")
                        else:
                            prev_states.append(state_dict['obs'])  # (n_rollout_threads, 2)
                    prev_states = torch.tensor(np.array(prev_states), device=self.device).float()  # (n_timesteps, n_rollout_threads, 2)
                    
                    # 이전 상태들의 임베딩 계산
                    prev_states = prev_states.view(-1, prev_states.shape[-1])  # (n_timesteps * n_rollout_threads, 2)
                    phi_x = self.tdd_model.s_encoder(prev_states)  # (n_timesteps * n_rollout_threads, hidden_dim)
                    
                    # MRN 거리 계산: phi_x(이전 상태들)와 phi_y(현재 상태) 간의 거리
                    # phi_x: (n_timesteps * n_rollout_threads, hidden_dim)
                    # phi_y: (n_rollout_threads, hidden_dim)
                    phi_y_repeated = phi_y.repeat(prev_states.shape[0] // self.n_rollout_threads, 1)  # (n_timesteps * n_rollout_threads, hidden_dim)를 인풋으로 하는데 결과는 재밌는게 [40]임
                    
                    # 각 rollout thread별로 MRN 거리 계산
                    dists = mrn_distance(phi_x, phi_y_repeated)  # (n_timesteps * n_rollout_threads, n_timesteps * n_rollout_threads)
                    dists = dists.view(prev_states.shape[0] // self.n_rollout_threads, self.n_rollout_threads)  # (n_timesteps, n_rollout_threads)로 쪼개짐
                    
                    # 각 rollout thread별로 최소 거리 계산
                    min_dists = torch.min(dists, dim=0)[0]  # (n_rollout_threads,)
                    int_rew[agent_id].append(min_dists.cpu().numpy())
                else:
                    # 이전 상태가 없는 경우 0으로 설정
                    int_rew[agent_id].append(np.zeros(self.n_rollout_threads))
                
                """ 이전 rollout histiry mean state와의 mrn_distance 계산 """
                if self.tdd_args["train"]["tdd_with_mean_state"]:
                    prev_all_mean_state = self.rollout_buffer.all_rollouts_mean_state[agent_id]
                    if prev_all_mean_state:
                        prev_all_mean_state = torch.tensor(prev_all_mean_state, device=self.device).float()
                        phi_x = self.tdd_model.s_encoder(prev_all_mean_state)  # (n_rollout_threads, hidden_dim)
                        dists = mrn_distance(phi_x, phi_y)  # (n_rollout_threads, n_rollout_threads)
                        
                        int_rew[agent_id] += dists.cpu().numpy()
            
        int_rew = np.array(int_rew).transpose(2, 0, 1)  # (n_rollout_threads, n_agents, 1)
        
        # 리워드 정규화
        # int_rew = self.normalize_rewards(int_rew)
        
        return int_rew