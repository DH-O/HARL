"""Rollout buffer for saving and updating mean state of rollouts."""
import numpy as np
import torch

class TddRolloutBuffer:
    """Buffer for storing and updating rollout mean states."""

    def __init__(self, args, obs_shape, num_agents=1, n_rollout_threads=1, device=torch.device("cpu")):
        """Initialize rollout buffer.
        Args:
            args: (dict) arguments
            obs_shape: (tuple) observation shape for single agent, or list of shapes for multi-agent
            num_agents: (int) number of agents in multi-agent mode
            device: (torch.device) device to use
        """
        self.args = args
        self.obs_shape = obs_shape # 멀티에이전트라서 주로 리스트고, 각 요소별 Box(-inf, inf, (14,), float32) 형태로 되어있음 in case MPE
        self.device = device
        self.num_agents = num_agents
        self.n_rollout_threads = n_rollout_threads
        
        # Rollout statistics
        self.rollout_count = np.zeros(self.num_agents, dtype=np.int)
        self.current_rollout_states = [[] for _ in range(self.num_agents)]
        self.rollout_history = [[] for _ in range(self.num_agents)]
        self.prev_obs_shape = None
        self.prev_next_obs_shape = None
        self.prev_dones_shape = None
        self.prev_share_obs_shape = None
        self.prev_next_share_obs_shape = None
        
    def add_data(self, data):
        """Add observation to current rollout. Agent-wise로 저장됨
        Args:
            obs: (numpy.ndarray) observation to add
                shape is (n_envs, n_agents, *obs_shape) or list of (n_envs, *obs_shape)
        """
        
        for agent_id in range(self.num_agents):
            if self.prev_obs_shape is None and self.prev_next_obs_shape is None and self.prev_dones_shape is None:
                self.prev_obs_shape = data["obs"][agent_id].shape
                self.prev_next_obs_shape = data["next_obs"][agent_id].shape
                self.prev_dones_shape = data["dones"][agent_id].shape
                self.prev_share_obs_shape = data["share_obs"][agent_id].shape
                self.prev_next_share_obs_shape = data["next_share_obs"][agent_id].shape
            else:
                assert self.prev_obs_shape == data["obs"][agent_id].shape, f"Observation shape mismatch: {self.prev_obs_shape} != {data['obs'][agent_id].shape}"
                assert self.prev_next_obs_shape == data["next_obs"][agent_id].shape, f"Next observation shape mismatch: {self.prev_next_obs_shape} != {data['next_obs'][agent_id].shape}"
                assert self.prev_dones_shape == data["dones"][agent_id].shape, f"Dones shape mismatch: {self.prev_dones_shape} != {data['dones'][agent_id].shape}"  
                assert self.prev_share_obs_shape == data["share_obs"][agent_id].shape, f"Share observation shape mismatch: {self.prev_share_obs_shape} != {data['share_obs'][agent_id].shape}"
                assert self.prev_next_share_obs_shape == data["next_share_obs"][agent_id].shape, f"Next share observation shape mismatch: {self.prev_next_share_obs_shape} != {data['next_share_obs'][agent_id].shape}"
            self.current_rollout_states[agent_id].append(
                {"share_obs": data["share_obs"][agent_id],
                 "obs": data["obs"][agent_id],
                 "next_share_obs": data["next_share_obs"][agent_id],
                 "next_obs": data["next_obs"][agent_id], 
                 "dones": data["dones"][agent_id]
                 })
    
    def end_rollout(self):
        """End current rollout and update statistics.
        Returns:
            all_rollouts_mean_state: updated mean state of all rollouts
        """
        for agent_id in range(self.num_agents):
            self.rollout_history[agent_id].append(self.current_rollout_states[agent_id])  # [agent_id]해서 (n_trajs, n_rollout_steps, {'obs': (n_threads,2), 'next_obs': (n_threads,2), 'dones': (n_threads,)})
            self.current_rollout_states[agent_id] = []

    def clear(self):
        """Clear rollout history."""
        for agent_id in range(self.num_agents):
            self.rollout_history[agent_id].clear()
            self.current_rollout_states[agent_id].clear()
    
class WmRolloutBuffer:
    def __init__(self, args, x_dim, action_spaces, num_agents=3, n_rollout_threads=20, device=torch.device("cpu")):
        self.args = args
        self.num_agents = num_agents
        if self.args["network"]["use_share_obs"]:
            self.obs_shape = x_dim // self.num_agents
        else:
            self.obs_shape = x_dim
        self.n_rollout_threads = n_rollout_threads
        self.device = device
        self.action_spaces = action_spaces
        
        """ WM 관련 """
        self.buffer_size = self.args["wm"]["buffer_size"]
        self.episode_limit = self.args["max_cycles"]
        if self.args["network"]["use_full_p_obs"]:
            self.obs_s = self.obs_shape
        elif self.args["network"]["use_intra_obs"]:
            self.obs_s = 4
        elif self.args["network"]["use_p_obs_without_others"]:
            self.obs_s = 2 + 2 + 2 * (self.num_agents)
        elif self.args["network"]["use_share_obs"]:
            self.obs_s = self.obs_shape
            self.share_obs_s = self.obs_shape * self.num_agents
        else:
            self.obs_s = 2
        self.episode_num = 0
        self.current_size = 0
        
        self.current_buffer = {
                        'share_obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.share_obs_s]),
                        'obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.obs_s]),
                        'actions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'a_before': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'available_actions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'rewards': np.zeros([self.n_rollout_threads, self.episode_limit, 1]),
                        'dones': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, 1]),
                        'valid_transitions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, 1]),
                        'terms': np.zeros([self.n_rollout_threads, self.episode_limit, 1]),
                        'next_share_obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.share_obs_s]),
                        'next_obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.obs_s]),
                        'next_available_actions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'active': np.zeros([self.n_rollout_threads, self.episode_limit, 1])
                       }
        self.buffer = {'share_obs': np.zeros([self.buffer_size, self.episode_limit, self.share_obs_s]),
                       'obs': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.obs_s]),
                       'actions': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                       'a_before': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                       'available_actions': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                       'rewards': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'dones': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, 1]),
                       'valid_transitions': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, 1]),
                       'terms': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'next_share_obs': np.zeros([self.buffer_size, self.episode_limit, self.share_obs_s]),
                       'next_obs': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.obs_s]),
                       'next_available_actions': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                       'a_before': np.zeros([self.buffer_size, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                       }
        self.current_episode_step = 0
        self.episode_len = np.zeros(self.buffer_size)
    
    def clear_current_buffer(self):
        """현재 버퍼를 초기화합니다."""
        self.current_buffer = {
                        'share_obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.share_obs_s]),
                        'obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.obs_s]),
                        'actions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'a_before': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'available_actions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'rewards': np.zeros([self.n_rollout_threads, self.episode_limit, 1]),
                        'dones': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, 1]),
                        'valid_transitions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, 1]),
                        'terms': np.zeros([self.n_rollout_threads, self.episode_limit, 1]),
                        'next_share_obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.share_obs_s]),
                        'next_obs': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.obs_s]),
                        'next_available_actions': np.zeros([self.n_rollout_threads, self.episode_limit, self.num_agents, self.action_spaces[0].shape[0]]),
                        'active': np.zeros([self.n_rollout_threads, self.episode_limit, 1])
                       }
        self.current_episode_step = 0
    
    def store_transition(self, 
                         episode_step, 
                         share_obs, 
                         obs, 
                         actions, 
                         available_actions, 
                         rewards, 
                         dones,
                         valid_transitions, 
                         terms, 
                         next_share_obs, 
                         next_obs, 
                         next_available_actions):
        """현재 버퍼에 transition을 저장합니다."""
        # obs_n: (num_agents, n_rollout_threads, obs_s)
        # a_n: (num_agents, n_rollout_threads, action_dim)
        # 한 번에 모든 rollout thread에 저장
        self.current_buffer['share_obs'][:, episode_step, :] = share_obs  # (n_rollout_threads, share_obs_s)
        self.current_buffer['obs'][:, episode_step, :, :] = obs.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, obs_s)
        self.current_buffer['actions'][:, episode_step, :, :] = actions.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, action_dim)
        if episode_step < self.episode_limit - 1:
            self.current_buffer['a_before'][:, episode_step + 1, :, :] = actions.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, action_dim)
        if available_actions is not None:
            self.current_buffer['available_actions'][:, episode_step, :, :] = available_actions.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, action_dim)
        self.current_buffer['rewards'][:, episode_step, :] = rewards  # (n_rollout_threads, 1)
        self.current_buffer['dones'][:, episode_step, :, :] = dones  # (n_rollout_threads, num_agents, 1)
        self.current_buffer['valid_transitions'][:, episode_step, :, :] = valid_transitions.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, 1)
        self.current_buffer['terms'][:, episode_step, :] = terms  # (n_rollout_threads, 1)
        self.current_buffer['next_share_obs'][:, episode_step, :] = next_share_obs  # (n_rollout_threads, num_agents, share_obs_s)
        self.current_buffer['next_obs'][:, episode_step, :, :] = next_obs.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, obs_s)
        if next_available_actions is not None:
            self.current_buffer['next_available_actions'][:, episode_step, :, :] = next_available_actions.transpose(1, 0, 2)  # (n_rollout_threads, num_agents, action_dim)
        self.current_buffer['active'][:, episode_step] = 1.0
        self.current_episode_step = episode_step
    
    def store_last_step(self, 
                        episode_step, 
                        share_obs, 
                        obs, 
                        actions, 
                        available_actions, 
                        rewards,
                        dones,
                        valid_transitions,
                        terms,
                        next_share_obs,
                        next_obs,
                        next_available_actions):
        """현재 버퍼에 마지막 스텝을 저장하고, 전체 버퍼에 복사합니다."""
        # obs_n: (num_agents, n_rollout_threads, obs_s)
        # a_n: (num_agents, n_rollout_threads, action_dim)
        # 한 번에 모든 rollout thread에 저장
        self.current_buffer['share_obs'][:, episode_step, :] = share_obs  # (n_rollout_threads, share_obs_s)
        self.current_buffer['obs'][:, episode_step, :, :] = obs.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, obs_s)
        self.current_buffer['actions'][:, episode_step, :, :] = actions.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, action_dim)
        # self.current_buffer['a_before'][:, episode_step + 1, :, :] = actions.transpose(1, 0, 2)
        if available_actions is not None:
            self.current_buffer['available_actions'][:, episode_step, :, :] = available_actions.transpose(1, 0, 2)  # (n_rollout_threads, num_agents, action_dim)
        self.current_buffer['rewards'][:, episode_step, :] = rewards  # (n_rollout_threads, 1)
        self.current_buffer['dones'][:, episode_step, :, :] = dones  # (n_rollout_threads, num_agents, 1)
        self.current_buffer['valid_transitions'][:, episode_step, :, :] = valid_transitions.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, 1)
        self.current_buffer['terms'][:, episode_step, :] = terms  # (n_rollout_threads, 1)
        self.current_buffer['next_share_obs'][:, episode_step, :] = next_share_obs  # (n_rollout_threads, num_agents, share_obs_s)
        self.current_buffer['next_obs'][:, episode_step, :, :] = next_obs.transpose(1, 0, 2)  # transpose해서 (n_rollout_threads, num_agents, obs_s)
        if next_available_actions is not None:
            self.current_buffer['next_available_actions'][:, episode_step, :, :] = next_available_actions.transpose(1, 0, 2)  # (n_rollout_threads, num_agents, action_dim)
        self.current_buffer['active'][:, episode_step] = 1.0
        self.current_episode_step = episode_step
        
        # 현재 버퍼의 데이터를 전체 버퍼에 한 번에 복사
        start_idx = self.episode_num
        end_idx = self.episode_num + self.n_rollout_threads
        self.buffer['share_obs'][start_idx:end_idx] = self.current_buffer['share_obs'].copy()
        self.buffer['obs'][start_idx:end_idx] = self.current_buffer['obs'].copy()   # 이렇게 하면 self.buffer['obs'][start_idx]는 (episode_limit + 1, num_agents, obs_s) 형태가 됨
        self.buffer['actions'][start_idx:end_idx] = self.current_buffer['actions'].copy()
        self.buffer['a_before'][start_idx:end_idx] = self.current_buffer['a_before'].copy()
        self.buffer['available_actions'][start_idx:end_idx] = self.current_buffer['available_actions'].copy()
        self.buffer['rewards'][start_idx:end_idx] = self.current_buffer['rewards'].copy()
        self.buffer['dones'][start_idx:end_idx] = self.current_buffer['dones'].copy()
        self.buffer['valid_transitions'][start_idx:end_idx] = self.current_buffer['valid_transitions'].copy()
        self.buffer['terms'][start_idx:end_idx] = self.current_buffer['terms'].copy()
        self.buffer['next_share_obs'][start_idx:end_idx] = self.current_buffer['next_share_obs'].copy()
        self.buffer['next_obs'][start_idx:end_idx] = self.current_buffer['next_obs'].copy()
        self.buffer['next_available_actions'][start_idx:end_idx] = self.current_buffer['next_available_actions'].copy()
        self.buffer['active'][start_idx:end_idx] = self.current_buffer['active'].copy()
        self.episode_len[start_idx:end_idx] = episode_step + 1
        
        self.episode_num = (self.episode_num + self.n_rollout_threads) % self.buffer_size
        self.current_size = min(self.current_size + self.n_rollout_threads, self.buffer_size)
        
        # 현재 버퍼 초기화
        self.clear_current_buffer()
    
    def sample(self, batch_size):
        # Randomly sampling
        batch = {}
        if self.current_size < batch_size:
            if self.current_size == 0:
                index  = np.random.choice(self.n_rollout_threads, size=batch_size, replace=True)
                max_episode_len = self.current_episode_step
                
                for key in self.current_buffer.keys():
                    if key == 'active':
                        batch[key] = torch.tensor(self.current_buffer[key][index, :max_episode_len], dtype=torch.float32, device=self.device)
                    else:
                        batch[key] = torch.tensor(self.current_buffer[key][index, :max_episode_len + 1], dtype=torch.float32, device=self.device)
                
                return batch, max_episode_len
            else:
                index = np.random.choice(self.current_size, size=batch_size, replace=True)
                max_episode_len = int(np.max(self.episode_len[index]))
        else:
            index = np.random.choice(self.current_size, size=batch_size, replace=False)
            max_episode_len = int(np.max(self.episode_len[index]))
            
        for key in self.buffer.keys():
            if key == 'active':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32, device=self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32, device=self.device)
        return batch, max_episode_len