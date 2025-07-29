"""Rollout buffer for saving and updating mean state of rollouts."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Headless backend 설정
import matplotlib.pyplot as plt
import os

# from collections import deque

class RolloutBuffer:
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
        self.all_rollouts_mean_state = [None for _ in range(self.num_agents)]
        self.current_rollout_states = [[] for _ in range(self.num_agents)]
        self.rollout_history = [[] for _ in range(self.num_agents)]
        self.prev_obs_shape = None
        self.prev_next_obs_shape = None
        self.prev_dones_shape = None
        
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
            else:
                assert self.prev_obs_shape == data["obs"][agent_id].shape, f"Observation shape mismatch: {self.prev_obs_shape} != {data['obs'][agent_id].shape}"
                assert self.prev_next_obs_shape == data["next_obs"][agent_id].shape, f"Next observation shape mismatch: {self.prev_next_obs_shape} != {data['next_obs'][agent_id].shape}"
                assert self.prev_dones_shape == data["dones"][agent_id].shape, f"Dones shape mismatch: {self.prev_dones_shape} != {data['dones'][agent_id].shape}"  
            self.current_rollout_states[agent_id].append({"obs": data["obs"][agent_id], "next_obs": data["next_obs"][agent_id], "dones": data["dones"][agent_id]})
    
    def end_rollout(self):
        """End current rollout and update statistics.
        Returns:
            all_rollouts_mean_state: updated mean state of all rollouts
        """
        for agent_id in range(self.num_agents):
            self.rollout_history[agent_id].append(self.current_rollout_states[agent_id])  # [agent_id]해서 (n_trajs, n_rollout_steps, {'obs': (n_threads,2), 'next_obs': (n_threads,2), 'dones': (n_threads,)})
            self.current_rollout_states[agent_id] = []
        
        return self.all_rollouts_mean_state  # (n_agents, n_rollout_threads, 2) 형태의 리스트

    def clear(self):
        """Clear rollout history."""
        for agent_id in range(self.num_agents):
            self.rollout_history[agent_id].clear()
            self.current_rollout_states[agent_id].clear()
    
class WM_RolloutBuffer:
    def __init__(self, args, obs_shape, action_spaces, num_agents=3, n_rollout_threads=20, device=torch.device("cpu")):
        self.args = args
        self.obs_shape = obs_shape[0].shape[0]
        self.num_agents = num_agents
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
        else:
            self.obs_s = 2
        self.episode_num = 0
        self.current_size = 0
        
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.num_agents, self.obs_s]),
                       'a_n_before': np.zeros([self.buffer_size, self.episode_limit + 1, self.num_agents, self.action_spaces[0].shape[0]]),
                       'active': np.zeros([self.buffer_size, self.episode_limit + 1, 1])
                       }
        self.episode_len = np.zeros(self.buffer_size)
    
    def store_transition(self, episode_step, obs_n, a_n, n_rollout_threads):
        for i in range(n_rollout_threads):
            self.buffer['obs_n'][self.episode_num + i][episode_step] = obs_n[:, i, :]
            self.buffer['a_n_before'][self.episode_num + i][episode_step + 1] = a_n[:, i, :]
            self.buffer['active'][self.episode_num + i][episode_step] = 1.0
    
    def store_last_step(self, episode_step, obs_n, a_n, n_rollout_threads):
        for i in range(n_rollout_threads):
            self.buffer['obs_n'][self.episode_num + i][episode_step] = obs_n[:, i, :]
            self.buffer['obs_n'][self.episode_num + i][episode_step + 1] = obs_n[:, i, :]
            self.buffer['a_n_before'][self.episode_num + i][episode_step + 1] = a_n[:, i, :]
            self.buffer['active'][self.episode_num + i][episode_step] = 1.0
            self.buffer['active'][self.episode_num + i][episode_step + 1] = 0
        
            self.episode_len[self.episode_num + i] = episode_step + 1
        self.episode_num = (self.episode_num + n_rollout_threads) % self.buffer_size
        self.current_size = min(self.current_size + n_rollout_threads, self.buffer_size)
    
    def sample(self, batch_size):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 'a_n_before':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32)

        return batch, max_episode_len