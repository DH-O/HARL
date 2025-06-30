"""Rollout buffer for saving and updating mean state of rollouts."""
import numpy as np
import torch
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
        
    def add_observation(self, obs):
        """Add observation to current rollout.
        Args:
            obs: (numpy.ndarray) observation to add
                shape is (n_envs, n_agents, *obs_shape) or list of (n_envs, *obs_shape)
        """
        
        for agent_id in range(self.num_agents):
            if self.prev_obs_shape is None and self.prev_next_obs_shape is None and self.prev_dones_shape is None:
                self.prev_obs_shape = obs["obs"][agent_id].shape
                self.prev_next_obs_shape = obs["next_obs"][agent_id].shape
                self.prev_dones_shape = obs["dones"][agent_id].shape
                if self.args["use_central_SD"]:
                    self.prev_share_obs_shape = obs["share_obs"][agent_id].shape
                    self.prev_next_share_obs_shape = obs["next_share_obs"][agent_id].shape
            else:
                assert self.prev_obs_shape == obs["obs"][agent_id].shape, f"Observation shape mismatch: {self.prev_obs_shape} != {obs['obs'][agent_id].shape}"
                assert self.prev_next_obs_shape == obs["next_obs"][agent_id].shape, f"Next observation shape mismatch: {self.prev_next_obs_shape} != {obs['next_obs'][agent_id].shape}"
                assert self.prev_dones_shape == obs["dones"][agent_id].shape, f"Dones shape mismatch: {self.prev_dones_shape} != {obs['dones'][agent_id].shape}"
                if self.args["use_central_SD"]:
                    assert self.prev_share_obs_shape == obs["share_obs"][agent_id].shape, f"Share observation shape mismatch: {self.prev_share_obs_shape} != {obs['share_obs'][agent_id].shape}"
                    assert self.prev_next_share_obs_shape == obs["next_share_obs"][agent_id].shape, f"Next share observation shape mismatch: {self.prev_next_share_obs_shape} != {obs['next_share_obs'][agent_id].shape}"
            if self.args["use_central_SD"]:
                self.current_rollout_states[agent_id].append({"share_obs": obs["share_obs"][agent_id], "next_share_obs": obs["next_share_obs"][agent_id], "dones": obs["dones"][agent_id]})
            else:
                self.current_rollout_states[agent_id].append({"obs": obs["obs"][agent_id], "next_obs": obs["next_obs"][agent_id], "dones": obs["dones"][agent_id]})
    
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