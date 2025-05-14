"""Rollout buffer for saving and updating mean state of rollouts."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


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
        self.rollout_count = np.zeros(num_agents, dtype=np.int)
        self.all_rollouts_mean_state = [None for _ in range(num_agents)]
        self.current_rollout_states = [[] for _ in range(self.num_agents)]
        self.rollout_history = [[] for _ in range(self.num_agents)]
        
    def add_observation(self, obs):
        """Add observation to current rollout.
        Args:
            obs: (numpy.ndarray) observation to add
                shape is (n_envs, n_agents, *obs_shape) or list of (n_envs, *obs_shape)
        """
        
        for agent_id in range(self.num_agents):
            self.current_rollout_states[agent_id].append({"obs": obs["obs"][agent_id], "next_obs": obs["next_obs"][agent_id], "dones": obs["dones"][agent_id]})
    
    def compute_rollout_mean_state(self, agent_id=None):
        """Compute mean state of current rollout.
        Args:
            agent_id: (int) agent ID for multi-agent mode
        Returns:
            rollout_mean_state: (numpy.ndarray) mean state of current rollout
        """
        results = []
        for a_id in range(self.num_agents):
            results.append(self._compute_single_rollout_mean(a_id))
        
        return results
            
    def _compute_single_rollout_mean(self, agent_id):
        """Helper method to compute mean state for a single agent."""
        # 모든 타임스텝의 obs와 마지막 타임스텝의 next_obs를 수집
        rollout_states = self.current_rollout_states[agent_id]
        if not rollout_states:
            return None
            
        # 각 rollout_thread별로 관측값을 수집
        n_threads = rollout_states[0]['obs'].shape[0]  # n_rollout_threads
        all_obs = [[] for _ in range(n_threads)]  # 각 thread별로 관측값을 저장할 리스트
        
        # 모든 타임스텝의 obs만 수집
        for timestep_data in rollout_states:
            obs = timestep_data['obs']  # (n_rollout_threads, [x, y])
            for thread_idx in range(n_threads):
                all_obs[thread_idx].append(obs[thread_idx])  # 각 thread의 관측값을 해당 thread의 리스트에 추가
        
        # 각 thread별로 평균 계산
        mean_states = []
        for thread_obs in all_obs:
            if thread_obs:
                mean_state = np.mean(thread_obs, axis=0)  # 각 thread의 [x, y] 좌표의 평균
                mean_states.append(mean_state)
            else:
                mean_states.append(None)
        
        return mean_states  # (n_rollout_threads, 2) 형태의 리스트 반환
    
    def update_all_rollouts_mean_state(self, current_rollout_mean_state, agent_id=None):
        """Update mean state of all rollouts.
        Args:
            current_rollout_mean_state: (numpy.ndarray) mean state of current rollout
            agent_id: (int) agent ID for multi-agent mode
        """
        for a_id in range(self.num_agents):
            self._update_single_rollout_mean(current_rollout_mean_state[a_id], a_id)
                
    def _update_single_rollout_mean(self, current_mean, agent_id):
        """Helper method to update mean state for a single agent."""
        if current_mean is None:
            return
            
        n_threads = len(current_mean)
        self.rollout_count[agent_id] += 1
        
        if self.all_rollouts_mean_state[agent_id] is None:
            self.all_rollouts_mean_state[agent_id] = current_mean
        else:
            # 각 thread별로 가중 평균 업데이트
            for thread_idx in range(n_threads):
                if current_mean[thread_idx] is not None:
                    if self.all_rollouts_mean_state[agent_id][thread_idx] is None:
                        self.all_rollouts_mean_state[agent_id][thread_idx] = current_mean[thread_idx]
                    else:
                        self.all_rollouts_mean_state[agent_id][thread_idx] = (
                            0.99 * (self.rollout_count[agent_id] - 1) / self.rollout_count[agent_id] * 
                            self.all_rollouts_mean_state[agent_id][thread_idx] +
                            1 / self.rollout_count[agent_id] * current_mean[thread_idx]
                        )
    
    def end_rollout(self):
        """End current rollout and update statistics.
        Returns:
            all_rollouts_mean_state: updated mean state of all rollouts
        """
        # 모든 에이전트의 롤아웃 종료 및 업데이트
        current_rollout_mean_states = self.compute_rollout_mean_state()
        self.update_all_rollouts_mean_state(current_rollout_mean_states)
        for agent_id in range(self.num_agents):
            self.rollout_history[agent_id].append(self.current_rollout_states[agent_id])  # [agent_id]해서 (n_rollout_steps, n_rollout_threads, n_truncated_obs_dim)
            self.current_rollout_states[agent_id] = []
        
        return self.all_rollouts_mean_state  # (n_agents, n_rollout_threads, 2) 형태의 리스트
    
    def get_all_rollouts_mean_state(self, agent_id=None):
        """Get mean state of all rollouts.
        Args:
            agent_id: (int) agent ID for multi-agent mode
        Returns:
            all_rollouts_mean_state: mean state of all rollouts for each thread
        """
        if agent_id is not None:
            return self.all_rollouts_mean_state[agent_id]  # (n_rollout_threads, 2) 형태의 리스트
        return self.all_rollouts_mean_state  # [agent_id][thread_id] 형태의 2차원 리스트

    def plot_mean_state_trajectory(self, save_dir="rollout_mean_state_plots", map_size=1.0):
        """
        self.all_rollouts_mean_state의 변화를 시각화합니다.
        각 에이전트, 각 thread별로 평균 위치 궤적을 2D plot으로 저장합니다.
        """
        os.makedirs(save_dir, exist_ok=True)
        num_agents = self.num_agents
        n_threads = self.n_rollout_threads

        # self.rollout_history[agent_id] : (step, n_threads, 2)
        for agent_id in range(num_agents):
            plt.figure(figsize=(6, 6))
            for thread_id in range(n_threads):
                # 각 step별 평균 위치 궤적 추출
                traj = []
                for rollout in self.rollout_history[agent_id]:
                    # rollout: (n_rollout_steps, n_threads, 2)
                    if len(rollout) > thread_id:
                        # rollout[thread_id] : (n_rollout_steps, 2)
                        # 평균 위치 계산
                        mean_pos = np.mean([step['obs'][thread_id] for step in rollout], axis=0)
                        traj.append(mean_pos)
                traj = np.array(traj)
                if len(traj) > 0:
                    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f"thread_{thread_id}")
            plt.title(f"Agent {agent_id} Mean State Trajectory")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.xlim(-map_size, map_size)
            plt.ylim(-map_size, map_size)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"agent_{agent_id}_mean_state_trajectory.png"))
            plt.close()