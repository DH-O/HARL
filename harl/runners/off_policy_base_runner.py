"""Base runner for off-policy algorithms."""
import os
import cv2
import time
import torch
import imageio
import numpy as np
import setproctitle
import logging
logger = logging.getLogger(__name__)
""" exploration metric """
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
""" exploration metric 끝 """
from harl.common.valuenorm import ValueNorm
from torch.distributions import Categorical
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
""" TDD 관련 """
from harl.runners.tdd_runner import TddRunner
""" TDD 관련 끝 """

def plot_rollout_trajectory(rollout_data, n_roll_out_threads, n_agents, save_dir, map_size, step=None, warmup=False):
        save_dir =  save_dir + "/exploration_metric"
        if step is not None:
            save_dir = os.path.join(save_dir, f'step_{step}')
            os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        else:
            os.makedirs(save_dir, exist_ok=True)
        max_graphs_per_row = 3
        n_cols = min(max_graphs_per_row, n_agents)
        n_rows = int(np.ceil(n_agents / n_cols))
        
        for env_id in range(min(n_roll_out_threads, 5)):
            # 각 환경별로 새로운 figure 생성
            fig, axes = plt.subplots(
                nrows=n_rows, 
                ncols=n_cols, 
                figsize=(5 * n_cols, 4 * n_rows)
            )
            
            # axes가 1차원 배열인 경우 2차원 배열로 변환
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                # shape: (n_cols, ) -> (1, n_cols)
                axes = axes.reshape(1, n_cols)
            elif n_cols == 1:
                # shape: (n_rows, ) -> (n_rows, 1)
                axes = axes.reshape(n_rows, 1)
        
            for agent_id in range(n_agents):
                row = agent_id // n_cols
                col = agent_id % n_cols
                
                agent_traj = np.array(rollout_data[env_id][agent_id])
                steps = agent_traj[:, 2]
                norm_steps = (steps - np.min(steps)) / (np.max(steps) - np.min(steps) + 1e-8)
                
                colormap = matplotlib.colormaps['viridis']
                colors = colormap(norm_steps)
                
                axes[row, col].scatter(agent_traj[:, 0], agent_traj[:, 1], c=colors, s=10, alpha=0.7)
                axes[row, col].set_title(f'env {env_id} - agent {agent_id}')
                axes[row, col].set_xlabel('x')
                axes[row, col].set_ylabel('y')
                axes[row, col].set_xlim(-1.2*map_size, 1.2*map_size)
                axes[row, col].set_ylim(-1.2*map_size, 1.2*map_size)
                
                cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=axes[row, col])
                cbar.set_label("Step Index")
            
            # 사용하지 않는 subplot 제거                
            for idx in range(n_agents, n_cols * n_rows):
                row = idx // n_cols
                col = idx % n_cols
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            
            # 파일 이름 중복 방지: 이미 존재하는 경우 숫자 추가
            if warmup:
                base_filename = f'warmup_rollout_trajectories_envs_{env_id}'
            else:
                base_filename = f'rollout_trajectories_envs_{env_id}_step_{step}'
            file_path = os.path.join(save_dir, f'{base_filename}.png')
            counter = 1
            while os.path.exists(file_path):
                filename = f"{base_filename}_{counter}.png"
                file_path = os.path.join(save_dir, filename)
                counter += 1

            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
 
class OffPolicyBaseRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args, tdd_args):
        """Initialize the OffPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        
        self.n_rollout_threads =  self.algo_args["train"]["n_rollout_threads"]
        
        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")   # state_type이 없으면 기본값은 "EP"로 가져오란 뜻. 
        # dict.get(key, default)는 dict에 key가 있으면 dict[key]를 반환하고, 없으면 default를 반환한다.
        # mpe의 경우 EP
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args["render"]["use_render"]:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, tdd_args, self.run_dir)
            self.log_file = open(
                os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
            )
        
        # 프로세스 이름 설정
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1)
        )

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        """ TDD 관련 """
        self.tdd_args = tdd_args
        if self.tdd_args is not None and not self.algo_args["render"]["use_render"]:
            # TDD 로깅 설정 - save_dir 안에 로그 파일 생성
            self._setup_tdd_logging()
            self.tdd_runner = TddRunner(algo_args["train"]["n_rollout_threads"], self.num_agents, self.envs.observation_space, self.tdd_args, self.save_dir)
        """ TDD 관련 끝 """
        
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},    # 딕셔너리를 언패킹할때는 별을 2개 붙여야 키와 벨류 모두 나와진다.
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if not self.algo_args["render"]["use_render"]:
            self.critic = CRITIC_REGISTRY[args["algo"]](    # 저렇게 해서 클래스를 가져온다.
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.share_observation_space[0],
                self.envs.action_space,
                self.num_agents,
                self.state_type,
                device=self.device,
            )

            if self.state_type == "EP": # MPE의 경우 EP
                self.buffer = OffPolicyBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            elif self.state_type == "FP":
                self.buffer = OffPolicyBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            else:
                raise NotImplementedError

        if (
            "use_valuenorm" in self.algo_args["train"].keys()
            and self.algo_args["train"]["use_valuenorm"]
        ):
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

        self.total_it = 0  # total iteration

        # 알파 값 설정
        if (
            "auto_alpha" in self.algo_args["algo"].keys()
            and self.algo_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                if (
                    self.envs.action_space[agent_id].__class__.__name__ == "Box"
                ):  # Differential entropy can be negative
                    if self.tdd_args is not None and self.tdd_args["train"]["use_state_entropy"] and not self.tdd_args["train"]["use_actor_entropy"]:
                        self.target_entropy.append(
                            -np.prod(2 * (self.num_agents - 1))    # successor distance에 관여하는 차원이 2차원이다. 본인 제외한 모든 에이전트 사이와의 거리 갯수만큼 차원의 기준으로 정해봤다.
                        )
                        print(f"use_state_entropy and not use_actor_entropy, therefore target_entropy is {self.target_entropy[-1]}")
                    elif self.tdd_args is not None and self.tdd_args["train"]["use_actor_entropy"] and self.tdd_args["train"]["use_state_entropy"]:
                        self.target_entropy.append(
                            -np.prod(self.envs.action_space[agent_id].shape)    # (상, 하, 좌, 우) 라서 5차원
                        )
                        print(f"use_actor_entropy for decentralized and use_state_entropy for centralized, therefore target_entropy is {self.target_entropy[-1]}")
                    else:
                        self.target_entropy.append(
                            -np.prod(self.envs.action_space[agent_id].shape)    # (상, 하, 좌, 우) 라서 5차원
                        )
                        if self.tdd_args is not None:
                            print(f"use_state_entropy: {self.tdd_args['train']['use_state_entropy']}, use_actor_entropy: {self.tdd_args['train']['use_actor_entropy']}, therefore target_entropy is {self.target_entropy[-1]}")
                        else:
                            print("tdd_args is None")
                else:  # Discrete entropy is always positive. Thus we set the max possible entropy as the target entropy
                    self.target_entropy.append(
                        -0.98
                        * np.log(1.0 / np.prod(self.envs.action_space[agent_id].shape))
                    )
            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha = []
            for agent_id in range(self.num_agents):
                _log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.log_alpha.append(_log_alpha)
                self.alpha_optimizer.append(
                    torch.optim.Adam(
                        [_log_alpha], lr=self.algo_args["algo"]["alpha_lr"]
                    )
                )
                self.alpha.append(torch.exp(_log_alpha.detach()))
        elif "alpha" in self.algo_args["algo"].keys():
            self.alpha = [self.algo_args["algo"]["alpha"]] * self.num_agents
        
        
    def _setup_tdd_logging(self):
        """TDD 관련 로깅 설정을 초기화합니다."""
        # TDD 로깅이 비활성화된 경우 설정하지 않음
        if self.tdd_args is not None and "logging" in self.tdd_args and not self.tdd_args["logging"]["enable_tdd_logging"]:
            return
            
        # 이미 설정되었는지 확인
        tdd_logger = logging.getLogger('harl.runners.tdd_runner')
        base_logger = logging.getLogger('harl.runners.off_policy_base_runner')
        
        # 이미 핸들러가 있으면 추가 설정하지 않음
        if tdd_logger.handlers or base_logger.handlers:
            return
        
        # 기본 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # TDD 관련 로거 설정
        tdd_logger.setLevel(logging.INFO)
        
        # TDD 파일 핸들러 (save_dir 안에 생성)
        tdd_file_handler = logging.FileHandler(os.path.join(self.save_dir, 'tdd_runner.log'))
        tdd_file_handler.setFormatter(formatter)
        tdd_logger.addHandler(tdd_file_handler)
        tdd_logger.propagate = False  # 상위 로거로 전파하지 않음
        
        # Base runner 로거 설정
        base_logger.setLevel(logging.INFO)
        
        # Base runner 파일 핸들러 (save_dir 안에 생성)
        base_file_handler = logging.FileHandler(os.path.join(self.save_dir, 'tdd_base_runner.log'))
        base_file_handler.setFormatter(formatter)
        base_logger.addHandler(base_file_handler)
        base_logger.propagate = False  # 상위 로거로 전파하지 않음
        
        # Root 로거는 터미널 출력만 방지 (한 번만 설정)
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.setLevel(logging.WARNING)  # WARNING 이상만 처리
            root_logger.addHandler(logging.NullHandler())
        
    def run(self):
        print(self.run_dir)
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"]:  # render, not train
            self.render()
            self.close()  # 렌더링 후 리소스 정리
            return
        self.train_episode_rewards = np.zeros(
            self.n_rollout_threads
        )
        
        """ use_eval 안 쓸 때 활용하는 것으로 보인다 """
        self.done_episodes_rewards = []
        
        # warmup
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        # obs, share_obs, available_actions = self.envs.reset()
        print("finish warmup, start training")
        
        # train and eval
        steps = (
            self.algo_args["train"]["num_env_steps"]    # 지금은 10^7 사용중
            // self.n_rollout_threads
        )
        
        update_num = int(  # update number per train    # update_per_train이 1이고, train_invercal이 50일 경우, 매 50스텝마다 50번 학습. 만일 update_per_train이 2라면 100번 학습.
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        
        """ exploration metric """
        if self.args["use_exploration_metric"]:
            rollout_data_for_metric = {env_id: {agent_id: [] for agent_id in range(self.num_agents)} for env_id in range(self.n_rollout_threads)}
            target_dim = [2, 3] # 랜드마크와 아군의 수와 상관 없이, 커서 에이전트의 절대좌표는 obs에서 [2, 3]에 있다.
        """ exploration metric 끝 """
        
        if self.tdd_args is not None:
            self.tdd_runner.rollout_buffer.clear()
        rollout_history_count = 0
        
        train_tdd_flag = True if self.tdd_args is not None else False
        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs, available_actions=available_actions, add_random=True
            )
            
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(
                actions
            )  # rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents)
            # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
            # dones를 판별하는 기준은 mpe에서는 그냥 self.steps가 max_cycles 이상인지 검사해서 판별한다.
            # dones 뜨면 new_obs는 초기 위치로 간다.
            next_obs = new_obs.copy()
            
            """ exploration metric """
            if self.args["use_exploration_metric"]:
                for env_id in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        xy_coords = new_obs[env_id, agent_id, target_dim]
                        rollout_data_for_metric[env_id][agent_id].append([xy_coords[0], xy_coords[1], step])
            """ exploration metric 끝"""
            
            """ TDD intrinsic reward """
            if self.tdd_args is not None:    
                """ intrinsic reward 계수 계산 """
                if self.tdd_args["train"]["coeff_stop_ratio"] == 0:
                    int_rew_coeff = 1.0
                else:
                    if step <= steps // self.tdd_args["train"]["coeff_stop_ratio"]:
                        # 선형적으로 감소하는 계수 계산 (1.0에서 0.0으로)
                        int_rew_coeff = 1.0 - (step / (steps // self.tdd_args["train"]["coeff_stop_ratio"]))
                
                if self.tdd_args["network"]["use_central_SD"]:
                    pos = share_obs
                    new_pos = new_share_obs
                else:
                    pos = obs[:, :, 2:4] # obs: (n_threads, n_agents, obs_dim)
                    new_pos = new_obs[:, :, 2:4] # new_obs: (n_threads, n_agents, obs_dim)
                if dones.any():
                    int_rew = self.tdd_runner.compute_intrinsic_reward(pos.transpose(1, 0, 2), pos.transpose(1, 0, 2), n_rollout_threads=self.n_rollout_threads)
                else:
                    int_rew = self.tdd_runner.compute_intrinsic_reward(pos.transpose(1, 0, 2), new_pos.transpose(1, 0, 2), n_rollout_threads=self.n_rollout_threads)
                if self.tdd_args["train"]["use_suppression_reward"]:
                    # 조금 무서운게 rewards 왜 다 똑같은 걸로 나오냐?
                    rewards = (1 - int_rew_coeff) * rewards + int_rew_coeff * self.tdd_args["train"]["coeff_magnitude"] * int_rew   # size (n_threads, n_agents, 1)
                else:
                    if self.tdd_args["train"]["off_extrinsic_reward"]:
                        rewards = int_rew_coeff * self.tdd_args["train"]["coeff_magnitude"] * int_rew
                    else:
                        rewards += int_rew_coeff * self.tdd_args["train"]["coeff_magnitude"] * int_rew
            """ TDD intrinsic reward 끝 """
            
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self.insert(data)   # 여기서 이제 롤아웃 버퍼도 야무지게 충전 중일 것이다.
            
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            
            dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
            if self.tdd_args is not None:
                if any(dones_env):
                    rollout_history_count += 1
                if len(self.tdd_runner.rollout_buffer.rollout_history[0]) != len(self.tdd_runner.rollout_buffer.rollout_history[1]):
                    raise ValueError("rollout_history[0] and rollout_history[1] must have the same length")
                if len(self.tdd_runner.rollout_buffer.rollout_history[0]) >= (self.tdd_args["train"]["update_interval_of_rollout_history"] // self.n_rollout_threads) and len(self.tdd_runner.rollout_buffer.rollout_history[0]) > 0:  # 3000 // 20 = 150
                    if train_tdd_flag:
                        self.tdd_runner.update_tdd_model()
                        train_tdd_flag = False
                    
                        if self.tdd_args["network"]["use_central_SD"]:
                            pass
                        else:
                            start_pos_ls = []
                            for i in range(self.num_agents):
                                start_pos = self.tdd_runner.rollout_buffer.rollout_history[i][-1][0]["obs"]  # (n_agents, traj_id, max_cycles, "obs" -> n_rollout_threads, 2) 그래서 좌항은 결국 (n_rollout_threads, 2)
                                start_pos_ls.append(start_pos)  # (n_rollout_threads, 2)
                            start_pos = np.stack(start_pos_ls, axis=1)  # (n_rollout_threads, n_agents, 2)
                            
                            midle_pos_ls = []
                            for i in range(self.num_agents):
                                midle_pos = self.tdd_runner.rollout_buffer.rollout_history[i][-1][self.env_args["max_cycles"] // 2]["obs"]  # (n_agents, max_cycles, "obs" -> n_rollout_threads, 2) 그래서 좌항은 결국 (n_rollout_threads, 2)
                                midle_pos_ls.append(midle_pos)  # (n_rollout_threads, 2)
                            midle_pos = np.stack(midle_pos_ls, axis=1)  # (n_rollout_threads, n_agents, 2)
                            
                            end_pos_ls = []
                            for i in range(self.num_agents):
                                end_pos = self.tdd_runner.rollout_buffer.rollout_history[i][-1][-1]["obs"]  # (n_agents, max_cycles, "obs" -> n_rollout_threads, 2) 그래서 좌항은 결국 (n_rollout_threads, 2)
                                end_pos_ls.append(end_pos)  # (n_rollout_threads, 2)
                            end_pos = np.stack(end_pos_ls, axis=1)  # (n_rollout_threads, n_agents, 2)
                            
                            pos_ls = [start_pos, midle_pos, end_pos]
                            pos_arr = np.stack(pos_ls, axis=0)  # (3, n_rollout_threads, n_agents, 2)   
                            
                            # 각 환경과 에이전트별로 거리 맵 생성
                            for thread_id in range(min(self.n_rollout_threads, 3)):
                                for agent_id in range(self.num_agents):
                                    for pos_idx in range(3):
                                        # 랜드마크와 장애물 정보 가져오기
                                        self.envs.remotes[thread_id].send(("get_landmarks_and_obstacles", None))
                                        landmarks, obstacles = self.envs.remotes[thread_id].recv()
                                        
                                        # 목표점 가져오기
                                        pos = pos_arr[pos_idx][thread_id, agent_id]
                                        
                                        # 거리 맵 생성
                                        self.tdd_runner.plot_distance_map(
                                            pos, 
                                            self.env_args["map_size"],
                                            agent_id,
                                            landmarks, 
                                            obstacles, 
                                            f"thread_{thread_id}_agent_{agent_id}_pos_{pos_idx}_{pos[0]}_{pos[1]}",
                                            step=step
                                        )
                
                if step % self.algo_args["train"]["train_interval"] == 0:
                    # TDD가 활성화된 경우 롤아웃 버퍼 충분성 체크
                    buffer_sufficient = True  # 기본값 설정
                    if self.tdd_args is not None:
                        # 롤아웃 버퍼가 충분한지 확인
                        
                        # tdd.yaml 설정값에 따라 동적으로 최소 요구사항 설정
                        min_trajectories = max(2, self.tdd_args["train"]["batch_size"] // 1000)  # batch_size에 따라 조정
                        min_steps_per_traj = max(10, self.tdd_args["train"]["max_historical_samples"] // 200)  # max_historical_samples에 따라 조정
                        
                        if len(self.tdd_runner.rollout_buffer.rollout_history) == 0:
                            buffer_sufficient = False
                        else:
                            for agent_id in range(self.num_agents):
                                agent_history = self.tdd_runner.rollout_buffer.rollout_history[agent_id]
                                if len(agent_history) < min_trajectories:
                                    buffer_sufficient = False
                                    break
                                
                                # 각 궤적에 충분한 스텝이 있는지 확인
                                for traj in agent_history:
                                    if len(traj) < min_steps_per_traj:
                                        buffer_sufficient = False
                                        break
                        
                        # 버퍼가 부족한 경우 경고 출력 (1000 스텝마다만)
                        if not buffer_sufficient:
                            if self.tdd_args is not None:
                                # 로깅이 활성화된 경우에만 출력
                                if "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"] and step % 1000 == 0:
                                    logger.warning(f"Step {step}: 롤아웃 버퍼가 부족합니다. (최소 {min_trajectories}개 궤적, 각 궤적당 {min_steps_per_traj}스텝 필요)")
                                    logger.warning(f"rollout_history_count: {rollout_history_count}, 현재 롤아웃 버퍼 상태: {len(self.tdd_runner.rollout_buffer.rollout_history[0]) if len(self.tdd_runner.rollout_buffer.rollout_history) > 0 else 0}개 궤적, 현재 step: {step}")
                            else:
                                print(f"Step {step}: 롤아웃 버퍼가 부족합니다. (최소 {min_trajectories}개 궤적, 각 궤적당 {min_steps_per_traj}스텝 필요)")
                            continue
                    
                    for _ in range(update_num): # update_num은 50이다.
                        critic_loss, actor_loss_ls, alpha_loss = self.train(step)    # 여기서 HASAC의 train()이 호출된다.
                        self.writter.add_scalar("critic_loss", critic_loss, step)
                        self.writter.add_scalar("actor_loss/agent_0", actor_loss_ls[0], step)
                        self.writter.add_scalar("actor_loss/agent_1", actor_loss_ls[1], step)
                        self.writter.add_scalar("actor_loss/agent_2", actor_loss_ls[2], step)
                        self.writter.add_scalar("alpha_loss", alpha_loss, step)
                    self.writter.add_scalar("rollout_history_count", rollout_history_count, step)
                    
                    # 버퍼가 충분한 경우에만 롤아웃 버퍼 클리어
                    if self.tdd_args is not None and buffer_sufficient and len(self.tdd_runner.rollout_buffer.rollout_history[0]) > (self.tdd_args["train"]["update_interval_of_rollout_history"] // self.n_rollout_threads):
                        self.tdd_runner.rollout_buffer.clear()
                        train_tdd_flag = True
                        if "logging" in self.tdd_args and self.tdd_args["logging"]["enable_performance_logs"]:
                            logger.info(f"Step {step}: 롤아웃 버퍼를 클리어했습니다. 새로운 궤적 수집을 시작합니다.")
            else:
                if step % self.algo_args["train"]["train_interval"] == 0:   # train_interval이 50이면 50스텝마다 학습. 근데 이거 tdd 업데이트랑 일치시키는게 좋을 것 같긴 한데
                    if self.algo_args["train"]["use_linear_lr_decay"]:  # False
                        if self.share_param:
                            self.actor[0].lr_decay(step, steps)
                        else:
                            for agent_id in range(self.num_agents):
                                self.actor[agent_id].lr_decay(step, steps)
                        self.critic.lr_decay(step, steps)
                    for _ in range(update_num): # update_num은 50이다.
                        critic_loss, actor_loss, alpha_loss = self.train()    # 여기서 HASAC의 train()이 호출된다.
                        self.writter.add_scalar("critic_loss", critic_loss, step)
                        self.writter.add_scalar("actor_loss", actor_loss, step)
                        self.writter.add_scalar("alpha_loss", alpha_loss, step)
            
            if step % self.algo_args["train"]["eval_interval"] == 0:
                print(f"rollout_history_count: {rollout_history_count}")
                cur_step = (
                    self.algo_args["train"]["warmup_steps"]
                    + step * self.n_rollout_threads
                )
                if self.algo_args["eval"]["use_eval"]:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.algo_args['train']['num_env_steps']}:"
                    )
                    self.eval(cur_step)
                else:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Step {cur_step} / {self.algo_args['train']['num_env_steps']}, average step reward in buffer: {self.buffer.get_mean_rewards()}.\n"
                    )
                    if len(self.done_episodes_rewards) > 0:
                        aver_episode_rewards = np.mean(self.done_episodes_rewards)
                        print(
                            "Some episodes done, average episode reward is {}.\n".format(
                                aver_episode_rewards
                            )
                        )
                        self.log_file.write(
                            ",".join(map(str, [cur_step, aver_episode_rewards])) + "\n"
                        )
                        self.log_file.flush()
                        self.done_episodes_rewards = []
                self.save()
                """ exploration metric """
                if self.args["use_exploration_metric"]:
                    plot_rollout_trajectory(rollout_data_for_metric, self.n_rollout_threads, self.num_agents, self.save_dir, self.env_args["map_size"], step)
                """ exploration metric 끝 """
        
        """ exploration metric """
        if self.args["use_exploration_metric"]:
            plot_rollout_trajectory(rollout_data_for_metric, self.n_rollout_threads, self.num_agents, self.save_dir, self.env_args["map_size"], step)
        """ exploration metric 끝 """
        
        # 학습이 완료된 후 리소스 정리
        self.close()

    def warmup(self):        
        """Warmup the replay buffer with random actions"""
        if self.tdd_args is not None and self.tdd_args["train"]["use_synthetic_data"]:
            n_trajs = self.tdd_args["train"]["n_trajs_for_synthetic_data"]
        else:
            warmup_steps = (
                self.algo_args["train"]["warmup_steps"] # petting_zoo_mpe기준 10000.
                // self.n_rollout_threads
            )
            n_rollout_threads = self.n_rollout_threads
            
        
        if self.tdd_args is not None and self.tdd_args["train"]["use_synthetic_data"]:
            # xy_coords: (n_trajs, n_agents, 2)의 차원인데, 각각의 데이터는 -0.8 * env_args["map_size"], -0.8 * env_args["map_size"]를 중심으로 하고 반경 env_args["agent_size"] * 3 안에서 랜덤하게 샘플링되도록 코딩
            def get_const_pos(pos_list, center, bounds):
                if not pos_list:
                    return center
                return center + np.random.uniform(-bounds, bounds, 2)
            
            rollout_data = {traj_id: {agent_id: [] for agent_id in range(self.num_agents)} for traj_id in range(n_trajs)}
            
            """ 1. 초기 좌표 설정 """
            respawn_coords = np.zeros((n_trajs, self.num_agents, 2))
            for i in range(n_trajs):
                for j in range(self.num_agents):
                    respawn_coords[i, j] = get_const_pos([respawn_coords[i, j]], [-0.8 * self.env_args["map_size"], -0.8 * self.env_args["map_size"]], self.env_args["agent_size"] * 2)

            steps_per_segment = self.env_args["max_cycles"] // 5  # 각 구간별 step 수
            """ 2. 각 구간별 좌표 설정 """
            for i in range(n_trajs):
                for j in range(self.num_agents):
                    x, y = respawn_coords[i, j]
                    traj = []
                    step_idx = 0
                    # 0. 처음 좌표 저장
                    rollout_data[i][j].append([x, y, step_idx])
                    traj.append([x, y])  # 1중 리스트로 저장
                    
                    # 1. y좌표가 map_size - agent_size * 1.5까지 위로 직진 (x는 고정)
                    start_y = y
                    end_y1 = self.env_args["map_size"] - self.env_args["agent_size"] * 1.5
                    for step in range(steps_per_segment - 1):
                        ratio = step / steps_per_segment
                        cur_y = start_y + (end_y1 - start_y) * ratio
                        pos = get_const_pos([[x, cur_y]], [x, cur_y], self.env_args["agent_size"] * 2)
                        rollout_data[i][j].append([pos[0], pos[1], step_idx])
                        step_idx += 1
                        traj.append([pos[0], pos[1]])
                    y = end_y1
                    if j == 0:
                        end_x_agent_0 = x
                        end_y_agent_0 = y

                    # 2. x좌표가 map_size / 2 - agent_size / 2 - map_size / 10까지 오른쪽 직진 (y는 고정)
                    start_x = x
                    end_x2 = self.env_args["map_size"] / 2 - self.env_args["agent_size"] / 2 - self.env_args["map_size"] / 10
                    for step in range(steps_per_segment):
                        ratio = step / steps_per_segment
                        cur_x = start_x + (end_x2 - start_x) * ratio
                        if j == 0:
                            pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], self.env_args["agent_size"] * 2)
                        else:
                            pos = get_const_pos([[cur_x, y]], [cur_x, y], self.env_args["agent_size"] * 2)
                        rollout_data[i][j].append([pos[0], pos[1], step_idx])
                        step_idx += 1
                        traj.append([pos[0], pos[1]])
                    x = end_x2

                    # 3. y좌표가 -map_size + agent_size * 1.5까지 아래로 직진 (x는 고정)
                    start_y = y
                    end_y3 = -self.env_args["map_size"] + self.env_args["agent_size"] * 1.5
                    for step in range(steps_per_segment):
                        ratio = step / steps_per_segment
                        cur_y = start_y + (end_y3 - start_y) * ratio
                        if j == 0:
                            pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], self.env_args["agent_size"] * 2)
                        else:
                            pos = get_const_pos([[x, cur_y]], [x, cur_y], self.env_args["agent_size"] * 2)
                        rollout_data[i][j].append([pos[0], pos[1], step_idx])
                        step_idx += 1
                        traj.append([pos[0], pos[1]])
                    y = end_y3
                    if j == 1:
                        end_x_agent_1 = x
                        end_y_agent_1 = y
                    
                    # 4. x좌표가 map_size - agent_size * 1.5까지 오른쪽 직진 (y는 고정)
                    start_x = x
                    end_x4 = self.env_args["map_size"] - self.env_args["agent_size"] * 1.5
                    for step in range(steps_per_segment):
                        ratio = step / steps_per_segment
                        cur_x = start_x + (end_x4 - start_x) * ratio
                        if j == 0:
                            pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], self.env_args["agent_size"] * 2)
                        elif j == 1:
                            pos = get_const_pos([[end_x_agent_1, end_y_agent_1]], [end_x_agent_1, end_y_agent_1], self.env_args["agent_size"] * 2)
                        else:
                            pos = get_const_pos([[cur_x, y]], [cur_x, y], self.env_args["agent_size"] * 2)
                        rollout_data[i][j].append([pos[0], pos[1], step_idx])
                        step_idx += 1
                        traj.append([pos[0], pos[1]])
                    x = end_x4
                    
                    # 5. y좌표가 map_size - agent_size * 1.5까지 위로 직진 (x는 고정)
                    start_y = y
                    end_y5 = self.env_args["map_size"] - self.env_args["agent_size"] * 1.5
                    for step in range(steps_per_segment):
                        ratio = step / steps_per_segment
                        cur_y = start_y + (end_y5 - start_y) * ratio
                        if j == 0:
                            pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], self.env_args["agent_size"] * 2)
                        elif j == 1:
                            pos = get_const_pos([[end_x_agent_1, end_y_agent_1]], [end_x_agent_1, end_y_agent_1], self.env_args["agent_size"] * 2)
                        else:
                            pos = get_const_pos([[x, cur_y]], [x, cur_y], self.env_args["agent_size"] * 2)
                        rollout_data[i][j].append([pos[0], pos[1], step_idx])
                        step_idx += 1
                        traj.append([pos[0], pos[1]])
                    y = end_y5
                    
                    # traj 저장한거 이제 롤아웃 버퍼에 저장하자
                    for t in range(len(traj) - 1):
                        obs_dict = {
                            "obs": np.array([traj[t]], dtype=np.float32),
                            "next_obs": np.array([traj[t + 1]], dtype=np.float32),
                            "dones": np.array([False])
                        }
                        self.tdd_runner.rollout_buffer.current_rollout_states[j].append(obs_dict)
                    obs_dict = {
                        "obs": np.array([traj[-1]], dtype=np.float32),
                        "next_obs": np.array([traj[0]], dtype=np.float32),
                        "dones": np.array([True])
                    }
                    self.tdd_runner.rollout_buffer.current_rollout_states[j].append(obs_dict)
                self.tdd_runner.rollout_buffer.end_rollout()
        else:
            # obs: (n_threads, n_agents, dim)
            # share_obs: (n_threads, n_agents, dim)
            obs, share_obs, available_actions = self.envs.reset()
            # 궤적 기록용 변수
            rollout_data = {env_id: {agent_id: [] for agent_id in range(self.num_agents)} for env_id in range(n_rollout_threads)}
            
            for step in range(warmup_steps):
                # action: (n_threads, n_agents, dim)
                actions = self.sample_actions(available_actions)    # available_actions는 discrete action space일 때만 존재한다.
                
                (
                    new_obs,
                    new_share_obs,
                    rewards,
                    dones,
                    infos,
                    new_available_actions,
                ) = self.envs.step(actions) # continuous action space에서는 new_available_actions도 계속 None, None이 된다.
                
                next_obs = new_obs.copy()
                next_share_obs = new_share_obs.copy()
                next_available_actions = new_available_actions.copy()
                data = (
                    share_obs,
                    obs.transpose(1, 0, 2), # 리플레이 버퍼에 데이터를 저장할 때, 에이전트 단위로 데이터를 저장하기 위함    # 롤아웃 버퍼도 저렇게 해야하는지 좀 고민이 되긴 하다
                    actions.transpose(1, 0, 2),
                    available_actions.transpose(1, 0, 2)
                    if len(np.array(available_actions).shape) == 3
                    else None,
                    rewards,
                    dones,
                    infos,
                    next_share_obs,
                    next_obs,   # 얘는 그런데 에이전트 단위로 저장 안 해도 되나보다?
                    next_available_actions.transpose(1, 0, 2)
                    if len(np.array(available_actions).shape) == 3
                    else None,
                )
                self.insert(data)
                obs = new_obs
                share_obs = new_share_obs
                available_actions = new_available_actions
            
                # 위치 기록
                for env_id in range(n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        respawn_coords = obs[env_id, agent_id, 2:4]  # x, y 좌표
                        rollout_data[env_id][agent_id].append([respawn_coords[0], respawn_coords[1], step])

        # warmup 궤적 시각화
        if self.tdd_args is None:
            plot_rollout_trajectory(
                rollout_data, 
                n_rollout_threads, 
                self.num_agents, 
                save_dir=self.run_dir,  # 또는 원하는 경로
                map_size=self.env_args["map_size"],
                warmup=True   # plot_rollout_trajectory에서 filename 인자를 받도록 수정 필요
            )
        else:
            plot_rollout_trajectory(
                rollout_data, 
                n_trajs if self.tdd_args["train"]["use_synthetic_data"] else n_rollout_threads, 
                self.num_agents, 
                save_dir=self.run_dir,  # 또는 원하는 경로
                map_size=self.env_args["map_size"],
                warmup=True   # plot_rollout_trajectory에서 filename 인자를 받도록 수정 필요
            )
        
        """ TDD update """
        if self.tdd_args is not None:
            # TDD 모델 업데이트
            self.tdd_runner.update_tdd_model(is_warm_up=True)
            # 환경 리셋
            obs, _, _ = self.envs.reset()
            # 에이전트의 위치만 추출 (x, y 좌표)
            agent_positions = obs[:, :, 2:4]  # (n_threads, n_agents, 2)
            # 각 환경과 에이전트별로 거리 맵 생성 (최적화: 샘플링으로 줄임)
            if self.tdd_args["network"]["use_central_SD"]:
                pass
            else:
                # 샘플링: 전체 환경과 에이전트 중 일부만 선택
                sample_threads = min(2, self.n_rollout_threads)  # 최대 2개 환경만
                sample_agents = self.num_agents  # 모든 에이전트 유지
                sample_positions = 3  # 위치 변화도 3개만
                
                for i in range(sample_positions):
                    for thread_id in range(sample_threads):
                        for agent_id in range(sample_agents):
                            # 랜드마크와 장애물 정보 가져오기
                            self.envs.remotes[thread_id].send(("get_landmarks_and_obstacles", None))
                            landmarks, obstacles = self.envs.remotes[thread_id].recv()
                            
                            # 현재 에이전트의 위치를 목표로 설정
                            start_pos = agent_positions[thread_id, agent_id].copy()
                            start_pos[0] = np.clip(start_pos[0] + self.env_args["map_size"]/3 * i, -self.env_args["map_size"], self.env_args["map_size"])
                            start_pos[1] = np.clip(start_pos[1] + self.env_args["map_size"]/3 * i, -self.env_args["map_size"], self.env_args["map_size"])
                            # 거리 맵 생성
                            self.tdd_runner.plot_distance_map(
                                start_pos, 
                                self.env_args["map_size"],
                                agent_id,
                                landmarks, 
                                obstacles, 
                                f"thread_{thread_id}_agent_{agent_id}_start_pos_ith_{i}_{start_pos[0]:.2f}_{start_pos[1]:.2f}",
                                step=0
                            )
            
            self.tdd_runner.rollout_buffer.clear()
            print("Representation learning 완료. 거리 맵이 생성되었습니다.")
        """ TDD update 끝 """
        if self.tdd_args is not None:
            if self.tdd_args["train"]["use_synthetic_data"]:
                # 모든 환경 프로세스 종료
                self.envs.close()
                if hasattr(self, 'eval_envs') and self.eval_envs is not None:
                    self.eval_envs.close()
                # 메인 프로세스 강제 종료
                os._exit(0)  # sys.exit(0) 대신 os._exit(0) 사용
            return obs, share_obs, available_actions
        else:
            return obs, share_obs, available_actions

    def insert(self, data, is_warmup=False):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents). 초기에는 보통 비어져 있다.
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten() # 각 환경 별로 3개의 에이전트들의 리워드를 axis=1 방향으로 평균을 내고, flatten()으로 1차원으로 펴준다. 결국 (2,)차원이 된다.
        self.train_episode_rewards += reward_env    # 어차피 한 에피소드 기준으로 다 더하는 것이다. 지금 당장에는 insert 불러질때마다 더하는 것

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        valid_transitions = 1 - self.agent_deaths   # shape: (n_threads, n_agents, 1)

        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
        if self.state_type == "EP": # Ego Perspective
            terms = np.full((self.n_rollout_threads, 1), False)
            for i in range(self.n_rollout_threads):
                if dones_env[i]:
                    if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True   # dones_env[i인데 bad_transition일 경우, infos[i][0]["bad_transition"]이 True로 바뀌어있을 것이다. 그럴 때는 terms[i]가 False로 남아있게 된다.
                    ):
                        terms[i][0] = True  # bad_transition이 아니라서 제대로 terminate된 경우에만 terms[i]는 True로 바뀐다.
        elif self.state_type == "FP":   # Full Perspective
            terms = np.full(
                (self.n_rollout_threads, self.num_agents, 1),
                False,
            )
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    if dones[i][agent_id]:
                        if not (
                            "bad_transition" in infos[i][agent_id].keys()
                            and infos[i][agent_id]["bad_transition"] == True
                        ):
                            terms[i][agent_id][0] = True

        for i in range(self.n_rollout_threads):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])    # self.done_episodes_rewards는 처음엔 그냥 빈 리스트. 계속 append.
                self.train_episode_rewards[i] = 0   # 다음 에피소드를 위해 해당 환경의 train_episode_rewards를 0으로 초기화한다.
                self.agent_deaths = np.zeros(
                    (self.n_rollout_threads, self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()    # i번째 환경에서 모든 에이전트가 끝났을 때의 다음 obs들을 저기에 넣는다. shape: (n_threads, n_agents, obs_dim)
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()    # shape: (n_threads, n_agents, share_obs_dim), 솔직히 next_obs[i]와 다를게 거의 없다.

        if self.state_type == "EP":
            data = (
                share_obs[:, 0],  # (n_threads, share_obs_dim)  # 이 조건이 굉장히 신박한건데 "첫번째 에이전트"의 share_obs만 저장한다는 것이다.
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards[:, 0],  # (n_threads, 1)
                np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)   이게 dones다.
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, 1)
                next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )
        elif self.state_type == "FP":
            data = (
                share_obs,  # (n_threads, n_agents, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards,  # (n_threads, n_agents, 1)
                np.expand_dims(dones, axis=-1),  # (n_threads, n_agents, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, n_agents, 1)
                next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # 트랜스포즈 거치면 (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )
        if is_warmup:
            if self.tdd_args["off_extrinsic_reward"]:
                pass
            else:
                self.buffer.insert(data)
        else:
            self.buffer.insert(data)
        
        """ TDD update """
        if self.tdd_args is not None:
            extracted_obs = obs[:, :, 2:4] # 에이전트 개인의 현재 절대 좌표만 뽑기 (n_agents, n_threads, 2)
            extracted_next_obs = next_obs[:, :, 2:4] # 에이전트 개인의 다음 절대 좌표만 뽑기 (n_threads, n_agents, 2)
            if self.tdd_args["network"]["use_central_SD"]:
                self.tdd_runner.rollout_buffer.add_observation({"share_obs": share_obs.transpose(1, 0, 2), "next_share_obs": next_share_obs.transpose(1, 0, 2), "dones": dones.transpose(1, 0)})
            else:
                self.tdd_runner.rollout_buffer.add_observation({"obs": extracted_obs, "next_obs": extracted_next_obs.transpose(1, 0, 2), "dones": dones.transpose(1, 0)})
            if np.any(np.all(dones, axis=1)):
                self.tdd_runner.rollout_buffer.end_rollout()
        """ TDD update 끝 """

    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
        Args:
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
        Returns:
            actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.n_rollout_threads):
                if available_actions[thread] is None:
                    action.append(self.action_spaces[agent_id].sample())
                    # self.action_spaces[agent_id]는 gym.spaces.Box(0, 1, (17,), float32)이기 때문에 .sample()하면 알아서 랜덤 샘플한다. 
                else:
                    action.append(
                        Categorical(
                            torch.tensor(available_actions[thread, agent_id, :])
                        ).sample()
                    )
            actions.append(action)
        if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
            return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)

        return np.array(actions).transpose(1, 0, 2) # (n_threads, n_agents, dim)

    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        if self.args["algo"] == "hasac":
            actions = []
            for agent_id in range(self.num_agents):
                if (
                    len(np.array(available_actions).shape) == 3
                ):  # (n_threads, n_agents, action_number)
                    actions.append(
                        _t2n(
                            self.actor[agent_id].get_actions(
                                obs[:, agent_id],
                                available_actions[:, agent_id],
                                add_random,
                            )
                        )
                    )
                else:  # (n_threads, ) of None
                    actions.append(
                        _t2n(
                            self.actor[agent_id].get_actions(
                                obs[:, agent_id], stochastic=add_random # 모든 환경의 agent_id번째 에이전트의 obs를 넣어서 action을 뽑아낸다.
                            )
                        )
                    )
        else:
            actions = []
            for agent_id in range(self.num_agents):
                actions.append(
                    _t2n(self.actor[agent_id].get_actions(obs[:, agent_id], add_random))
                )
        return np.array(actions).transpose(1, 0, 2)

    def train(self, step=None):
        """Train the model"""
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, cur_step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        n_eval_rollout_threads = min(self.algo_args["eval"]["n_eval_rollout_threads"], 5)
        for eval_i in range(n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        if "smac" in self.args["env"]:
            eval_battles_won = 0
        if "football" in self.args["env"]:
            eval_score_cnt = 0
        episode_lens = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int
        )

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        # 궤적 기록용 변수
        rollout_data = {env_id: {agent_id: [] for agent_id in range(self.num_agents)} for env_id in range(n_eval_rollout_threads)}
        temp_rollout_buffer = [[] for _ in range(self.num_agents)]
        
        # intrinsic rewards와 obs 데이터 저장을 위한 변수
        eval_data_dir = os.path.join(self.save_dir, "eval", f"cur_step_{cur_step}")
        os.makedirs(eval_data_dir, exist_ok=True)
        
        # Video/GIF 관련 변수
        gif_frames = []
        video_writer = None
        
        while True:
            temp_eval_data_dir = os.path.join(self.save_dir, "eval", f"cur_step_{cur_step}", f"eval_episode_{eval_episode}")
            os.makedirs(temp_eval_data_dir, exist_ok=True)
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False
            )
            (
                next_eval_obs,   # (n_threads, n_agents, obs_dim)
                next_eval_share_obs,
                eval_rewards,
                eval_dones, # (n_threads, n_agents)
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            
            # intrinsic rewards 계산 (TDD가 있는 경우)
            if self.tdd_args is not None:
                if self.tdd_args["network"]["use_central_SD"]:
                    pos = eval_share_obs  # obs: (n_threads, n_agents, obs_dim)
                    new_pos = next_eval_share_obs  # new_obs: (n_threads, n_agents, obs_dim)
                else:
                    pos = eval_obs[:, :, 2:4]  # obs: (n_threads, n_agents, obs_dim)
                    new_pos = next_eval_obs[:, :, 2:4]  # new_obs: (n_threads, n_agents, obs_dim)
                
                int_rew = self.tdd_runner.compute_intrinsic_reward(pos.transpose(1, 0, 2), new_pos.transpose(1, 0, 2), is_eval=True, temp_rollout_buffer=temp_rollout_buffer, n_rollout_threads=n_eval_rollout_threads)
                
                # intrinsic rewards 저장 (10 스텝마다만 저장)
                if one_episode_len[0] % 10 == 0:
                    for eval_i in range(n_eval_rollout_threads):
                        data_file = os.path.join(temp_eval_data_dir, f'rollout_{eval_i}_data.txt')
                        with open(data_file, 'a') as f:
                            f.write(f"Step {one_episode_len[eval_i]}:\n")
                            f.write(f"Intrinsic Rewards: {int_rew[eval_i]}\n")
                            f.write(f"Obs: {eval_obs[eval_i]}\n")
                            f.write(f"New Obs: {next_eval_obs[eval_i]}\n")
                            f.write("-" * 50 + "\n")
            
            for eval_i in range(n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                for agent_id in range(self.num_agents):
                    xy_coords = eval_obs[eval_i, agent_id, 2:4]
                    rollout_data[eval_i][agent_id].append([xy_coords[0], xy_coords[1], cur_step])

            for agent_id in range(self.num_agents):
                if self.tdd_args["network"]["use_central_SD"]:
                    temp_rollout_buffer[agent_id].append({"share_obs": eval_share_obs.transpose(1, 0, 2)[agent_id], "next_share_obs": next_eval_share_obs.transpose(1, 0, 2)[agent_id], "dones": eval_dones.transpose(1, 0)[agent_id]})
                else:
                    temp_rollout_buffer[agent_id].append({"obs": eval_obs.transpose(1, 0, 2)[agent_id, :, 2:4], "next_obs": next_eval_obs.transpose(1, 0, 2)[agent_id, :, 2:4], "dones": eval_dones.transpose(1, 0)[agent_id]})
            
            one_episode_len += 1
            eval_obs = next_eval_obs
            eval_share_obs = next_eval_share_obs
            
            # 첫 5 steps이랑 롤아웃 길이의 5등분 지점에서 distance map 생성
            if self.tdd_args is not None:
                for eval_i in range(n_eval_rollout_threads):
                    total_steps = one_episode_len[eval_i]
                    if total_steps > 0:
                        if total_steps % (self.env_args["max_cycles"] // 5) == 0 or total_steps < 5:
                            # 랜드마크와 장애물 정보 가져오기
                            self.eval_envs.remotes[eval_i].send(("get_landmarks_and_obstacles", None))
                            landmarks, obstacles = self.eval_envs.remotes[eval_i].recv()
                            
                            if self.tdd_args["network"]["use_central_SD"]:
                                pass
                            else:
                                for agent_id in range(self.num_agents):
                                    current_pos = eval_obs[eval_i, agent_id, 2:4]
                                    self.tdd_runner.plot_distance_map(
                                    current_pos,
                                    self.env_args["map_size"],
                                    agent_id,
                                    landmarks,
                                    obstacles,
                                    f"eval_thread_{eval_i}_agent_{agent_id}_step_{total_steps}",
                                    step=cur_step
                                )

            # 비디오/GIF 생성 (최적화)
            if one_episode_len[0] % 2 == 0:  # 2 스텝마다만 프레임 저장
                self.eval_envs.remotes[0].send(("render", None))
                frame = self.eval_envs.remotes[0].recv()
                
                # 에이전트 정보를 표시할 이미지 생성
                info_frame = np.ones((frame.shape[0], 300, 3), dtype=np.uint8) * 255
                
                # 에이전트 정보 텍스트 추가
                for agent_id in range(self.num_agents):
                    pos = eval_obs[0, agent_id, 2:4]
                    int_reward = float(int_rew[0][agent_id]) if self.tdd_args is not None else 0.0
                    ext_reward = float(eval_rewards[0][agent_id][0])
                    
                    # 각 정보를 별도의 텍스트로 표시
                    cv2.putText(info_frame, f"Agent {agent_id}", (5, 15 + agent_id * 100), # 5, 15 + agent_id * 100은 텍스트의 위치
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)   # 0.14가 크기
                    cv2.putText(info_frame, f"Position: ({pos[0]:.2f}, {pos[1]:.2f})", (5, 35 + agent_id * 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(info_frame, f"Intrinsic Reward: {int_reward:.4f}", (5, 55 + agent_id * 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(info_frame, f"Extrinsic Reward: {ext_reward:.4f}", (5, 75 + agent_id * 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # 원본 프레임과 정보 프레임 합치기
                combined_frame = np.hstack((frame, info_frame))
                gif_frames.append(combined_frame)
                
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    frame_height, frame_width = combined_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        os.path.join(temp_eval_data_dir, f'eval_video.mp4'),
                        fourcc,
                        5,  # FPS를 10에서 5로 줄임
                        (frame_width, frame_height)
                    )
                
                frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)   # 가끔 여기서 
                
            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    # 궤적 시각화
                    plot_rollout_trajectory(rollout_data, 
                                            n_eval_rollout_threads, 
                                            self.num_agents, 
                                            temp_eval_data_dir, 
                                            self.env_args["map_size"],
                                            step=cur_step,
                                            warmup=False)
                    # 비디오/GIF 저장
                    if video_writer is not None:
                        video_writer.release()
                        imageio.mimsave(os.path.join(temp_eval_data_dir, f'eval_animation.gif'), gif_frames, duration=0.2)  # duration 증가
                    else:
                        raise AssertionError("video_writer is None")
                    eval_episode += 1
                    
                    if "smac" in self.args["env"]:
                        if "v2" in self.args["env"]:
                            if eval_infos[eval_i][0]["battle_won"]:
                                eval_battles_won += 1
                        else:
                            if eval_infos[eval_i][0]["won"]:
                                eval_battles_won += 1
                    if "football" in self.args["env"]:
                        if eval_infos[eval_i][0]["score_reward"] > 0:
                            eval_score_cnt += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

            if np.any(eval_dones_env):
                if not np.all(eval_dones_env):
                    raise AssertionError("eval_dones_env is not all True")
                rollout_data[eval_i] = {agent_id: [] for agent_id in range(self.num_agents)}
                gif_frames = []
                video_writer = None
                temp_rollout_buffer = [[] for _ in range(self.num_agents)]
            
            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                if "smac" in self.args["env"]:
                    print(
                        "Eval win rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_battles_won / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                elif "football" in self.args["env"]:
                    print(
                        "Eval score rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_score_cnt / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                else:
                    print(
                        f"Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}."
                    )
                    cur_time = time.strftime("%m-%d %H:%M:%S", time.localtime(time.time()))
                    print(
                        f"The time is {cur_time}.\n"
                    )
                if "smac" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    cur_step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_battles_won / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                elif "football" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    cur_step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_score_cnt / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                else:
                    self.log_file.write(
                        ",".join(map(str, [cur_step, eval_avg_rew, eval_avg_len])) + "\n"
                    )
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards", eval_avg_rew, cur_step
                )
                self.writter.add_scalar(
                    "eval_average_episode_length", eval_avg_len, cur_step
                )
                break
        
        del rollout_data, temp_rollout_buffer

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        
        """ Video Save Dir """
        save_dir = self.algo_args["train"]["model_dir"] + "/videos" + f"/map_size_{self.env_args['map_size']}" + f"/max_cycles_{self.env_args['max_cycles']}" + f"/seed_{self.algo_args['seed']['seed']}"
        os.makedirs(save_dir, exist_ok=True)
        
        if self.manual_expand_dims: # true
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for episode in range(self.algo_args["render"]["render_episodes"]):
                
                """ Video/Gif 관련 """
                base_gif_filename = os.path.join(save_dir, f'gif_episode_{episode}')
                base_video_filename = os.path.join(save_dir, f'video_episode_{episode}')
                gif_counter = 1
                video_counter = 1
                while os.path.exists(f'{base_gif_filename}.gif'):
                    base_gif_filename = os.path.join(save_dir, f'gif_episode_{episode}_{gif_counter}')
                    gif_counter += 1
                
                while os.path.exists(f'{base_video_filename}.mp4'):
                    base_video_filename = os.path.join(save_dir, f'video_episode_{episode}_{video_counter}')
                    video_counter += 1
                
                gif_frames = []
                video_writer = None
                """ Video/Gif 관련 끝 """
                
                """ exploration metric """
                if self.args["use_exploration_metric"]:
                    rollout_data = {env_id: {agent_id: [] for agent_id in range(self.num_agents)} for env_id in range(1)}
                    target_dim = [2, 3] # 랜드마크와 아군의 수와 상관 없이, 커서 에이전트의 위치는 2, 3에 있다.
                """ exploration metric 끝 """
                
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = np.array([eval_available_actions])
                rewards = 0
                
                step = 1
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=eval_available_actions,
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    
                    step_reward = eval_rewards[0][0]
                    if eval_rewards[0][0] != eval_rewards[1][0]:
                        print(f"step_reward: {step_reward}")
                        raise AssertionError("step_reward is not equal")
                    rewards += step_reward
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.array([eval_available_actions])
                    
                    """ exploration metric """
                    if self.args["use_exploration_metric"]:
                        # for env_id in range(self.n_rollout_threads):
                        for agent_id in range(self.num_agents):
                            xy_coords = eval_obs[0, agent_id, target_dim]
                            rollout_data[0][agent_id].append([xy_coords[0], xy_coords[1], step])
                    """ exploration metric 끝"""
                    
                    if self.manual_render:
                        frame = self.envs.render()
                        gif_frames.append(frame)
                    
                        if video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            frame_height, frame_width = frame.shape[:2]
                            video_writer = cv2.VideoWriter(
                                f'{base_video_filename}.mp4',
                                fourcc,
                                10,
                                (frame_width, frame_height)
                            )

                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                        
                    if self.manual_delay:
                        time.sleep(0.1)
                    
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        
                        if video_writer is not None:
                            video_writer.release()
                        break
                    step += 1
                
                if self.args["use_exploration_metric"]:
                    plot_rollout_trajectory(rollout_data, 1, self.num_agents, save_dir, self.env_args["map_size"])
                imageio.mimsave(f'{base_gif_filename}.gif', gif_frames, duration=0.1)
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=eval_available_actions,
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(self.algo_args["train"]["model_dir"], agent_id)
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(self.algo_args["train"]["model_dir"])
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def save(self):
        """Save the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.algo_args["render"]["use_render"]:
            if hasattr(self.envs, 'close'):
                self.envs.close()
            # pygame 리소스 정리
            if hasattr(self.envs, 'viewer') and self.envs.viewer is not None:
                import pygame
                pygame.quit()
        else:
            if hasattr(self.envs, 'close'):
                self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                if hasattr(self.eval_envs, 'close'):
                    self.eval_envs.close()
                # eval 환경의 pygame 리소스 정리
                if hasattr(self.eval_envs, 'viewer') and self.eval_envs.viewer is not None:
                    import pygame
                    pygame.quit()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.log_file.close()
