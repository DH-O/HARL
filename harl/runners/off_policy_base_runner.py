"""Base runner for off-policy algorithms."""
import os
import cv2
import time
import torch
import imageio
import numpy as np
import setproctitle
""" exploration metric """
import matplotlib.pyplot as plt
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

def plot_rollout_trajectory(rollout_data, n_roll_out_threads, n_agents, model_dir, map_size):
        save_dir =  model_dir + "/exploration_metric"
        
        chunk_size = 3
        env_ids = list(rollout_data.keys())
        
        for i in range(0, n_roll_out_threads, chunk_size):
            sub_env_ids = env_ids[i:min(i+chunk_size, n_roll_out_threads)]
            num_sub_envs = len(sub_env_ids)
        
            fig, axes = plt.subplots(nrows=num_sub_envs, ncols=n_agents, figsize=(12, 4 * num_sub_envs))
            
            # if num_sub_envs == 1:
            #     axes = np.expand_dims(axes, axis=0)
            
            for row, env_id in enumerate(sub_env_ids):
                for agent_id in range(n_agents):
                    ax = axes[row, agent_id] if num_sub_envs > 1 else axes[agent_id]
                    if isinstance(ax, plt.Axes):
                        agent_traj = np.array(rollout_data[env_id][agent_id])
                        steps = agent_traj[:, 2]
                        norm_steps = (steps - np.min(steps)) / (np.max(steps) - np.min(steps) + 1e-8)
                        
                        colormap = cm.get_cmap('viridis')
                        colors = colormap(norm_steps)
                        
                        ax.scatter(agent_traj[:, 0], agent_traj[:, 1], c=colors, s=10, alpha=0.7)
                        
                        # ax.plot(agent_traj[:, 0], agent_traj[:, 1], marker='o', color='b', linestyle='-', alpha=0.7)
                        ax.set_title(f'env {env_id} - agent {agent_id}')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_xlim(-1.2*map_size, 1.2*map_size)
                        ax.set_ylim(-1.2*map_size, 1.2*map_size)
                        
                        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=ax)
                        cbar.set_label("Step Index")
                    else:
                        raise ValueError('axes must be an instance of plt.Axes')
        
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            
            base_filename = f'rollout_trajectories_envs_{i//chunk_size + 1}'
            file_path = os.path.join(save_dir, f'{base_filename}.png')
            
            counter = 1
            while os.path.exists(file_path):
                filename = f"{base_filename}_{counter}.png"
                file_path = os.path.join(save_dir, filename)
                counter += 1
            
            plt.savefig(file_path, dpi=300)
            plt.close()
        
class OffPolicyBaseRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
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
        
        """ exploration metric """
        # if self.args["use_exploration_metric"]:
        #     (
        #         _,
        #         self.manual_render,
        #         self.manual_expand_dims,
        #         self.manual_delay,
        #         self.env_num,
        #     ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        """ exploration metric 끝 """
        
        
        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")   # state_type이 없으면 기본값은 "EP"로 가져오란 뜻. 
        # dict.get(key, default)는 dict에 key가 있으면 dict[key]를 반환하고, 없으면 default를 반환한다.
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
            save_config(args, algo_args, env_args, self.run_dir)
            self.log_file = open(
                os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
            )
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
            if self.state_type == "EP":
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

        if (
            "auto_alpha" in self.algo_args["algo"].keys()
            and self.algo_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                if (
                    self.envs.action_space[agent_id].__class__.__name__ == "Box"
                ):  # Differential entropy can be negative
                    self.target_entropy.append(
                        -np.prod(self.envs.action_space[agent_id].shape)
                    )
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
        
    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"]:  # render, not train
            self.render()
            return
        self.train_episode_rewards = np.zeros(
            self.n_rollout_threads
        )
        self.done_episodes_rewards = []
        # warmup
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start training")
        # train and eval
        steps = (
            self.algo_args["train"]["num_env_steps"]    # 무려 20M
            // self.n_rollout_threads # 이걸 나의 경우 2로 나눈셈
        )
        update_num = int(  # update number per train    # update_per_train이 1이고, train_invercal이 50일 경우, 매 50스텝마다 50번 학습. 만일 update_per_train이 2라면 100번 학습.
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        """ exploration metric """
        if self.args["use_exploration_metric"]:
            rollout_data = {env_id: {agent_id: [] for agent_id in range(self.num_agents)} for env_id in range(self.n_rollout_threads)}
            target_dim = [2, 3] # 랜드마크와 아군의 수와 상관 없이, 커서 에이전트의 위치는 2, 3에 있다.
        
            # # Video Save Dir
            # save_dir = self.save_dir + "/videos"
            # os.makedirs(save_dir, exist_ok=True)
            # gif_filenames = [f'{save_dir}/gif_env_id_{env_id}.gif' for env_id in range(self.n_rollout_threads)]
            # gif_frames = [[] for _ in range(self.n_rollout_threads)]
            
        """ exploration metric 끝 """
        
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
            next_obs = new_obs.copy()
            
            """ exploration metric """
            if self.args["use_exploration_metric"]:
                for env_id in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        xy_coords = new_obs[env_id, agent_id, target_dim]
                        rollout_data[env_id][agent_id].append([xy_coords[0], xy_coords[1], step])
            """ exploration metric 끝"""
            
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
            self.insert(data)
            
            """ exploration metric """
            # if self.args["use_exploration_metric"]:
            #     for env_id in range(self.n_rollout_threads):
            #         if self.manual_render:
            #             gif_frame = self.envs[env_id].get_attr("render")()
            #             gif_frames[env_id].append(gif_frame)    
                
            #         if self.manual_delay:
            #             time.sleep(0.1)
            """ exploration metric 끝 """
            
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            if step % self.algo_args["train"]["train_interval"] == 0:   # train_interval이 50이면 50스텝마다 학습
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    if self.share_param:
                        self.actor[0].lr_decay(step, steps)
                    else:
                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num): # update_num은 50이다.
                    self.train()    # 여기서 HASAC의 train()이 호출된다.
            if step % self.algo_args["train"]["eval_interval"] == 0:
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
            # for env_id in range(self.n_rollout_threads):
            #     imageio.mimsave(gif_filenames[env_id], gif_frames[env_id], duration=0.1)
            plot_rollout_trajectory(rollout_data, self.n_rollout_threads, self.num_agents, self.save_dir, self.env_args["map_size"])
        """ exploration metric 끝 """
        
    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"] # petting_zoo_mpe기준 10000.
            // self.n_rollout_threads
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        obs, share_obs, available_actions = self.envs.reset()
        for _ in range(warmup_steps):
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
                obs.transpose(1, 0, 2), # 리플레이 버퍼에 데이터를 저장할 때, 에이전트 단위로 데이터를 저장하기 위함
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
        return obs, share_obs, available_actions

    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten() # 각 환경 별로 3개의 에이전트들의 리워드를 axis=1 방향으로 평균을 내고, flatten()으로 1차원으로 펴준다. 결국 (2,)차원이 된다.
        self.train_episode_rewards += reward_env    # 어차피 한 에피소드 기준으로 다 더하는 것이다.

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
                        and infos[i][0]["bad_transition"] == True   # bad_transition이 True면 terms[i]는 False로 남아있게 된다.
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
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0   # 다음 에피소드를 위해 해당 환경의 train_episode_rewards를 0으로 초기화한다.
                self.agent_deaths = np.zeros(
                    (self.n_rollout_threads, self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()    # i번째 환경에서 모든 에이전트의 reset()된 직후 초기 obs들을 저기에 넣는다.
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()

        if self.state_type == "EP":
            data = (
                share_obs[:, 0],  # (n_threads, share_obs_dim)  # 이 조건이 굉장히 신박한건데 "첫번째 에이전트"의 share_obs만 저장한다는 것이다.
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards[:, 0],  # (n_threads, 1)
                np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
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
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )

        self.buffer.insert(data)

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

    def train(self):
        """Train the model"""
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
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

        while True:
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False
            )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
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
                        f"Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}.\n"
                    )
                if "smac" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
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
                                    step,
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
                        ",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n"
                    )
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards", eval_avg_rew, step
                )
                self.writter.add_scalar(
                    "eval_average_episode_length", eval_avg_len, step
                )
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        
        """ Video Save Dir """
        save_dir = self.algo_args["train"]["model_dir"] + "/videos"
        os.makedirs(save_dir, exist_ok=True)
        
        if self.manual_expand_dims: # true
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for episode in range(self.algo_args["render"]["render_episodes"]):
                
                """ Video/Gif 관련 """
                gif_filename = f'{save_dir}/gif_episode_{episode}.gif'
                gif_frames = []
                
                """ exploration metric """
                if self.args["use_exploration_metric"]:
                    rollout_data = {env_id: {agent_id: [] for agent_id in range(self.num_agents)} for env_id in range(self.n_rollout_threads)}
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
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.array([eval_available_actions])
                    
                    """ exploration metric """
                    if self.args["use_exploration_metric"]:
                        for env_id in range(self.n_rollout_threads):
                            for agent_id in range(self.num_agents):
                                xy_coords = eval_obs[env_id, agent_id, target_dim]
                                rollout_data[env_id][agent_id].append([xy_coords[0], xy_coords[1], step])
                    """ exploration metric 끝"""
                    
                    if self.manual_render:
                        frame = self.envs.render()
                        gif_frame = frame
                        gif_frames.append(gif_frame)
                        
                    
                    if self.manual_delay:
                        time.sleep(0.1)
                    
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
                    step += 1
                
                if self.args["use_exploration_metric"]:
                    # for env_id in range(self.n_rollout_threads):
                    #     imageio.mimsave(gif_filenames[env_id], gif_frames[env_id], duration=0.1)
                    plot_rollout_trajectory(rollout_data, self.n_rollout_threads, self.num_agents, save_dir, self.env_args["map_size"])
                imageio.mimsave(gif_filename, gif_frames, duration=0.1)
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
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.log_file.close()
