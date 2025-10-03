"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner

class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""

    def train(self, step=None, use_rollout_buffer=False, rollout_buffer=None, wm_runner=None):
        """ Train the model """ # batch가 주로 1000이다. rollout으로 할때는 한 128개만 일단 뽑아볼까?
        self.total_it += 1  # train 할때마다 하나씩 증가
        
        """ use_rollout_buffer가 True인 경우, rollout_buffer를 사용하고, False인 경우, buffer를 사용한다. """
        if use_rollout_buffer:
            batch, max_episode_len = self.wm_runner.wm_buffer.sample(self.algo_args["algo"]["batch_size"])  # 실제 128개 뽑음. 롤아웃 갯수가 적으면 복원추출로 뽑음
            step_idx = torch.randint(0, max_episode_len, (self.algo_args["algo"]["batch_size"],), device=self.device)
            is_first = torch.zeros((self.algo_args["algo"]["batch_size"], max_episode_len), device=self.device).unsqueeze(-1)
            is_first[:, 0] = 1.0
            actions = batch['actions'].reshape(self.algo_args["algo"]["batch_size"], -1, (self.num_agents * self.action_spaces[0].shape[0]))
            target_post, _ = self.wm_runner.wm.dynamics.observe_efficient(batch['next_share_obs'], actions, is_first, step_idx)
            batch_indices = torch.arange(actions.shape[0], device=actions.device)
            
            sp_share_h_t = target_post['deter'][:, 0, :]
            sp_share_h_t_plus_1 = target_post['deter'][:, 1, :]
            sp_share_z_t = target_post['stoch'][:, 0, :]
            sp_share_z_t_plus_1 = target_post['stoch'][:, 1, :]
            
            sp_share_h_z_t = torch.cat([sp_share_h_t, sp_share_z_t], dim=-1)
            sp_share_h_z_t_plus_1 = torch.cat([sp_share_h_t_plus_1, sp_share_z_t_plus_1], dim=-1)
            
            sp_obs = batch['obs'][batch_indices, step_idx, :, :].transpose(1, 0)    # transpose결과 (n_agents, batch_size, dim)이 된다.
            sp_actions = batch['actions'][batch_indices, step_idx, :, :].transpose(1, 0)
            sp_available_actions = batch['available_actions'][batch_indices, step_idx, :, :].transpose(1, 0)
            sp_reward = batch['rewards'][batch_indices, step_idx, :]
            sp_done = batch['dones'][batch_indices, step_idx, 0, :]
            sp_valid_transition = batch['valid_transitions'][batch_indices, step_idx, :, :].transpose(1, 0)
            sp_term = batch['terms'][batch_indices, step_idx, :]
            # sp_next_share_obs = data['next_share_obs']
            sp_next_obs = batch['next_obs'][batch_indices, step_idx, :, :].transpose(1, 0)
            sp_next_available_actions = batch['next_available_actions'][batch_indices, step_idx, :, :].transpose(1, 0)
            sp_gamma = torch.full((self.algo_args["algo"]["batch_size"], 1), self.algo_args["algo"]["gamma"], device=self.device, dtype=torch.float32)
        else:
            data = self.buffer.sample()
            (
                sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
                sp_obs,  # (n_agents, batch_size, dim)
                sp_actions,  # (n_agents, batch_size, dim)
                sp_available_actions,  # (n_agents, batch_size, dim)
                sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
                sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
                sp_valid_transition,  # (n_agents, batch_size, 1)
                sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
                sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
                sp_next_obs,  # (n_agents, batch_size, dim)
                sp_next_available_actions,  # (n_agents, batch_size, dim)
                sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            ) = data
        
        """ train critic """
        self.critic.turn_on_grad()  # 부모 클래스의 마지막(twin_continuous_q_critic.py)에 있는 메소드. grad를 하나하나 켜준다.
        if self.args["algo"] == "hasac":
            """ actor을 이용하여 next_actions와 next_entropy_terms_critics를 구한다. """
            next_actions = []
            next_entropy_terms_critics = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                if self.tdd_runner is not None and self.tdd_args["train"]["use_state_entropy"]:
                    # 약 1000개의 sp_next_obs (n_rollout_threads, batch_size, obs의 차원)
                    # 각각의 스레드에 대해 temporal distance top k를 찾아야 한다.
                    # 현재 agent_wise로 잘 진행중에 있으며 그래서 건네줘야할 정보는 sp_next_obs[agent_id]랑면 될 듯?
                    next_entropy_terms_critics.append(-self.tdd_runner.calculate_central_state_entropy(sp_next_obs[agent_id], agent_id, step).unsqueeze(-1))
                else:
                    next_entropy_terms_critics.append(next_logp_action)
            
            """ NaN 체크를 위한 디버깅 코드 추가 """
            
            # numpy 배열인 경우 텐서로 변환 후 체크
            def check_nan(data, name):
                if isinstance(data, np.ndarray):
                    if np.isnan(data).any():
                        print(f"{name} has NaN!")
                        return True
                elif isinstance(data, torch.Tensor):
                    if torch.isnan(data.clone().detach()).any():
                        print(f"{name} has NaN!")
                        return True
                return False
            
            if check_nan(sp_actions, "sp_actions"):
                return None, None, None
            if check_nan(sp_reward, "sp_reward"):
                return None, None, None
            if check_nan(sp_done, "sp_done"):
                return None, None, None
            if check_nan(sp_valid_transition, "sp_valid_transition"):
                return None, None, None
            if check_nan(sp_term, "sp_term"):
                return None, None, None
            if check_nan(sp_gamma, "sp_gamma"):
                return None, None, None
            
            # 리스트인 경우 각 요소 체크
            for i, action in enumerate(next_actions):
                if torch.isnan(action.clone().detach()).any():
                    print(f"next_actions[{i}] has NaN!")
                    return None, None, None
            
            for i, entropy in enumerate(next_entropy_terms_critics):
                if torch.isnan(entropy.clone().detach()).any():
                    print(f"next_entropy_terms_critics[{i}] has NaN!")
                    return None, None, None
            
            """ 실제 크리틱 학습 하는 곳 -> soft_twin_continuous_q_critic.py로 간다. """
            if use_rollout_buffer:
                critic_loss = self.critic.train(
                sp_share_h_z_t,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_share_h_z_t_plus_1,
                next_actions,
                next_entropy_terms_critics,
                sp_gamma,
                self.value_normalizer,
            )
            else:
                critic_loss = self.critic.train(
                    sp_share_obs,
                    sp_actions,
                    sp_reward,
                    sp_done,
                    sp_valid_transition,
                    sp_term,
                    sp_next_share_obs,
                    next_actions,
                    next_entropy_terms_critics,
                    sp_gamma,
                    self.value_normalizer,
                    use_rollout_buffer
                )
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device) # 샘플링 된 에이전트들 생환 여부를 나타내는 텐서
        if self.total_it % self.policy_freq == 0:   # policy_freq는 1로 설정되어 있다. 즉, 매번 policy를 업데이트 한다.
            # train actors
            if self.args["algo"] == "hasac":
                actions = []
                entropy_terms_ls_actors = []
                alpha_losses = []  # 각 에이전트의 alpha_loss를 저장
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        if self.tdd_runner is None or self.tdd_args["train"]["use_actor_entropy"]:
                            action, entropy_term = self.actor[agent_id].get_actions_with_logprobs(
                                sp_obs[agent_id],
                                sp_available_actions[agent_id]
                                if sp_available_actions is not None
                                else None,
                            )
                        else:
                            if not self.tdd_args["train"]["use_state_entropy"]:
                                raise ValueError("use_state_entropy must be True when use_actor_entropy is False")
                            action = self.actor[agent_id].get_actions(
                                sp_obs[agent_id],
                                sp_available_actions[agent_id]
                                if sp_available_actions is not None
                                else None,
                            )
                            entropy_term = -self.tdd_runner.calculate_decentral_state_entropy(sp_obs[agent_id], agent_id, step).unsqueeze(-1)
                        actions.append(action)
                        entropy_terms_ls_actors.append(entropy_term)
                # actions shape: (n_agents, batch_size, dim)
                # entropy_terms_ls shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    # agent_order = list(range(self.num_agents))
                    agent_order = list(range(self.num_agents - 1, -1, -1))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                actor_loss_ls = [0 for _ in range(self.num_agents)]
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    if self.tdd_runner is None or self.tdd_args["train"]["use_actor_entropy"]:
                        actions[agent_id], entropy_terms_ls_actors[agent_id] = self.actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                    else:
                        if not self.tdd_args["train"]["use_state_entropy"]:
                            raise ValueError("use_state_entropy must be True when use_actor_entropy is False")
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        entropy_terms_ls_actors[agent_id] = -self.tdd_runner.calculate_decentral_state_entropy(sp_obs[agent_id], agent_id, step).unsqueeze(-1)
                        
                    if self.state_type == "EP":
                        entropy_term_agent_wise = entropy_terms_ls_actors[agent_id]
                        actions_t = torch.cat(actions, dim=-1)
                    elif self.state_type == "FP":
                        entropy_term_agent_wise = torch.tile(
                            entropy_terms_ls_actors[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                    if wm_runner is not None:
                        value_pred = self.critic.get_values(sp_share_h_z_t, actions_t)    # 여기에 다른 에이전트들의 액션도 포함시키기 위해 아까 torch.no_grad()로 일단 액션을 먼저 구한 것
                    else:
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)    # 여기에 다른 에이전트들의 액션도 포함시키기 위해 아까 torch.no_grad()로 일단 액션을 먼저 구한 것
                    if self.algo_args["algo"]["use_policy_active_masks"]:   # 이거 True
                        if self.state_type == "EP":
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * entropy_term_agent_wise)   # 왼쪽 self.alpha가 off_policy_base_runner.py에서 선언
                                    * sp_valid_transition[agent_id] # lopg_action은 aget_wise로 액터 엔트로피
                                )
                                / sp_valid_transition[agent_id].sum()   # batch_size만큼의 valid_transition이 있으므로, agent_id가 모든 경우에서 이를 모두 더해주면 batch_size가 된다.
                            )
                        elif self.state_type == "FP":
                            valid_transition = torch.tile(
                                sp_valid_transition[agent_id], (self.num_agents, 1)
                            )
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * entropy_term_agent_wise)
                                    * valid_transition
                                )
                                / valid_transition.sum()
                            )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * entropy_term_agent_wise
                        )
                    """ 액터 업데이트 """
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    actor_loss_ls[agent_id] = actor_loss.item()
                    
                    """ 알파 업데이트 """
                    if self.algo_args["algo"]["auto_alpha"]:    # 이거 True
                        log_prob = (
                            entropy_terms_ls_actors[agent_id].detach() # 확률밀도 함수의 로그값
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                        alpha_losses.append(alpha_loss.item())
                    else:
                        alpha_losses.append(0.0)
                    """ 다음 반복을 위한 액션 업데이트 """
                    actions[agent_id], _ = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    if self.tdd_runner is not None and self.tdd_args["train"]["use_state_entropy"] and not self.tdd_args["train"]["use_actor_entropy"]:
                        self.critic.update_alpha(next_entropy_terms_critics, np.sum(self.target_entropy)) # 이건 agnet_wise로 하면 안 되니까 np.sum()하는거다.
                    else:
                        self.critic.update_alpha(entropy_terms_ls_actors, np.sum(self.target_entropy)) # 이건 agnet_wise로 하면 안 되니까 np.sum()하는거다.
            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                # soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
        return critic_loss, actor_loss_ls, np.mean(alpha_losses) if alpha_losses else 0.0
