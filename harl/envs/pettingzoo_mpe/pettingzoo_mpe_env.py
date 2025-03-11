import copy
import importlib
import logging
import numpy as np
import supersuit as ss

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMPEEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]    # args["scenario"] = "simple_spread_v2", 그러니까 pettingzoo에서도 simple_spread_v2를 사용하겠다는 의미
        del self.args["scenario"]   # 굳이 왜 지우는진 몰라도 암튼 self.args에는 "scenario"가 없어짐
        self.discrete = True
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] == True
        ):
            self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26    # self.cur_step = 0으로 하기 위해 1을 더해줬다고 생각해도 됨
        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)    
        # import pettingzoo.mpe.simple_spread_v2를 self.module에 저장해서 동적으로 불러오겠다는 의미
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))   # ... make_env -> ... raw_env ... -> reset_world
        )
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)    # 전역 state_space를 self.n_agents만큼 반복해서 저장
        self.observation_space = self.unwrap(self.env.observation_spaces)    # 각 agent의 observation_space를 하나의 리스트로 저장
        self.action_space = self.unwrap(self.env.action_spaces)              # 각 agent의 action_space를 하나의 리스트로 저장
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions)) # self.wrap(actions)는 actions를 agent별로 나눠서 저장한 것
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}  # trunc가 term보다 우선순위가 높음
            for agent in self.agents:
                info[agent]["bad_transition"] = True    # bad_transition은 self.cur_step == self.max_cycles일 때만 True, 즉 원하는 종료가 아니다.
                # 재밌는 점은 simple_spread_v2에서는 무조건 bad_transition이 True로 끝난다.
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.state())   # 이게 share_obs. 놀랍게도 그냥 글로벌 state를 self.n_agents만큼 반복해서 저장한 것
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(seed=self._seed))
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()
        return self.env.render()    # 이거 리턴 굳이 왜 만들었는지 기억이 나질 않는다.

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
