from harl.common.buffers.tdd_rollout_buffer import RolloutBuffer
from harl.algorithms.representation.tdd import TDDModel

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
        self.device = None
        
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents
        self.observation_space = observation_space
        
        self.tdd_model = TDDModel(self.tdd_args, 2)
        
        # 보상 계산 관련 변수 초기화
        self.reward_update_counter = 0
        self.last_reward_update = None
        self.prev_obs = None
        
        self.rollout_buffer = RolloutBuffer(
                {**self.tdd_args["network"], **self.tdd_args["train"], **self.tdd_args["tdd"]},
                self.observation_space,
                self.num_agents,
                self.n_rollout_threads
        )