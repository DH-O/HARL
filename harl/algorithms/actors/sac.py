import torch
from harl.models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase

class SAC(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.actor = SquashedGaussianPolicy(args, obs_space, act_space, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()
    
    def get_actions(self, obs, stochastic=True):
        obs = check(obs).to(**self.tpdv)
        actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
        return actions

    def get_actions_with_logprobs(self, obs, stochastic=True):
        obs = check(obs).to(**self.tpdv)
        actions, logp_actions = self.actor(obs, stochastic=stochastic, with_logprob=True)
        return actions, logp_actions

    def save(self, save_dir, id):
        torch.save(self.actor.state_dict(), str(save_dir) + f"/actor_agent{id}.pt")
    
    def restore(self, model_dir, id):
        actor_state_dict = torch.load(str(model_dir) + f"/actor_agent{id}.pt")
        self.actor.load_state_dict(actor_state_dict)