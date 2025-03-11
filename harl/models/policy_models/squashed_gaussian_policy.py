import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianPolicy(nn.Module):
    """Squashed Gaussian policy network for HASAC."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize SquashedGaussianPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"] # [256, 256]
        activation_func = args["activation_func"]   # relu
        final_activation_func = args["final_activation_func"]   #tanh
        obs_shape = get_shape_from_obs_space(obs_space) # 18
        if len(obs_shape) == 3: # 1이겠지. 3인건 이미지인 경우
            self.feature_extractor = PlainCNN(
                obs_shape, hidden_sizes[0], activation_func
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]  # feature dim: 인풋의 차원
        act_dim = action_space.shape[0]
        self.net = PlainMLP(
            [feature_dim] + list(hidden_sizes), activation_func, final_activation_func
        )   # hidden_sizes가 리스트가 아닌 경우도 있나 보다.
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = action_space.high[
            0
        ]  # action limit for clamping (assumes all dimensions share the same bound)    
        # action_space: Box(0.0, 1.0, (5,), float32)인데 action_space.hiigh하면 array([1., 1., 1., 1., 1.], dtype=float32)이 나옴. 그래서 [0]을 해주는 것.
        self.to(device) # nn.Module의 메소드. 모델을 device로 보냄.

    def forward(self, obs, stochastic=True, with_logprob=True):
        # Return output from network scaled to action space limits.
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        net_out = self.net(x)   # 이게 진짜 재밌는게 256차원이 마지막 아웃풋인데, 마지막 activation function이 tanh이다. 
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if not stochastic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()   # rsample은 reparameterized sample이다. 이걸 쓰면 backpropagation이 가능해진다.

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)    
            # 액션 a (=pi_action)가 각 차원별로 독립적이라고 가정하면, 전체 로그 a 확률은 개별 차원의 로그 확률의 합과 같다. (p(a_1) * ... * p(a_n) = p(a))
            #.log_prob()은 -1/2 * (a - mu)^2 / sigma^2 - log(2 * pi * sigma)를 반환한다. 이는 log(p(a))와 같다.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1, keepdim=True
            )   # F.softplus(x) = log(1 + exp(x))이다. 이건 ReLU와 비슷한데, x가 큰 음수일 때 0으로 수렴하는 것이 아니라 x로 수렴한다.
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action  # Scale to [-act_limit, act_limit]

        return pi_action, logp_pi
