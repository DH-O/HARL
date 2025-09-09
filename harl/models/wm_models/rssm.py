import torch
import torch.nn as nn
import harl.utils.models_tools as tools
from torch import distributions as torchd
from torch.nn import functional as F
from harl.models.wm_models.GRU import GRUCell

class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
        role_config=None
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        inp_layers = []
        if role_config is not None:
            self.role_config = role_config
        else:
            self.role_config = {}
        # TODO: change input dim to include role embed dimension
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions + self.role_config.get("role_embed_size", 0)
        else:
            inp_dim = self._stoch + num_actions +  self.role_config.get("role_embed_size", 0)
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed # h_t, x_t를 인풋으로 받기 위함. 우리 코드의 경우, h_t, o_t가 될 것이다.
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":  # 학습된 초기화?
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )
        self.to(self._device)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe_step(self, prev_state, embed, action, is_first):
        """단일 스텝 observe 처리 - 이전 상태를 받아서 새로운 상태를 반환"""
        post, prior = self.obs_step(prev_state, action[:, 0, :], embed[:, 0, :], is_first[:, 0])    
        # 이 때 action은 주로 (n_threads, 1, action_dim)이다. 그래서 (n_threads, 1, action_dim)이 아니라 (n_threads, action_dim)으로 변환되어서 들어간다.
        
        return post, prior

    def observe(self, embed, action, is_first, state=None):
        
        """ 일단 차원부터 수정하고 """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first    # 여기서 prev_state[0]인 이유는 'posterior'를 가져오기 위함
            ),
            (action, embed, is_first),
            (state, state), # state는 오직 초기 상태 설정을 위함이라고 봐도 된다. (state, state)로 한건 불필요한 구현.
        )   # static_scan은 모든 시간 스텝에 대해서 함수를 적용하고, 결과를 반환한다.

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def observe_efficient(self, embed, action, is_first, target_idx=None, state=None):
        """
        target_idx: (batch_size, ) - 각 배치별로 관심 있는 인덱스
        """
        # 현재 배치에 있는 time idx중 최대값 산정
        max_time_idx = torch.max(target_idx).item()
        
        # 필요한 부분만 자르기
        truncated_embed = embed[:, :max_time_idx + 1, :]    # 자르기 전엔 (batch_size, n_timesteps, embed_size)
        truncated_action = action[:, :max_time_idx + 1, :]
        truncated_is_first = is_first[:, :max_time_idx + 1, :]
        
        """ 일단 차원부터 수정하고 """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        truncated_embed, truncated_action, truncated_is_first = swap(truncated_embed), swap(truncated_action), swap(truncated_is_first)
        
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (truncated_action, truncated_embed, truncated_is_first),
            (state, state),
            max_time_idx + 1
        )
        
        batch_indices = torch.arange(action.shape[0], device=action.device)
        
        # 각 배치별로 target_idx-1과 target_idx에 해당하는 timestep을 가져오기
        target_post = {}
        target_prior = {}
        
        for k, v in post.items():
            # v shape: (timesteps, batch_size, ...)
            # 각 배치별로 [target_idx[i]-1, target_idx[i]] 인덱스의 값을 가져오기
            t_minus_1 = v[target_idx - 1, batch_indices]  # (batch_size, ...)
            t_current = v[target_idx, batch_indices]      # (batch_size, ...)
            
            # 두 timestep을 합쳐서 (batch_size, 2, ...) 형태로 만들기
            target_post[k] = torch.stack([t_minus_1, t_current], dim=1)
        
        for k, v in prior.items():
            t_minus_1 = v[target_idx - 1, batch_indices]
            t_current = v[target_idx, batch_indices]
            target_prior[k] = torch.stack([t_minus_1, t_current], dim=1)
        
        return target_post, target_prior
    
    """ 추후 사용할 수도 있음 """
    # def imagine_with_action(self, action, state, role_embed = None):
    #     assert role_embed is not None if self.role_config is not None else True
    #     swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    #     assert isinstance(state, dict), state
    #     action = swap(action)
    #     if role_embed is not None:
    #         role_embed = swap(role_embed)
    #     prior = tools.static_scan(self.img_step, [action, role_embed], state)
    #     prior = prior[0]
    #     prior = {k: swap(v) for k, v in prior.items()}
    #     return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """rssm 한 스텝 관련된 거의 모든 것 """
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):  # Question: torch.sum(is_first) == len(is_first)의 경우 멀티쓰레드라서 있다는건데 잘 이해는 안 간다.
            prev_state = self.initial(len(is_first))    # {'mean': [n_rollout_threads, stoch], 'std': [n_rollout_threads, stoch], 'stoch': [n_rollout_threads, stoch], 'deter': [n_rollout_threads, deter]}
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )   # prev_action shape: (n_rollout_threads, action_dim)
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:   
            # 일부 스레드만 초기화 되는 경우를 고려해서 초기화 되지 않은 스레드는 이전 상태를 유지하도록 함.
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )
        # img_step에서 h_{t-1}, x_{t-1}, a_{t-1}을 받아서 h_t를 계산하고, 그걸 이용해 prior까지 계산
        prior = self.img_step(prev_state, prev_action)  # prior = {z^hat_t ~ p_phi(z^hat_t | h_t), h_t, mean, std}
        x = torch.cat([prior["deter"], embed], -1) # prior['deter'] is basically the output from the GRU conditioned on past state and action.
        # x = [h_t, x_t]
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)    # 확률 분포를 완전히 결정하는 최소한의 통계량을 뽑아 내기 위한 작업. mean, std 여기서 나온다.
        if sample:  # Question: 이거 목적이 뭔데
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}   # {z_t ~ q_phi(z_t | h_t, x_t), h_t, mean, std}
        return post, prior  # post와 prior에서 h_t 값들이 같은지 검증해볼 필요가 있다.

    def img_step(self, prev_state, prev_action, sample=True):
        """ 순차 모델 관련 """
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]    # prev_stoch = z_{t-1}, prev_action = a_{t-1}
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)    # x_{t-1} = [z_{t-1}, a_{t-1}]
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)  # GRU인풋에 맞도록 변환하는 작업 x_{t-1} -> embed_(x_{t-1})
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented -> 그래서 self._rec_depth를 1로 설정되어있음
            deter = prev_state["deter"] # h_{t-1}. 애초에 RSSM 논문 (1)식을 봐도 sequence 모델은 결정적 모델이다. 그래서 "deter"
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            _, deter = self._cell(x, [deter])   # GRU: h_t = f(h_{t-1}, embed_(x_{t-1}))
            deter = deter[0]  # Keras wraps the state in a list.    deter은 리스트인데 길이가 1밖에 안 된다. 케라스 스타일의 잔재로 보임.
        
        """ 다이나믹스 예측기 관련 """
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x) # h_t를 인풋으로 하여 z^hat_t를 계산해내려는 전초작업
        
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()   # z^hat_t ~ p_phi(z^hat_t | h_t)
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}   # {z^hat, h_t, mean, std}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld( # RSSM논문에서 rep_loss 즉 dynamics predictor를 스탑그레디언트 먹이고 encoder 학습
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld( # RSSM논문에서 dyn_loss 즉 encoder를 스탑그레디언트 먹이고 dynamics predictor 학습
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)   # KL 값이 너무 작아지는 것을 방지. Free Bits 기법이며, 정보 보존을 보장한다.
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss  # 이걸 따로 리턴하는 것은 value가 너무 작으면 정보 손실을 의심하고, rep_loss가 항상 1이면 free bits가 과도하단 것을 의미