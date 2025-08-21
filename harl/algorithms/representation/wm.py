
import torch 
from harl.models.wm_models.enc_dec import MultiEncoder, MultiDecoder
from harl.models.wm_models.rssm import RSSM
from torch import nn
from harl.utils.models_tools import RequiresGrad, WM_Optimizer

to_np = lambda x: x.detach().cpu().numpy()
class DreamerWorldModel(nn.Module):
    def __init__(self, obs_dim, action_spaces, config):
        super(DreamerWorldModel, self).__init__()
        self._use_amp = True if config["precision"] == 16 else False
        self._config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_spaces = action_spaces
        act_shape_for_net = action_spaces[0].shape[0]
        
        shapes_for_net = {'vector_obs': [obs_dim]}
        self.encoder = MultiEncoder(shapes_for_net, **config["encoder"], use_wm_with_obs=config["use_wm_with_obs"])
        self.embed_size = self.encoder.outdim
        
        self.dynamics = RSSM(
            config["dyn_stoch"],
            config["dyn_deter"],
            config["dyn_hidden"],
            config["dyn_rec_depth"],
            config["dyn_discrete"],
            config["act"],
            config["norm"],
            config["dyn_mean_act"],
            config["dyn_std_act"],
            config["dyn_min_std"],
            config["unimix_ratio"],
            config["initial"],
            act_shape_for_net,
            self.embed_size,
            self.device
        )
        self.heads = nn.ModuleDict()
        if config["dyn_discrete"]:  # False임. 왜냐면 우리 코드에서는 디스크리트 액션을 사용하지 않기 때문이다.
            feat_size = config["dyn_stoch"] * config["dyn_discrete"] + config["dyn_deter"]
        else:
            feat_size = config["dyn_stoch"] + config["dyn_deter"]
        
        self.heads["decoder"] = MultiDecoder(
            feat_size, shapes_for_net, **config["decoder"]
        )   # h_t, z_t를 인풋으로 받고 우리 코드의 경우 o_t를 출력.
        for name in config["grad_heads"]:
            assert name in self.heads, name
        self._model_opt = WM_Optimizer(
            "model",
            self.parameters(),
            config["model_lr"],
            config["opt_eps"],
            config["grad_clip"],
            config["weight_decay"],
            opt=config["opt"],
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict()

    # def predict(self, data):
    #     data = self.preprocess(data)
    #     embed = self.encoder(data)
    #     states, _ = self.dynamics.observe(
    #         embed, data["action"], data["is_first"]
    #     )
    #     return self.heads["decoder"](self.dynamics.get_feat(states))['vector_obs']
    
    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with RequiresGrad(self):
            if self._use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    """ 인코더 통해서 임베딩을 뽑는데 구성이 어떻게 되려나? """
                    embed = self.encoder(data)  # embed의 shape은 (batch_size, max_episode_length, embed_size) 이런식으로 될거다.
                    # TODO : Include role information
                    role_embed_dyn = None 
                    role_embed_dec = None
                    
                    """ rssm 네트워크 불러와서 .observe하는데 """
                    post, prior = self.dynamics.observe(
                        embed, data["action"], data["is_first"], role_embed_dyn
                    )
                    kl_free = self._config["kl_free"]
                    dyn_scale = self._config["dyn_scale"]
                    rep_scale = self._config["rep_scale"]
                    
                    """ 다이나믹스 모델 학습을 위한 kl_loss 계산 """
                    kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                        post, prior, kl_free, dyn_scale, rep_scale
                    )   # 여기서 rep_loss는 representation loss
                    assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                    
                    """ Decoder 모델 학습을 위한 로스 계산. reward predictor, continue predictor는 멀티에이전트 환경에서는 사용하지 않는다. """
                    preds = {}
                    for name, head in self.heads.items():
                        grad_head = name in self._config["grad_heads"]
                        feat = self.dynamics.get_feat(post) # [h_t, z_t]
                        if self._config["decode_role"]:
                            feat = torch.cat([feat, role_embed_dec.detach()], dim=-1) if role_embed_dec is not None else feat
                        feat = feat if grad_head else feat.detach()
                        pred = head(feat)   # p_phi(x^hat_t | h_t, z_t)
                        if type(pred) is dict:
                            preds.update(pred)
                        else:
                            preds[name] = pred
                    losses = {}
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    scaled = {
                        key: value * self._scales.get(key, 1.0)
                        for key, value in losses.items()
                    }
                    model_loss = sum(scaled.values()) + kl_loss
                    model_loss = model_loss * data["mask"]
            else:
                """ 인코더 통해서 임베딩을 뽑음 """ 
                # data는 딕셔너리며, {'vector_obs': (batch_size, max_episode_length + 1, obs_dim), 
                # 'action': (batch_size, max_episode_length + 1, act_dim), 'is_first': (batch_size, max_episode_length + 1), 
                # 'mask': (batch_size, max_episode_length + 1)} 이런식으로 되어있다.
                embed = self.encoder(data)  # embed의 shape은 (batch_size, max_episode_length, embed_size) 이런식으로 될거다.
                # TODO : Include role information
                role_embed_dyn = None 
                role_embed_dec = None
                
                """ rssm 네트워크 불러와서 .observe하는데, 이게 imagine step """
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"], role_embed_dyn
                )
                kl_free = self._config["kl_free"]
                dyn_scale = self._config["dyn_scale"]   # 0.5
                rep_scale = self._config["rep_scale"]   # 0.1
                
                """ 다이나믹스 모델 학습을 위한 kl_loss 계산 """
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )   # 여기서 rep_loss는 representation loss
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                
                """ Decoder 모델 학습을 위한 로스 계산. reward predictor, continue predictor는 멀티에이전트 환경에서는 사용하지 않는다. """
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config["grad_heads"]
                    feat = self.dynamics.get_feat(post) # [h_t, z_t]
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)   # p_phi(x^hat_t | h_t, z_t)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                """ 아래가 recon_loss 계산 """
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                model_loss = model_loss * data["mask"]
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss * data["mask"]) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss * data["mask"])
        metrics["rep_loss"] = to_np(rep_loss * data["mask"])
        metrics["kl"] = to_np(torch.mean(kl_value * data["mask"]))
        
        if self._use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                metrics["prior_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(prior).entropy() * data["mask"])
                )
                metrics["post_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(post).entropy() * data["mask"])
                )
                context = dict(
                    embed=embed,
                    feat=self.dynamics.get_feat(post),
                    kl=kl_value,
                    postent=self.dynamics.get_dist(post).entropy(),
                )
        else:
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy() * data["mask"])
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy() * data["mask"])
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        # We dont have images
        obs = {k: torch.Tensor(v).to(self.device) for k, v in obs.items()}
        return obs
    
    """ 아래 주석들은 추후 사용할 수도 있음 """
    # def predict_test(self, data):
    #     data = self.preprocess(data)
    #     embed = self.encoder(data)

    #     states, _ = self.dynamics.observe(
    #         embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
    #     )
    #     # TODO: Remove the image portion
    #     recon = self.heads["decoder"](self.dynamics.get_feat(states))["vector_obs"].mode()[
    #         :6
    #     ]
    #     #reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
    #     init = {k: v[:, -1] for k, v in states.items()}
    #     prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
    #     openl = self.heads["decoder"](self.dynamics.get_feat(prior))["vector_obs"].mode()
    #     #reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
    #     # observed image is given until 5 steps
    #     model = torch.cat([recon[:, :5], openl], 1)
    #     truth = data["image"][:6]
    #     model = model
    #     error = (model - truth + 1.0) / 2.0

    #     return torch.cat([truth, model, error], 2) 

# class SimpleWorldModel(nn.Module):
    
#     def __init__(self,obs_space, act_space, step, config, role_config = None) -> None:
#         super().__init__()
#         self._step = step
#         self.obs_space = obs_space
        
        
        
    