"""TDD Intrinsic Reward Model for multi-agent environments with Hard‑Negative Mining (HNM)
및 Positive Sampling Restriction (PSR) 근데 PSR은 안 쓰는게 좋겠다."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter

def discounted_sampling(ranges, discount):
    assert 0 <= discount <= 1
    seeds = torch.rand(size=ranges.shape, device=ranges.device)
    if discount == 0:
        samples = torch.zeros_like(seeds, dtype=ranges.dtype, device=ranges.device)
    elif discount == 1:
        samples = torch.floor(seeds * ranges).int()
    else:
        samples = torch.log(1 - (1 - discount**ranges) *seeds) / np.log(discount)
        samples = torch.min(torch.floor(samples).long(), ranges - 1)
    return samples

def mrn_distance(x, y, batch_mode=False):
    """Metric Residual Network (MRN) distance.
    Args:
        x: (torch.Tensor) tensor of shape (batch_size, dim) or (dim) in non-batch mode
        y: (torch.Tensor) tensor of shape (batch_size, dim) or (dim) in non-batch mode
        batch_mode: (bool) whether to compute distances in batch mode
    Returns:
        distance: (torch.Tensor) tensor of shape (batch_size) or scalar in non-batch mode
    """
    eps = 1e-6
    
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    
    # 벡터화된 연산
    max_component = torch.max(F.relu(x_prefix - y_prefix), axis=-1).values
    l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    
    distance = max_component + l2_component
    
    return distance

class PotentialNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PotentialNet, self).__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

        self.value = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, obs: torch.Tensor, image=False):
        # obs: [N, H, W, C] -> [N, C, H, W] -> [N, 1]
        if image:
            obs = obs.permute(0, 3, 1, 2)
            state = self.img_encoder(obs)
        else:
            state = obs
        value = self.value(state)
        return value

class S_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(S_Encoder, self).__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.Linear(input_dim , latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, obs: torch.Tensor):
        # obs: [N, H, W, C] -> [N, C, H, W] -> [N, 1]
        # obs = obs.permute(0, 3, 1, 2)
        # state = self.img_encoder(obs)
        value = self.value(obs)
        return value

class TDDModel:
    def __init__(self, args, num_agents, input_dim, device=torch.device("cuda"), run_dir=None):
        self.args = args
        
        # 네트워크
        self.input_dim = input_dim
        self.latents_dim = self.args["network"]["latents_dim"]
        self.output_dim = self.args["network"]["output_dim"]
        self.device = device
        
        if self.args["network"]["use_independent_nets"]:
            self.potential_net = [PotentialNet(self.input_dim, self.latents_dim).to(device) for _ in range(num_agents)]
            self.s_encoder = [S_Encoder(self.input_dim, self.latents_dim, self.output_dim).to(device) for _ in range(num_agents)]
        else:
            self.potential_net = PotentialNet(self.input_dim, self.latents_dim).to(device)
            self.s_encoder = S_Encoder(self.input_dim, self.latents_dim, self.output_dim).to(device)
        
        # 기본 TDD 설정
        self.warmup_steps = self.args["train"]["warmup_steps"]
        self.learning_steps = self.args["train"]["learning_steps"]
        
        self.warmup_logging_interval = self.args["train"]["warmup_logging_interval"]
        self.learning_logging_interval = self.args["train"]["learning_logging_interval"]
        
        self.batch_size = self.args["train"]["batch_size"]
        
        self.learning_rate = float(self.args["train"]["learning_rate"])
        
        self.tdd_discount = self.args["tdd"]["tdd_discount"]
        
        if self.args["network"]["use_independent_nets"]:
            self.optimizer = [torch.optim.Adam(
                [
                    {"params": self.potential_net[agent_id].parameters(), "lr": self.learning_rate},
                    {"params": self.s_encoder[agent_id].parameters(), "lr": self.learning_rate}
                ]
            ) for agent_id in range(num_agents)]
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.potential_net.parameters(), "lr": self.learning_rate},
                    {"params": self.s_encoder.parameters(), "lr": self.learning_rate}
                ]
            )
        # Tensorboard 설정
        if run_dir is not None:
            self.run_dir = run_dir
        else:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = f"runs/TDD/{ts}"
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
        )
                    
    def update(self, data, is_warm_up=False): # data의 차원: (n_agent, episode 수, max_cycles, dict, n_rollout_threads, 2차원(s_t))
        obss = [[] for _ in range(len(data))]
        next_obss = [[] for _ in range(len(data))]
        n_trajs = [[] for _ in range(len(data))]
        n_cum_steps = [[] for _ in range(len(data))]
        n_threads = [[] for _ in range(len(data))]
        
        metrics = {}
        
        """ 모든 에이전트에 대한 데이터 후처리 """
        for agent_id in range(len(data)):
            obss[agent_id] = np.array([[step['obs'] for step in episode] for episode in data[agent_id]], dtype=np.float32)  # (n_episode, max_cycles, n_rollout_threads, 2)
            # data[0][0][0]['obs'].shape가 (1,1,2)f로 나오긴 했는데 음
            obss[agent_id] = torch.from_numpy(obss[agent_id]).to(self.device)
            next_obss[agent_id] = np.array([[step['next_obs'] for step in episode] for episode in data[agent_id]], dtype=np.float32)  # (n_episode, max_cycles, n_rollout_threads, 2) 
            next_obss[agent_id] = torch.from_numpy(next_obss[agent_id]).to(self.device)
            n_trajs[agent_id], n_cum_steps[agent_id], n_threads[agent_id] = obss[agent_id].shape[:3]
        # 설마 에이전트별 데이터 수가 다른 경우에 대한 예외처리
        if n_trajs[0] != n_trajs[1] or n_cum_steps[0] != n_cum_steps[1] or n_threads[0] != n_threads[1]\
            or n_trajs[0] != n_trajs[2] or n_cum_steps[0] != n_cum_steps[2] or n_threads[0] != n_threads[2]\
                or n_trajs[1] != n_trajs[2] or n_cum_steps[1] != n_cum_steps[2] or n_threads[1] != n_threads[2]:
            raise ValueError(f"n_trajs, n_steps, n_threads must be the same for all agents. \
                             n_trajs[0]: {n_trajs[0]}, n_trajs[1]: {n_trajs[1]}, n_trajs[2]: {n_trajs[2]}\
                             n_cum_steps[0]: {n_cum_steps[0]}, n_cum_steps[1]: {n_cum_steps[1]}, n_cum_steps[2]: {n_cum_steps[2]}\
                             n_threads[0]: {n_threads[0]}, n_threads[1]: {n_threads[1]}, n_threads[2]: {n_threads[2]}")
        
        for i in range(self.warmup_steps if is_warm_up else self.learning_steps):
            total_loss = 0.0
            for agent_id in range(len(data)):
                thread_metrics_list = []
                for thread_idx in range(n_threads[0]):
                    # Sample mini-batch data (positive pairs)
                    traj_idx = torch.randint(n_trajs[0], (self.batch_size,), device=self.device)   # 0 ~ n_trajs-1 중 랜덤 선택
                    step_idx = torch.randint(n_cum_steps[0], (self.batch_size,), device=self.device)  # 0 ~ n_steps-1 중 랜덤 선택
                    intervals = discounted_sampling(
                        n_cum_steps[0] - step_idx, self.tdd_discount
                    )
                    
                    # 현재 상태와 다음 상태 추출 (특정 thread에 대해서만)
                    obs = obss[agent_id][traj_idx, step_idx, thread_idx]  # (batch_size, 2)
                    goal = next_obss[agent_id][traj_idx, step_idx + intervals, thread_idx]  # (batch_size, 2)
                    
                    c_g = self.potential_net[agent_id](goal)
                    phi_s = self.s_encoder[agent_id](obs)
                    phi_g = self.s_encoder[agent_id](goal)
                    
                    mrn_dists = mrn_distance(phi_s[:, None], phi_g[None, :])    # 대각선 값이 양의 샘플이다. s_t 와 g_t사이의 거리를 재는 것이기 때문이다.
                    # 위와 같이 함으로써 phi_g 각각에 대한 모든 phi_s와의 거리를 계산하며, 그게 모든 행에서 나타난다.
                    logits = c_g.T - mrn_dists
                    I = torch.eye(self.batch_size, device=self.device)
                    
                    # Contrastive Loss
                    contrastive_loss = (F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2

                    # 전체 loss: contrastive
                    total_loss += contrastive_loss

                    self.optimizer[agent_id].zero_grad()
                    contrastive_loss.backward()
                    self.optimizer[agent_id].step()
                    
                    # grad norm 계산 및 출력
                    def get_grad_norm(model):
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        return total_norm ** 0.5

                    s_grad_norm = get_grad_norm(self.s_encoder[agent_id])
                    p_grad_norm = get_grad_norm(self.potential_net[agent_id])
                    
                    # 각 스레드별로 메트릭 계산 / 추후 모든 스레드의 메트릭에서 agent_id 고려할거니 여기서는 그냥 저장만
                    current_metrics = {
                        'loss/contrastive_loss': contrastive_loss.item(),
                        'loss/total_loss': total_loss.item(),
                        
                        # 그래디언트 관련 메트릭
                        'gradients/s_encoder_norm': s_grad_norm,
                        'gradients/potential_net_norm': p_grad_norm,
                    }
                    thread_metrics_list.append(current_metrics)
                    
                if i % (self.warmup_logging_interval if is_warm_up else self.learning_logging_interval) == 0:
                    # 모든 스레드의 메트릭 평균 계산
                    avg_thread_metrics = {}
                    for key in thread_metrics_list[0].keys():
                        values = [metrics[key] for metrics in thread_metrics_list]
                        avg_thread_metrics[f'agent_{agent_id}/{key}'] = np.mean(values)
                    
                    metrics.update(avg_thread_metrics)
            
            if i % (self.warmup_logging_interval if is_warm_up else self.learning_logging_interval) == 0:
                print(f"Step {i} - Average Loss: {total_loss / (len(data) * n_threads[0])}")
                # 전체 에이전트의 평균 메트릭 계산
                avg_metrics = {
                    'avg/contrastive_loss': np.mean([metrics[f'agent_{j}/loss/contrastive_loss'] for j in range(len(data))]),
                    'avg/total_loss': np.mean([metrics[f'agent_{j}/loss/total_loss'] for j in range(len(data))]),
                    'avg/s_encoder_grad_norm': np.mean([metrics[f'agent_{j}/gradients/s_encoder_norm'] for j in range(len(data))]),
                    'avg/potential_net_grad_norm': np.mean([metrics[f'agent_{j}/gradients/potential_net_norm'] for j in range(len(data))]),
                }
                metrics.update(avg_metrics)
                
                # 텐서보드에 메트릭 기록
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, i)