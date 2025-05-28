"""TDD Intrinsic Reward Model for multi-agent environments with Hard‑Negative Mining (HNM)
및 Positive Sampling Restriction (PSR) 근데 PSR은 안 쓰는게 좋겠다."""


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
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
            # nn.LayerNorm(1024), # 원래 생략
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            # nn.LayerNorm(1024), # 원래 생략
            nn.ReLU(),
            # nn.Linear(1024, 1024), # 원래 생략
            # nn.LayerNorm(1024), # 원래 생략
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, obs: torch.Tensor):
        # obs: [N, H, W, C] -> [N, C, H, W] -> [N, 1]
        # obs = obs.permute(0, 3, 1, 2)
        # state = self.img_encoder(obs)
        value = self.value(obs)
        return value

class TDDModel:
    def __init__(self, args, input_dim, device=torch.device("cuda"), run_dir=None):
        self.args = args
        
        # 네트워크
        self.input_dim = input_dim
        self.latents_dim = self.args["network"]["latents_dim"]
        self.output_dim = self.args["network"]["output_dim"]
        self.device = device
        
        self.potential_net = PotentialNet(self.input_dim, self.latents_dim).to(device)
        self.s_encoder = S_Encoder(self.input_dim, self.latents_dim, self.output_dim).to(device)
        
        # 기본 TDD 설정
        self.warmup_steps = self.args["train"]["warmup_steps"]
        self.learning_steps = self.args["train"]["learning_steps"]
        self.learning_logging_interval = self.args["train"]["learning_logging_interval"]
        self.warmup_logging_interval = self.args["train"]["warmup_logging_interval"]
        
        self.batch_size = self.args["train"]["batch_size"]
        
        self.max_grad_norm = float(self.args["train"]["max_grad_norm"])
        self.learning_rate = float(self.args["train"]["learning_rate"])
        
        self.max_grad_norm_init = float(self.args["train"].get("max_grad_norm_init", 0.01))
        self.max_grad_norm_final = float(self.args["train"].get("max_grad_norm_final", 0.0001))
        self.grad_norm_decay_type = self.args["train"].get("grad_norm_decay_type", "linear")
        self.lr_init = float(self.args["train"].get("learning_rate_init", 1e-5))
        self.lr_final = float(self.args["train"].get("learning_rate_final", 1e-7))
        self.lr_decay_type = self.args["train"].get("lr_decay_type", "linear")
        
        # HNM & PSR 설정
        # self.max_pos_steps = self.args["tdd"]["max_pos_steps"]
        # self.hard_neg_thr = self.args["tdd"]["hard_neg_thr"]
        # self.hard_neg_k = self.args["tdd"]["hard_neg_k"]
        self.hard_neg_thr = None
        self.hard_neg_k = None
        self.tdd_discount = self.args["tdd"]["tdd_discount"]
        
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.potential_net.parameters(), "lr": self.lr_init},
                {"params": self.s_encoder.parameters(), "lr": self.lr_init}
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
        
    # def get_scheduled_value(self, step, init, final, decay_type="linear"):
    #     if decay_type == "linear":
    #         ratio = min(step / self.warmup_steps, 1.0)
    #         return init + (final - init) * ratio
    #     elif decay_type == "cosine":
    #         ratio = min(step / self.warmup_steps, 1.0)
    #         cosine = 0.5 * (1 + math.cos(math.pi * ratio))
    #         return final + (init - final) * cosine
    #     else:
    #         return init

    # def _select_hard_negatives(self, dists):
    #     """ Return boolean mask (BxB) where True = hard negative """
    #     B = dists.shape[0]
    #     I = torch.eye(B, device=dists.device, dtype=torch.bool) # 이것만 보면 양의 샘플들에 대한 마스킹
    #     neg_d = dists.clone()
    #     neg_d[I] = 1e6
    #     mask = torch.zeros_like(dists, dtype=torch.bool)    # False로 초기화된 BxB 텐서
        
    #     # 거리값 임계 조건
    #     if self.hard_neg_thr is not None:
    #         mask |= neg_d < self.hard_neg_thr   # neg_d < hard_neg_thr 인 경우 True로 마스킹
        
    #     # 상위 k개 조건
    #     if self.hard_neg_thr is not None:
    #         topk = torch.topk(
    #             -neg_d, k=min(self.hard_neg_k, B - 1), dim=1     # 가장 작은 값부터 찾기에 -neg_d를 사용. 여기서 dim=1은 열을 의미하지만 결국 각 행에서 가장 작은 값을 찾겠다는 뜻으로 해석해야 한다.
    #             ).indices   # 각 행마다 column index가 반환된다. 결국 shape는 (batch_size, hard_neg_k)이다.
    #         row_idx = torch.arange(B, device=dists.device).unsqueeze(1).expand_as(topk) # 아까 구한 topk의 각 행에 대한 row index를 구한다. 얘와 같이 사용하면 mask의 크기는 (batch_size, hard_neg_k)이 된다.
    #         mask[row_idx, topk] = True  # row_idx도 (B, topk), topk도 (B, topk)이므로 결국 (B, B) 크기의 텐서가 된다. topk의 모든 행에 대한 열 인덱스에 대해 0, 1, 2, .. 맞춰주려고 row_idx 쓴거다.
    #     return mask & ~I    # 행여라도 대각선 True가 있을까봐 다시 False로 마스킹
                    
    def update(self, data, step=None, total_steps=None, is_warm_up=False): # data의 차원: (n_agent, episode 수, max_cycles, dict, n_rollout_threads, 2차원(s_t))
        obss = [[] for _ in range(len(data))]
        next_obss = [[] for _ in range(len(data))]
        
        # 단일 에이전트로 데이터 추출
        obss[0] = np.array([[step['obs'] for step in episode] for episode in data[0]])  # (n_episode, max_cycles, n_rollout_threads, 2)
        obss[0] = torch.from_numpy(obss[0]).to(self.device)
        next_obss[0] = np.array([[step['next_obs'] for step in episode] for episode in data[0]])  # (n_episode, max_cycles, n_rollout_threads, 2)
        next_obss[0] = torch.from_numpy(next_obss[0]).to(self.device)
        n_trajs, n_steps, n_threads = obss[0].shape[:3]
        
        print("TDD Training 시작")
        metrics = {}
        start_time = time.time()
        
        for i in range(self.warmup_steps if is_warm_up else 4):
            total_loss = 0.0
            # # learning rate, grad norm 스케줄 적용
            # cur_lr = self.get_scheduled_value(i, self.lr_init, self.lr_final, self.lr_decay_type)
            # cur_grad_norm = self.get_scheduled_value(i, self.max_grad_norm_init, self.max_grad_norm_final, self.grad_norm_decay_type)
            # # optimizer의 learning rate 동적 변경
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = cur_lr
            
            for thread_idx in range(n_threads):
                # Sample mini-batch data (positive pairs)
                traj_idx = torch.randint(n_trajs, (self.batch_size,), device=self.device)   # 0 ~ n_trajs-1 중 랜덤 선택
                step_idx = torch.randint(n_steps, (self.batch_size,), device=self.device)  # 0 ~ n_steps-1 중 랜덤 선택
                intervals = discounted_sampling(
                    n_steps - step_idx, self.tdd_discount
                )
                # intervals = torch.clamp(intervals, min=1, max=self.max_pos_steps)
                
                # 현재 상태와 다음 상태 추출 (특정 thread에 대해서만)
                obs = obss[0][traj_idx, step_idx, thread_idx]  # (batch_size, 2)
                goal = next_obss[0][traj_idx, step_idx + intervals, thread_idx]  # (batch_size, 2)
                
                c_g = self.potential_net(goal)
                phi_s = self.s_encoder(obs)
                phi_g = self.s_encoder(goal)
                
                mrn_dists = mrn_distance(phi_s[:, None], phi_g[None, :])    # 대각선 값이 양의 샘플이다. s_0 와 g_0사이의 거리를 재는 것이기 때문이다.
                logits = c_g.T - mrn_dists
                I = torch.eye(self.batch_size, device=self.device)
                
                """ Hard Negative Masking """
                if self.hard_neg_thr is not None and self.hard_neg_k is not None:
                    print("Hard Negative Masking 적용")
                    hn_mask = self._select_hard_negatives(mrn_dists)    # (batch_size, batch_size), 이렇게 하면 hard negative가 있는 경우만 True로 마스킹된다.
                    valid_mask = hn_mask | I.bool() # 양의 샘플링, 즉 대각선도 다시 트루로 표시
                    # fallback: if 어떤 row도 hard negative가 없으면 모든 쌍을 유지
                    rows_no_hn = (~hn_mask).all(dim=1)  # 하드 네가티브가 없는 행을 찾는다. 얘의 결과 shape는 (batch_size,)이다. 각 행마다 모든 열값이 트루일 때만 트루가 출력되기에, rows_no_hn이 (batch_size,)의 모든 값이 True라면 hn_mask가 모두 False인 것이다.
                    if rows_no_hn.any():
                        print(f"하드 네거티브가 없는 행이 발견되었습니다. 해당 행은 모든 네거티브를 사용하도록 fallback합니다.")    # fallback: 원래 시도한 방법이 통하지 않을 때 대신 사용되는 예비 방법. (=대체경로, 안전망 등)
                        valid_mask[rows_no_hn] = True
                    logits_masked = logits.clone()
                    logits_masked[~valid_mask] = -float('inf')
                else:
                    logits_masked = logits
                    
                # Contrastive Loss
                contrastive_loss = (F.cross_entropy(logits_masked, I) + F.cross_entropy(logits_masked.T, I)) / 2

                # 전체 loss: contrastive + margin loss (가중치 조정 가능)
                loss = contrastive_loss

                total_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.s_encoder.parameters(), cur_grad_norm)
                # torch.nn.utils.clip_grad_norm_(self.potential_net.parameters(), cur_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.s_encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.potential_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # grad norm 계산 및 출력
                def get_grad_norm(model):
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    return total_norm ** 0.5

                s_grad_norm = get_grad_norm(self.s_encoder)
                p_grad_norm = get_grad_norm(self.potential_net)
                
                # 각 thread별로 메트릭 기록
                if i % (self.warmup_logging_interval if is_warm_up else self.learning_logging_interval) == 0:
                    thread_metrics = {
                        f'thread_{thread_idx}/contrastive/contrastive_loss': contrastive_loss.item(),
                        f'thread_{thread_idx}/contrastive/categorical_accuracy': torch.mean((torch.argmax(logits, axis=1) ==
                                                                        torch.arange(self.batch_size, device=self.device)).float()).item(),
                        f'thread_{thread_idx}/contrastive/logits_pos': torch.diag(logits).mean(),
                        f'thread_{thread_idx}/contrastive/logits_neg': torch.mean(logits * (1 - I)),
                        f'thread_{thread_idx}/contrastive/logits_logsumexp': torch.mean((torch.logsumexp(logits, axis=1)**2)),
                        f'thread_{thread_idx}/contrastive/c_g_pos': torch.diag(c_g).mean(),
                        f'thread_{thread_idx}/contrastive/c_g_neg': torch.mean(c_g * (1 - I)),
                        f'thread_{thread_idx}/contrastive/s_grad_norm': s_grad_norm,
                        f'thread_{thread_idx}/contrastive/p_grad_norm': p_grad_norm
                    }
                    metrics.update(thread_metrics)
            
            if i % (self.warmup_logging_interval if is_warm_up else self.learning_logging_interval) == 0:
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, i)
                end_time = time.time()
                print(f"Step {i} contrastive_loss {contrastive_loss:.3f} time {end_time - start_time:.3f}")
                print(f"Step {i} total_loss {total_loss/n_threads:.3f} time {end_time - start_time:.3f}")
                # print(f"Step {i} learning rate {cur_lr:.2e} grad norm upper {cur_grad_norm:.2e} | "
                #       f"s_grad_norm {s_grad_norm:.2e} | p_grad_norm {p_grad_norm:.2e}\n")
                print(f"Step {i} learning rate {self.learning_rate:.2e} grad norm upper {self.max_grad_norm:.2e} | "
                      f"s_grad_norm {s_grad_norm:.2e} | p_grad_norm {p_grad_norm:.2e}\n")
                start_time = end_time
                
    def plot_distance_map(self, start_pos, map_size, landmarks, obstacles, agent_id=None, step=None):
        """목표 지점으로부터의 거리를 시각화합니다.
        Args:
            start_pos: (tuple) 시작 위치 (x, y)
            landmarks: (list) 랜드마크 정보 리스트, 각 요소는 {'position': (x, y), 'size': size} 형태
            obstacles: (list) 장애물 정보 리스트, 각 요소는 {'position': (x, y), 'size': size} 형태
            map_size: (int) 맵의 크기
            agent_id: (str) 에이전트 식별자 (예: "thread_0_agent_1")
        """
        # 모든 가능한 위치 생성
        x = np.linspace(-map_size, map_size, 100)   # -map_size ~ map_size 사이의 100개의 점
        y = np.linspace(-map_size, map_size, 100)
        X, Y = np.meshgrid(x, y)    # 100x100 크기의 그리드 생성. x와 y의 모든 조합을 포함
        positions = np.stack([X.flatten(), Y.flatten()], axis=1)    # 100x2 크기의 행렬로 변환
        
        # 목표 위치를 텐서로 변환
        start_pos_tensor = torch.tensor(start_pos, device=self.device).float()  # (2,)
        
        # 배치 크기 설정 (더 작게 조정)
        batch_size = 100  # 한 번에 처리할 점의 수
        n_positions = len(positions)
        dists = np.zeros(n_positions)
        
        with torch.no_grad():
            # 배치 단위로 처리
            for i in range(0, n_positions, batch_size): # n_positions = 10000, batch_size = 100
                end_idx = min(i + batch_size, n_positions)
                batch_positions = positions[i:end_idx]  # 100개의 위치 (100, 2)
                
                # 현재 배치의 위치와 목표 위치를 인코딩
                positions_tensor = torch.from_numpy(batch_positions).to(self.device).float()
                start_pos_tensor_batch = start_pos_tensor.unsqueeze(0).repeat(len(batch_positions), 1)
                
                # 인코딩 수행
                phi_s = self.s_encoder(positions_tensor)    # (100, 32)
                phi_start = self.s_encoder(start_pos_tensor_batch)    # (100, 32)
                
                batch_dists = mrn_distance(phi_start[:, None], phi_s[None, :])  #(100, 10)
                dists[i:end_idx] = torch.diag(batch_dists).cpu().numpy().squeeze()
                
                # 메모리 해제
                del phi_s, phi_start, batch_dists
                torch.cuda.empty_cache()
            
            # 거리 맵 생성
            dist_map = dists.reshape(X.shape)
            
            # 시각화
            plt.figure(figsize=(10, 8))
            im = plt.imshow(dist_map, extent=[-map_size, map_size, -map_size, map_size], 
                          origin='lower', cmap='viridis')
            plt.colorbar(im, label='Distance from Start')
            
            # 목표 위치 표시
            plt.plot(start_pos[0], start_pos[1], 'r*', markersize=15, label='Start')
            
            # 랜드마크 표시
            for landmark in landmarks:
                plt.plot(landmark['position'][0], landmark['position'][1], 'g*', markersize=15, label='Landmark')
            
            # 장애물 표시
            for obstacle in obstacles:
                obstacle_circle = plt.Circle(
                    (obstacle['position'][0], obstacle['position'][1]),
                    obstacle['size'],
                    facecolor='red', alpha=0.3,
                    label='Wall'
                )
                plt.gca().add_patch(obstacle_circle)
            
            plt.title(f'Distance Map from Start (with Landmarks and Wall) - {agent_id}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            
            # 저장
            if step is not None:
                save_dir = os.path.join(self.run_dir, f'step_{step}')
                os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
                save_path = os.path.join(save_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{agent_id}.png')
            else:
                save_path = os.path.join(self.run_dir, f'distance_map_{time.strftime("%Y%m%d_%H%M%S")}_{agent_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{agent_id}의 거리 맵이 저장되었습니다: {save_path}")