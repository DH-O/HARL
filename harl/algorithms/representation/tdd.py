"""TDD Intrinsic Reward Model for multi-agent environments"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, args, input_dim, device=torch.device("cuda")):
        self.args = args
        # GPU 사용 강제
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"GPU 사용: {torch.cuda.get_device_name(self.device)}")
        else:
            print("경고: GPU를 사용할 수 없습니다. CPU를 사용하지만 성능이 저하될 수 있습니다.")
            self.device = torch.device("cpu")
            
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        
        self.input_dim = input_dim
        self.latents_dim = self.args["network"]["latents_dim"]
        self.output_dim = self.args["network"]["output_dim"]
        
        self.potential_net = PotentialNet(self.input_dim, self.latents_dim).to(device)
        self.s_encoder = S_Encoder(self.input_dim, self.latents_dim, self.output_dim).to(device)
        self.g_encoder = S_Encoder(self.input_dim, self.latents_dim, self.output_dim).to(device)
        
        # 기본 TDD 설정
        self.total_steps = self.args["train"]["total_steps"]
        self.batch_size = self.args["train"]["batch_size"]
        self.learning_rate = self.args["train"]["learning_rate"]
        self.max_grad_norm = self.args["train"]["max_grad_norm"]
        self.temperature = self.args["train"]["temperature"]
        # learning_rate 처리
        learning_rate_raw = self.args["train"].get("learning_rate", 1e-4)
        if isinstance(learning_rate_raw, str):
            try:
                self.learning_rate = float(learning_rate_raw)
            except ValueError:
                self.learning_rate = 1e-4
                print(f"경고: learning_rate '{learning_rate_raw}'를 숫자로 변환할 수 없습니다. 기본값 1e-4를 사용합니다.")
        else:
            self.learning_rate = learning_rate_raw
        
        self.tdd_discount = self.args["tdd"]["tdd_discount"]
        
        self.optimizer = torch.optim.Adam([{"params": self.potential_net.parameters(), "lr": self.learning_rate},
                                          {"params": self.s_encoder.parameters(), "lr": self.learning_rate}])
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/TDD/{timestamp}"
        self.writer = SummaryWriter(run_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
        )
            
    def update(self, data): # data의 차원: (n_agent, episode 수, max_cycles, dict, n_rollout_threads, 2차원(s_t))
        obss = [[] for _ in range(len(data))]
        next_obss = [[] for _ in range(len(data))]
        
        # 단일 에이전트로 데이터 추출
        obss[0] = np.array([[step['obs'] for step in episode] for episode in data[0]])
        obss[0] = torch.from_numpy(obss[0]).to(self.device)
        next_obss[0] = np.array([[step['next_obs'] for step in episode] for episode in data[0]])
        next_obss[0] = torch.from_numpy(next_obss[0]).to(self.device)
        n_trajs, n_steps, n_threads = obss[0].shape[:3]
        
        print("TDD Training 시작")
        metrics = {}
        start_time = time.time()
        
        for i in range(self.total_steps):
            # 각 thread별로 독립적으로 처리
            for thread_idx in range(n_threads):
                # Sample mini-batch data (positive pairs)
                traj_idx = torch.randint(n_trajs, (self.batch_size,), device=self.device)
                step_idx = torch.randint(n_steps, (self.batch_size,), device=self.device)
                intervals = discounted_sampling(
                    n_steps - step_idx, self.tdd_discount
                )
                
                # 현재 상태와 다음 상태 추출 (특정 thread에 대해서만)
                obs = obss[0][traj_idx, step_idx, thread_idx]  # (batch_size, 2)
                goal = next_obss[0][traj_idx, step_idx + intervals, thread_idx]  # (batch_size, 2)
                
                c_g = self.potential_net(goal)
                phi_s = self.s_encoder(obs)
                phi_g = self.s_encoder(goal)
                
                logits = c_g.T - mrn_distance(phi_s[:, None], phi_g[None, :])
                # thread_logits.append(logits)
                I = torch.eye(self.batch_size, device=self.device)
                contrastive_loss = (F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2
                
                contrastive_loss = torch.mean(contrastive_loss)
                # logsumexp = torch.mean((torch.logsumexp(logits + 1e-6, axis=1)**2))
                # Backprop
                loss = contrastive_loss # 여기에 args.logsumexp_coef * logsumexp 추가를 할 수도 있다.
                
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            if i % self.args["train"]["logging_interval"] == 0:
                metrics['contrastive/contrastive_loss'] = contrastive_loss.item()
                metrics['contrastive/categorical_accuracy'] = torch.mean((torch.argmax(logits, axis=1) ==
                                                                          torch.arange(self.batch_size, device=self.device)).float()).item()
                metrics['contrastive/logits_pos'] = torch.diag(logits).mean()
                metrics['contrastive/logits_neg'] = torch.mean(logits * (1 - I))
                metrics['contrastive/logits_logsumexp'] = torch.mean((torch.logsumexp(logits, axis=1)**2))
                metrics['contrastive/c_g_pos'] = torch.diag(c_g).mean()
                metrics['contrastive/c_g_neg'] = torch.mean(c_g * (1 - I))
                
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, i)
                end_time = time.time()
                print(f"Step {i} contrastive_loss {contrastive_loss.item():.3f} time {end_time - start_time:.3f}")
                start_time = end_time
                
    def plot_distance_map(self, goal_pos, map_size, landmarks, obstacles):
        """목표 지점으로부터의 거리를 시각화합니다.
        Args:
            goal_pos: (tuple) 목표 위치 (x, y)
            landmarks: (list) 랜드마크 정보 리스트, 각 요소는 {'position': (x, y), 'size': size} 형태
            obstacles: (list) 장애물 정보 리스트, 각 요소는 {'position': (x, y), 'size': size} 형태
            map_size: (int) 맵의 크기
        """
        # 모든 가능한 위치 생성
        x = np.linspace(-map_size, map_size, 100)   # -map_size ~ map_size 사이의 100개의 점
        y = np.linspace(-map_size, map_size, 100)
        X, Y = np.meshgrid(x, y)    # 100x100 크기의 그리드 생성
        positions = np.stack([X.flatten(), Y.flatten()], axis=1)    # 100x2 크기의 행렬로 변환
        
        # 목표 위치를 텐서로 변환
        goal_pos_tensor = torch.tensor(goal_pos, device=self.device).float()
        
        # 배치 크기 설정 (더 작게 조정)
        batch_size = 100  # 한 번에 처리할 점의 수
        n_positions = len(positions)
        dists = np.zeros(n_positions)
        
        with torch.no_grad():
            # 배치 단위로 처리
            for i in range(0, n_positions, batch_size):
                end_idx = min(i + batch_size, n_positions)
                batch_positions = positions[i:end_idx]
                
                # 현재 배치의 위치와 목표 위치를 인코딩
                positions_tensor = torch.from_numpy(batch_positions).to(self.device).float()
                goal_pos_tensor_batch = goal_pos_tensor.unsqueeze(0).repeat(len(batch_positions), 1)
                
                # 인코딩 수행
                phi_s = self.s_encoder(positions_tensor)
                phi_g = self.s_encoder(goal_pos_tensor_batch)
                
                # MRN 거리 계산 (메모리 효율적으로)
                d = phi_s.shape[-1]
                x_prefix = phi_s[..., :d // 2]
                x_suffix = phi_s[..., d // 2:]
                y_prefix = phi_g[..., :d // 2]
                y_suffix = phi_g[..., d // 2:]
                
                max_component = torch.max(F.relu(x_prefix - y_prefix), axis=-1).values
                l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(axis=-1) + 1e-6)
                
                batch_dists = max_component + l2_component
                dists[i:end_idx] = batch_dists.cpu().numpy()
                
                # 메모리 해제
                del phi_s, phi_g, x_prefix, x_suffix, y_prefix, y_suffix, max_component, l2_component, batch_dists
                torch.cuda.empty_cache()
            
            # 거리 맵 생성
            dist_map = dists.reshape(X.shape)
            
            # 시각화
            plt.figure(figsize=(10, 8))
            im = plt.imshow(dist_map, extent=[-map_size, map_size, -map_size, map_size], 
                          origin='lower', cmap='viridis')
            plt.colorbar(im, label='Distance to Goal')
            
            # 목표 위치 표시
            plt.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
            
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
            
            plt.title('Distance Map to Goal (with Landmarks and Wall)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            
            # 저장
            save_path = f"runs/TDD/distance_map_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"거리 맵이 저장되었습니다: {save_path}")

# class TDDModel(nn.Module):
#     """TDD Intrinsic Reward Model for multi-agent environments."""

#     def __init__(self, args, obs_shape, device=torch.device("cpu")):
#         """Initialize TDD model.
#         Args:
#             args: arguments
#             obs_shape: observation shape
#             device: device to use
#         """
#         super(TDDModel, self).__init__()
#         self.args = args
        
#         # GPU 사용 강제
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda")
#             print(f"GPU 사용: {torch.cuda.get_device_name(self.device)}")
#         else:
#             print("경고: GPU를 사용할 수 없습니다. CPU를 사용하지만 성능이 저하될 수 있습니다.")
#             self.device = torch.device("cpu")
            
#         self.tpdv = dict(dtype=torch.float32, device=self.device)
        
#         # obs_shape 처리
#         if isinstance(obs_shape, (tuple, list)):
#             if len(obs_shape) == 2:  # (batch_size, obs_dim) 형태
#                 self.obs_shape = (obs_shape[1],)  # 실제 관측 차원만 사용
#             else:
#                 self.obs_shape = obs_shape
#         else:
#             self.obs_shape = (obs_shape,)  # 단일 숫자인 경우
            
        # self.feaures_dim = args["network"]["features_dim"]
        # self.latents_dim = args["network"]["latents_dim"]
        # self.output_dim = args["network"]["output_dim"]
        
#         # activation function 처리
#         activation_name = args["network"]["activation_function"]
#         self.activation = get_active_func(activation_name)
        
#         # TDD specific parameters
#         self.aggregate_fn = args.get("aggregate_fn", "mean")
#         self.energy_fn = args.get("energy_fn", "mrn_pot")
#         self.temperature = args["train"].get("temperature", 1.0)
#         self.knn_k = args.get("knn_k", 10)
        
#         # learning_rate 처리
#         learning_rate_raw = args["train"].get("learning_rate", 1e-4)
#         if isinstance(learning_rate_raw, str):
#             try:
#                 self.learning_rate = float(learning_rate_raw)
#             except ValueError:
#                 self.learning_rate = 1e-4
#                 print(f"경고: learning_rate '{learning_rate_raw}'를 숫자로 변환할 수 없습니다. 기본값 1e-4를 사용합니다.")
#         else:
#             self.learning_rate = learning_rate_raw
            
#         # self.batch_size = args["train"].get("batch_size", 256)
        
#         # CNN feature extractor
#         # if len(self.obs_shape) == 3:  # Image observations [C, H, W]
#         #     self.feature_extractor = self._build_cnn_extractor()
#         # else:  # Vector observations
#         #     self.feature_extractor = self._build_mlp_extractor()
        
#         # Encoder for representations
#         self.encoder = ModelOutputHeads(
#             feature_dim=self.model_features_dim,
#             latent_dim=self.model_latents_dim,
#             activation_fn=self.activation_fn,
#             mlp_norm=self.model_mlp_norm,
#             mlp_layers=self.model_mlp_layers,
#             output_dim=64,
#         )

#         # Potential network for energy calculation
#         self.potential_net = nn.Sequential(
#             nn.Linear(self.features_dim, self.latents_dim),
#             self.activation,
#             nn.Linear(self.latents_dim, 1)
#         )

#         # 옵티마이저
#         all_params = list(self.feature_extractor.parameters()) + \
#                     list(self.encoder.parameters()) + \
#                     list(self.potential_net.parameters())
#         self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate, weight_decay=1e-5)
#         self.max_grad_norm = args.get("max_grad_norm", 1.0)
        
#         # Buffer for training
#         # self.buffer_size = args["train"].get("buffer_size", 10000)
#         # self.transition_buffer = deque(maxlen=self.buffer_size)
        
#         # 모델을 지정된 디바이스로 이동
#         self.to(self.device)

#     def _build_cnn_extractor(self):
#         """Build CNN feature extractor."""
#         cnn = nn.Sequential(
#             nn.Conv2d(self.obs_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(self._get_conv_output_size(), self.features_dim),
#             nn.ReLU()
#         )
#         return cnn

#     def _get_conv_output_size(self):
#         """Get conv output size."""
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *self.obs_shape)
#             x = self._forward_conv(dummy_input)
#             return x.size(1)

#     def _forward_conv(self, obs):
#         """Forward pass through CNN layers without final linear layer."""
#         x = obs
#         x = nn.Conv2d(self.obs_shape[0], 32, kernel_size=8, stride=4)(x)
#         x = nn.ReLU()(x)
#         x = nn.Conv2d(32, 64, kernel_size=4, stride=2)(x)
#         x = nn.ReLU()(x)
#         x = nn.Conv2d(64, 64, kernel_size=3, stride=1)(x)
#         x = nn.ReLU()(x)
#         x = nn.Flatten()(x)
#         return x

#     def _build_mlp_extractor(self):
#         """Build MLP feature extractor."""
#         print(f"MLP Extractor 입력 차원: {self.obs_shape}")  # 디버깅용
#         input_dim = self.obs_shape[0] if isinstance(self.obs_shape[0], int) else self.obs_shape[0].item()
#         mlp = nn.Sequential(
#             nn.Linear(input_dim, self.features_dim),
#             self.activation,
#             nn.Linear(self.features_dim, self.features_dim),
#             self.activation
#         )
#         return mlp

#     def forward(self, obs):
#         """Extract features from obs.
#         Args:
#             obs: (torch.Tensor) observations
#         Returns:
#             features: (torch.Tensor) extracted features
#         """
#         return self.feature_extractor(obs)

#     def encode(self, features):
#         """Encode features to latent space.
#         Args:
#             features: (torch.Tensor) extracted features
#         Returns:
#             encodings: (torch.Tensor) encoded features
#         """
#         encodings = self.encoder(features)
#         return torch.clamp(encodings, min=-10.0, max=10.0)

#     def compute_mrn_distance(self, x, y):
#         """거리 계산 함수
#         Args:
#             x: (torch.Tensor) 인코딩된 벡터 1
#             y: (torch.Tensor) 인코딩된 벡터 2
#         Returns:
#             distance: (torch.Tensor) 거리 값
#         """
#         return mrn_distance(x, y, batch_mode=True)

#     def get_potential(self, features):
#         """Get potential value for features.
#         Args:
#             features: (torch.Tensor) extracted features
#         Returns:
#             potential: (torch.Tensor) potential value
#         """
#         return self.potential_net(features)

#     def calculate_intrinsic_reward(self, obs, next_obs, obs_history, rollout_mean_state=None):
#         """Calculate intrinsic reward.
#         Args:
#             obs: (torch.Tensor) current observation, shape (n_threads, obs_dim)
#             next_obs: (torch.Tensor) next observation, shape (n_threads, obs_dim)
#             obs_history: (list) list of observation history tensors for each environment
#             rollout_mean_state: (numpy.ndarray) mean state of all previous rollouts
#         Returns:
#             intrinsic_reward: (numpy.ndarray) intrinsic reward, shape (n_threads)
#             obs_history: (list) updated observation history
#             int_rews_part_1: (numpy.ndarray) part 1 of intrinsic reward, shape (n_threads)
#             int_rews_part_2: (numpy.ndarray) part 2 of intrinsic reward, shape (n_threads)
#         """
#         with torch.no_grad():
#             batch_size = obs.shape[0]
#             obs = obs.to(self.device)
#             next_obs = next_obs.to(self.device)
            
#             curr_features = self.forward(obs)
#             next_features = self.forward(next_obs)
            
#             curr_encoding = self.encode(curr_features)
#             next_encoding = self.encode(next_features)
            
#             intrinsic_reward = torch.zeros(obs.shape[0], device=self.device)
#             int_rews_part_1 = torch.zeros(obs.shape[0], device=self.device)
#             int_rews_part_2 = torch.zeros(obs.shape[0], device=self.device)
            
#             updated_history = []
#             for i in range(obs.shape[0]):
#                 if obs_history[i] is None:
#                     updated_history.append(torch.stack([curr_encoding[i], next_encoding[i]]))
#                 else:
#                     if isinstance(obs_history[i], torch.Tensor):
#                         hist = obs_history[i].to(self.device)
#                     else:
#                         hist = torch.tensor(obs_history[i], device=self.device)
#                     updated_history.append(torch.cat([hist, next_encoding[i].unsqueeze(0)], dim=0))
                
#                 if obs_history[i] is not None:
#                     hist_dist = self.compute_mrn_distance(updated_history[i], curr_encoding[i].unsqueeze(0))
#                     int_rews_part_1[i] = hist_dist.min()
                
#                 if rollout_mean_state is not None:
#                     mean_state = torch.tensor(rollout_mean_state, device=self.device)
#                     mean_encoding = self.encode(self.forward(mean_state))
#                     int_rews_part_2[i] = self.compute_mrn_distance(next_encoding[i].unsqueeze(0), mean_encoding)
                
#                 intrinsic_reward[i] = int_rews_part_1[i] + int_rews_part_2[i]
            
#             return (
#                 intrinsic_reward.cpu().numpy(),
#                 [h.cpu().numpy() if isinstance(h, torch.Tensor) else h for h in updated_history],
#                 int_rews_part_1.cpu().numpy(),
#                 int_rews_part_2.cpu().numpy()
#             )

#     # def store_transition(self, obs, next_obs):
#     #     """Store transition (obs, next_obs) in buffer for training.
#     #     Args:
#     #         obs: (torch.Tensor) current observation
#     #         next_obs: (torch.Tensor) next observation
#     #     """
#     #     obs_np = obs.detach().cpu().numpy()
#     #     next_obs_np = next_obs.detach().cpu().numpy()
        
#     #     batch_size = obs.size(0)
#     #     for i in range(min(batch_size, 4)):
#     #         self.transition_buffer.append((
#     #             obs_np[i],
#     #             next_obs_np[i]
#     #         ))

#     # def sample_batch(self, batch_size=None):
#     #     """Sample batch from transition buffer.
#     #     Args:
#     #         batch_size: (int) batch size
#     #     Returns:
#     #         obs_batch: (torch.Tensor) batch of observations
#     #         next_obs_batch: (torch.Tensor) batch of next observations
#     #     """
#     #     if batch_size is None:
#     #         batch_size = self.batch_size
            
#     #     if len(self.transition_buffer) < batch_size:
#     #         batch_size = len(self.transition_buffer)
            
#     #     indices = np.random.choice(len(self.transition_buffer), batch_size, replace=False)
#     #     obs_list, next_obs_list = [], []
        
#     #     for idx in indices:
#     #         obs, next_obs = self.transition_buffer[idx]
#     #         obs_list.append(obs)
#     #         next_obs_list.append(next_obs)
            
#     #     obs_batch = torch.tensor(np.array(obs_list), **self.tpdv)
#     #     next_obs_batch = torch.tensor(np.array(next_obs_list), **self.tpdv)
        
#     #     return obs_batch, next_obs_batch

#     def compute_infoNCE_loss(self, obs_batch, next_obs_batch):
#         """Compute InfoNCE loss for representation learning.
#         Args:
#             obs_batch: (torch.Tensor) batch of observations
#             next_obs_batch: (torch.Tensor) batch of next observations
#         Returns:
#             loss: (torch.Tensor) InfoNCE loss
#         """
#         batch_size = obs_batch.shape[0]
        
#         obs_features = self.forward(obs_batch)
#         next_features = self.forward(next_obs_batch)
        
#         obs_encodings = self.encode(obs_features)
#         next_encodings = self.encode(next_features)
        
#         obs_norm = torch.linalg.norm(obs_encodings, dim=1, keepdim=True)
#         next_norm = torch.linalg.norm(next_encodings, dim=1, keepdim=True)
        
#         obs_encodings_norm = obs_encodings / (obs_norm + 1e-8)
#         next_encodings_norm = next_encodings / (next_norm + 1e-8)
        
#         logits = torch.matmul(obs_encodings_norm, next_encodings_norm.t()) / self.temperature
#         labels = torch.arange(batch_size, device=self.device)
        
#         loss = F.cross_entropy(logits, labels)
        
#         v_curr = self.get_potential(obs_features)
#         v_next = self.get_potential(next_features)
        
#         potential_loss = F.smooth_l1_loss(v_curr, v_next - 0.05)
        
#         total_loss = loss + 0.1 * potential_loss
        
#         return total_loss, loss, potential_loss

#     def update(self, batch_size=None):
#         """Update model parameters using InfoNCE loss.
#         Args:
#             batch_size: (int) batch size
#         Returns:
#             loss_dict: (dict) dictionary of losses
#         """
#         if len(self.transition_buffer) < max(self.batch_size // 4, 16):
#             return {"total_loss": 0, "infoNCE_loss": 0, "potential_loss": 0}
        
#         actual_batch_size = min(self.batch_size, len(self.transition_buffer))
#         mini_batch_size = min(actual_batch_size, 64)
#         n_updates = max(1, actual_batch_size // mini_batch_size)
        
#         total_loss_sum = 0
#         infoNCE_loss_sum = 0
#         potential_loss_sum = 0
        
#         for _ in range(n_updates):
#             obs_batch, next_obs_batch = self.sample_batch(mini_batch_size)
            
#             total_loss, infoNCE_loss, potential_loss = self.compute_infoNCE_loss(obs_batch, next_obs_batch)
            
#             self.optimizer.zero_grad()
#             total_loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            
#             self.optimizer.step()
            
#             total_loss_sum += total_loss.item()
#             infoNCE_loss_sum += infoNCE_loss.item()
#             potential_loss_sum += potential_loss.item()
        
#         return {
#             "total_loss": total_loss_sum / n_updates,
#             "infoNCE_loss": infoNCE_loss_sum / n_updates,
#             "potential_loss": potential_loss_sum / n_updates
#         }

#     def save(self, save_dir, id):
#         """Save the model."""
#         state_dict = self.state_dict()
#         torch.save(state_dict, str(save_dir) + "/tdd_model_agent" + str(id) + ".pt")

#     def restore(self, model_dir, id):
#         """Restore the model."""
#         try:
#             state_dict = torch.load(str(model_dir) + "/tdd_model_agent" + str(id) + ".pt", map_location=self.device)
#             self.load_state_dict(state_dict)
#         except FileNotFoundError:
#             print(f"TDD 모델 파일을 찾을 수 없습니다: /tdd_model_agent{id}.pt")
#         except Exception as e:
#             print(f"TDD 모델 복원 중 오류 발생: {e}")