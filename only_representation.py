import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

""" 하이퍼파라미터 설정 """
n_trajs = 3000
n_agents = 3
agent_size = 0.04
bounds = 2 * agent_size
map_size = 1.0
epi_len = 900
steps_per_seg = epi_len // 5
learn_steps = 5000
batch_ = 1024
tdd_dis = 0.99
lat_dim = 128
output_emb_dim = 64
lr = 1e-3
use_sep_net = True
use_sep_g_encod = False

metric_logging_interval = 500
base_dir = f"runs/SD_n_trajs_{n_trajs}_learn_steps_{learn_steps}_batch_size_{batch_}_lat_dim_{lat_dim}_output_emb_dim_{output_emb_dim}_lr_{lr}_use_sep_net_{use_sep_net}/{time.strftime('%m-%d_%H-%M-%S')}"

""" 텐서보드 설정 """
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=base_dir)

""" 디바이스 설정 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("경고: GPU를 사용할 수 없습니다. CPU를 사용하지만 성능이 저하될 수 있습니다.")
else:
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")

""" 네트워크 정의 """
if use_sep_net:
    potential_net = [nn.Sequential(
        nn.Linear(2, lat_dim),
        nn.LayerNorm(lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, lat_dim),
        nn.LayerNorm(lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, lat_dim),
        nn.LayerNorm(lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, 1)
    ).to(device) for _ in range(n_agents)]
else:
    potential_net = nn.Sequential(
        nn.Linear(2, lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, 1)   # 한 층 뺌
    ).to(device)

if use_sep_net:
    s_encoder = [nn.Sequential(
        nn.Linear(2, lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, output_emb_dim)
    ).to(device) for _ in range(n_agents)]
else:
    s_encoder = nn.Sequential(
        nn.Linear(2, lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, output_emb_dim)
    ).to(device)

if use_sep_g_encod:
    g_encoder = nn.Sequential(
        nn.Linear(2, lat_dim),
        nn.LayerNorm(lat_dim),
        nn.ReLU(),
        nn.Linear(lat_dim, output_emb_dim)
    ).to(device)
else:
    g_encoder = None

if g_encoder:
    params = [
        {"params": potential_net.parameters(), "lr": lr},
        {"params": s_encoder.parameters(), "lr": lr},
        {"params": g_encoder.parameters(), "lr": lr}
    ]
else:
    if use_sep_net:
        params = [
            {"params": potential_net[agent_id].parameters(), "lr": lr}
            for agent_id in range(n_agents)
        ]
        params += [
            {"params": s_encoder[agent_id].parameters(), "lr": lr}
            for agent_id in range(n_agents)
        ]
    else:
        params = [
            {"params": potential_net.parameters(), "lr": lr},
            {"params": s_encoder.parameters(), "lr": lr}
        ]

optimizer = torch.optim.Adam(params)

""" center를 기준으로 bounds 범위 내에서 랜덤하게 좌표를 생성하는 함수 """
def get_const_pos(pos_list, center, bounds):
    if not pos_list:
        return center
    return center + np.random.uniform(-bounds, bounds, 2)

""" 기하분포에 따라 랜덤하게 샘플링하는 함수 """
def discounted_sampling(ranges, discount):
    seeds = torch.rand(size=ranges.shape, device=ranges.device)
    
    if not 0 < discount < 1:
        raise ValueError("discount must be between 0 and 1")
    
    samples = torch.log(1 - (1 - discount**ranges) *seeds) / np.log(discount)
    samples = torch.min(torch.floor(samples).long(), ranges - 1)
    return samples

""" 궤적 시각화 함수 """
def plot_rollout_trajectory(rollout_data, n_trajs, n_agents, save_dir, map_size):
    save_dir = save_dir + "/exploration_metric"
    os.makedirs(save_dir, exist_ok=True)
    
    # 그래프 한 줄에 최대 3개만 생기도록
    max_graphs_per_row = 3
    n_cols = min(max_graphs_per_row, n_agents)
    n_rows = int(np.ceil(n_agents / n_cols))
    
    for traj_id in range(min(n_trajs, 5)):
        # 각 traj별로 새로운 figure 생성
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols, 
            figsize=(5 * n_cols, 4 * n_rows)
        )
        
        # axes가 1차원 배열인 경우 2차원 배열로 변환
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            # shape: (n_cols, ) -> (1, n_cols)
            axes = axes.reshape(1, n_cols)
        elif n_cols == 1:
            # shape: (n_rows, ) -> (n_rows, 1)
            axes = axes.reshape(n_rows, 1)
        
        for agent_id in range(n_agents):
            row = agent_id // n_cols
            col = agent_id % n_cols
            
            agent_traj = np.array(rollout_data[traj_id][agent_id])
            steps = agent_traj[:, 2]
            norm_steps = (steps - np.min(steps)) / (np.max(steps) - np.min(steps) + 1e-8)
            
            colormap = matplotlib.colormaps['viridis']
            colors = colormap(norm_steps)
            
            axes[row, col].scatter(agent_traj[:, 0], agent_traj[:, 1], c=colors, s=10, alpha=0.7)
            axes[row, col].set_title(f'traj {traj_id} - agent {agent_id}')
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('y')
            axes[row, col].set_xlim(-1.2*map_size, 1.2*map_size)
            axes[row, col].set_ylim(-1.2*map_size, 1.2*map_size)
            
            cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=axes[row, col])
            cbar.set_label("Step Index")
        
        # 사용하지 않는 subplot 제거
        for idx in range(n_agents, n_cols * n_rows):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        
        # 파일 이름 중복 방지: 이미 존재하는 경우 숫자 추가
        base_filename = f'rollout_trajectories_traj_{traj_id}'
        file_path = os.path.join(save_dir, f'{base_filename}.png')
        counter = 1
        while os.path.exists(file_path):
            filename = f"{base_filename}_{counter}.png"
            file_path = os.path.join(save_dir, filename)
            counter += 1

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

""" 거리 맵 시각화 함수 """
def plot_distance_map(start_pos, map_size, agent_id, suffix=None, step=None):
    # 시작점으로부터 거리를 잴 모든 목표지점들 설정
    x = np.linspace(-map_size, map_size, 100)   # -map_size ~ map_size 사이의 100개의 점
    y = np.linspace(-map_size, map_size, 100)
    X, Y = np.meshgrid(x, y)    # 100x100 크기의 그리드 생성. x와 y의 모든 조합을 포함
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)    # 100x2 크기의 행렬로 변환
    
    # 목표 위치를 텐서로 변환
    start_pos_tensor = torch.tensor(start_pos, device=device).float()  # (2,)
    
    # 거리 계산할 배치 크기 설정
    batch_size = 100  # 한 번에 처리할 점의 수
    n_positions = len(positions)
    dists = np.zeros(n_positions)
    
    with torch.no_grad():
        # 배치 단위로 처리
        for i in range(0, n_positions, batch_size):
            end_idx = min(i + batch_size, n_positions)
            batch_positions = positions[i:end_idx]  # 100개의 위치 (100, 2)
            
            # 현재 배치의 위치와 목표 위치를 인코딩
            positions_tensor = torch.from_numpy(batch_positions).to(device).float()
            start_pos_tensor_batch = start_pos_tensor.unsqueeze(0).repeat(len(batch_positions), 1)
            
            # 인코딩 수행
            if use_sep_net:
                phi_g = s_encoder[agent_id](positions_tensor)
                phi_start = s_encoder[agent_id](start_pos_tensor_batch)
            else:
                phi_g = g_encoder(positions_tensor) if g_encoder else s_encoder(positions_tensor)    # (100, 32)
                phi_start = s_encoder(start_pos_tensor_batch)    # (100, 32)
            
            batch_dists = mrn_distance(phi_start[:, None], phi_g[None, :])  #(100, 10)
            dists[i:end_idx] = torch.diag(batch_dists).cpu().numpy().squeeze()
            
            # 메모리 해제
            del phi_g, phi_start, batch_dists
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
    
    # 파일 저장
    save_dir = base_dir + "/distance_maps"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

""" Successor Distance 계산 함수 """
def mrn_distance(x, y):
    eps = 1e-6
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    
    max_component = torch.max(F.relu(x_prefix - y_prefix), axis=-1).values
    l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    
    distance = max_component + l2_component
    
    return distance

""" Successor Distance Update 함수 """
def update_SD_model(rollout_history):
    obss = [[] for _ in range(len(rollout_history))]
    next_obss = [[] for _ in range(len(rollout_history))]
    n_trajs = [[] for _ in range(len(rollout_history))]
    n_cumulative_steps = [[] for _ in range(len(rollout_history))]
    n_threads = [[] for _ in range(len(rollout_history))] # 해당 코드에서는 스레드 수가 1이다.
    
    metrics = {}
    
    """ 모든 에이전트에 대한 데이터 후처리 """
    for agent_id in range(len(rollout_history)):
        obss[agent_id] = np.array([[step['obs'] for step in episode] for episode in rollout_history[agent_id]], dtype=np.float32)
        obss[agent_id] = torch.from_numpy(obss[agent_id]).to(device)
        next_obss[agent_id] = np.array([[step['next_obs'] for step in episode] for episode in rollout_history[agent_id]], dtype=np.float32)
        next_obss[agent_id] = torch.from_numpy(next_obss[agent_id]).to(device)
        n_trajs[agent_id], n_cumulative_steps[agent_id], n_threads[agent_id] = obss[agent_id].shape[:3]
    
    # 설마 에이전트별 데이터 수가 다른 경우에 대한 예외처리
    if n_trajs[0] != n_trajs[1] or n_cumulative_steps[0] != n_cumulative_steps[1] or n_threads[0] != n_threads[1]\
        or n_trajs[0] != n_trajs[2] or n_cumulative_steps[0] != n_cumulative_steps[2] or n_threads[0] != n_threads[2]\
            or n_trajs[1] != n_trajs[2] or n_cumulative_steps[1] != n_cumulative_steps[2] or n_threads[1] != n_threads[2]:
        raise ValueError(f"n_trajs, n_steps, n_threads must be the same for all agents. \
                         n_trajs[0]: {n_trajs[0]}, n_trajs[1]: {n_trajs[1]}, n_trajs[2]: {n_trajs[2]}\
                         n_cumulative_steps[0]: {n_cumulative_steps[0]}, n_cumulative_steps[1]: {n_cumulative_steps[1]}, n_cumulative_steps[2]: {n_cumulative_steps[2]}\
                         n_threads[0]: {n_threads[0]}, n_threads[1]: {n_threads[1]}, n_threads[2]: {n_threads[2]}")
    
    for i in range(learn_steps):
        total_loss = 0.0
        
        for agent_id in range(n_agents):
            thread_metrics_list = []
            """ 각 스레드별로 traj_idx, step_idx를 랜덤하게 선택하고, 그 인덱스에 대한 obs와 goal을 추출한다. """
            for thread_idx in range(n_threads[0]):
                traj_idx = torch.randint(n_trajs[0], (batch_,), device=device)
                step_idx = torch.randint(n_cumulative_steps[0], (batch_,), device=device)
                intervals = discounted_sampling(
                    n_cumulative_steps[0] - step_idx, tdd_dis
                )
                obs = obss[agent_id][traj_idx, step_idx, thread_idx]  # (batch_size, 2)
                goal = next_obss[agent_id][traj_idx, step_idx + intervals, thread_idx]  # (batch_size, 2)
                
                """ Contrastive Learning에 사용될 인코더들 """
                if use_sep_net:
                    c_g = potential_net[agent_id](goal)
                    phi_s = s_encoder[agent_id](obs)
                    phi_g = s_encoder[agent_id](goal)
                else:
                    c_g = potential_net(goal)
                    phi_s = s_encoder(obs)
                    phi_g = g_encoder(goal) if g_encoder else s_encoder(goal)
                
                """ Successor Distance 계산 """
                mrn_dists = mrn_distance(phi_s[:, None], phi_g[None, :])    # 대각선 값이 양의 샘플이다. s_t 와 g_t사이의 거리를 재는 것이기 때문이다.
                # 위와 같이 함으로써 phi_g 각각에 대한 모든 phi_s와의 거리를 계산하며, 그게 모든 행에서 나타난다.
                
                """ logits 계산 """
                logits = c_g.T - mrn_dists
                I = torch.eye(batch_, device=device)
                
                """ Contrastive Loss 계산 """
                contrastive_loss = (F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2
                
                """ 전체 loss: contrastive """
                total_loss += contrastive_loss  # 모든 스레드와 모든 에이전트의 loss를 더한다.
                
                """ 그래디언트 계산 및 업데이트 """
                optimizer.zero_grad()
                contrastive_loss.backward()
                optimizer.step()
                
                # grad_norm 계산 및 출력
                def get_grad_norm(model):
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    return total_norm ** 0.5  # L2 norm의 제곱근 반환

                if use_sep_net:
                    s_grad_norm = get_grad_norm(s_encoder[agent_id])
                    p_grad_norm = get_grad_norm(potential_net[agent_id])
                    g_grad_norm = 0
                else:
                    s_grad_norm = get_grad_norm(s_encoder)
                    p_grad_norm = get_grad_norm(potential_net)
                    g_grad_norm = get_grad_norm(g_encoder) if g_encoder else 0
                
                # 각 스레드별로 메트릭 계산 / 추후 모든 스레드의 메트릭에서 agent_id 고려할거니 여기서는 그냥 저장만
                thread_metrics_list.append({
                    f"loss/contrastive_loss": contrastive_loss.item(),
                    f"gradients/s_encoder_norm": s_grad_norm,
                    f"gradients/potential_net_norm": p_grad_norm,
                    f"gradients/g_encoder_norm": g_grad_norm
                })
            
            if i % metric_logging_interval == 0:
                # 모든 스레드의 메트릭 평균 계산
                avg_thread_metrics = {}
                for key in thread_metrics_list[0].keys():
                    values = [metrics[key] for metrics in thread_metrics_list]
                    avg_thread_metrics[f'agent_{agent_id}/{key}'] = np.mean(values)
            
            metrics.update(avg_thread_metrics)
        
        if i % metric_logging_interval == 0:
            print(f"Step {i} - Average Loss: {total_loss / (n_agents * n_threads[0])}")
            # 전체 에이전트의 평균 메트릭 계산
            avg_metrics = {
                "avg/contrastive_loss": np.mean([metrics[f'agent_{j}/loss/contrastive_loss'] for j in range(n_agents)]),
                "avg/s_encoder_norm": np.mean([metrics[f'agent_{j}/gradients/s_encoder_norm'] for j in range(n_agents)]),
                "avg/potential_net_norm": np.mean([metrics[f'agent_{j}/gradients/potential_net_norm'] for j in range(n_agents)]),
                "avg/g_encoder_norm": np.mean([metrics[f'agent_{j}/gradients/g_encoder_norm'] for j in range(n_agents)]) if g_encoder else 0
            }
            metrics.update(avg_metrics)
            
            # 텐서보드에 메트릭 기록
            for k, v in metrics.items():
                writer.add_scalar(k, v, i)

# 궤적 표현을 위한 데이터 저장소
rollout_data = {traj_id: {agent_id: [] for agent_id in range(n_agents)} for traj_id in range(n_trajs)}
current_rollout_states = [[] for _ in range(n_agents)]
rollout_history = [[] for _ in range(n_agents)]

print("base_dir: ", base_dir)

""" 1. 초기 좌표 설정 """
respawn_coords = np.zeros((n_trajs, n_agents, 2))
for i in range(n_trajs):
    for j in range(n_agents):
        respawn_coords[i, j] = get_const_pos([respawn_coords[i, j]], -0.8 * map_size, bounds)

""" 2. 각 구간별 좌표 설정 """
start_time = time.time()
for i in range(n_trajs):
    for j in range(n_agents):
        # representation을 위한 데이터 temp 저장소
        traj = []
        
        # 첫 좌표 저장
        x, y = respawn_coords[i, j]
        step_idx = 0
        rollout_data[i][j].append([x, y, step_idx])
        traj.append([x, y])
        
        # 1) y좌표가 map_size - agent_size * 1.5까지 위로 직진 (x는 고정)
        start_y = y
        end_y1 = map_size - agent_size * 1.5
        for step in range(steps_per_seg - 1):
            ratio = step / steps_per_seg
            cur_y = start_y + (end_y1 - start_y) * ratio
            pos = get_const_pos([[x, cur_y]], [x, cur_y], bounds)
            rollout_data[i][j].append([pos[0], pos[1], step_idx])
            step_idx += 1
            traj.append([pos[0], pos[1]])
        y = end_y1
        if use_sep_net and j == 0:
            end_x_agent_0 = x
            end_y_agent_0 = y
        # 2) x좌표가 map_size / 2 - agent_size / 2 - map_size / 10까지 오른쪽 직진 (y는 고정)
        start_x = x
        end_x2 = map_size / 2 - agent_size / 2 - map_size / 10
        for step in range(steps_per_seg):
            ratio = step / steps_per_seg
            cur_x = start_x + (end_x2 - start_x) * ratio
            if use_sep_net and j == 0:
                pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], bounds)
            else:
                pos = get_const_pos([[cur_x, y]], [cur_x, y], bounds)
            rollout_data[i][j].append([pos[0], pos[1], step_idx])
            step_idx += 1
            traj.append([pos[0], pos[1]])
        x = end_x2
        
        # 3) y좌표가 -map_size + agent_size * 1.5까지 아래로 직진 (x는 고정)
        start_y = y
        end_y3 = -map_size + agent_size * 1.5
        for step in range(steps_per_seg):
            ratio = step / steps_per_seg
            cur_y = start_y + (end_y3 - start_y) * ratio
            if use_sep_net and j == 0:
                pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], bounds)
            else:
                pos = get_const_pos([[x, cur_y]], [x, cur_y], bounds)
            rollout_data[i][j].append([pos[0], pos[1], step_idx])
            step_idx += 1
            traj.append([pos[0], pos[1]])
        y = end_y3
        if use_sep_net and j == 1:
            end_x_agent_1 = x
            end_y_agent_1 = y
        
        # 4) x좌표가 map_size - agent_size * 1.5까지 오른쪽 직진 (y는 고정)
        start_x = x
        end_x4 = map_size - agent_size * 1.5
        for step in range(steps_per_seg):
            ratio = step / steps_per_seg
            cur_x = start_x + (end_x4 - start_x) * ratio
            if use_sep_net:
                if j == 0:
                    pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], bounds)
                elif j == 1:
                    pos = get_const_pos([[end_x_agent_1, end_y_agent_1]], [end_x_agent_1, end_y_agent_1], bounds)
                else:
                    pos = get_const_pos([[cur_x, y]], [cur_x, y], bounds)
            else:
                pos = get_const_pos([[cur_x, y]], [cur_x, y], bounds)
            rollout_data[i][j].append([pos[0], pos[1], step_idx])
            step_idx += 1
            traj.append([pos[0], pos[1]])
        x = end_x4
        
        # 5) y좌표가 map_size - agent_size * 1.5까지 위로 직진 (x는 고정)
        start_y = y
        end_y5 = map_size - agent_size * 1.5
        for step in range(steps_per_seg):
            ratio = step / steps_per_seg
            cur_y = start_y + (end_y5 - start_y) * ratio
            if use_sep_net:
                if j == 0:
                    pos = get_const_pos([[end_x_agent_0, end_y_agent_0]], [end_x_agent_0, end_y_agent_0], bounds)
                elif j == 1:
                    pos = get_const_pos([[end_x_agent_1, end_y_agent_1]], [end_x_agent_1, end_y_agent_1], bounds)
                else:
                    pos = get_const_pos([[x, cur_y]], [x, cur_y], bounds)
            else:
                pos = get_const_pos([[x, cur_y]], [x, cur_y], bounds)
            rollout_data[i][j].append([pos[0], pos[1], step_idx])
            step_idx += 1
            traj.append([pos[0], pos[1]])
        y = end_y5
        
        # traj 저장한거 이제 롤아웃 버퍼에 저장하자
        for t in range(len(traj) - 1):
            obs_dict = {
                "obs": np.array([traj[t]], dtype=np.float32),
                "next_obs": np.array([traj[t + 1]], dtype=np.float32),
                "dones": np.array([False])
            }
            current_rollout_states[j].append(obs_dict)
        obs_dict = {
            "obs": np.array([traj[-1]], dtype=np.float32),
            "next_obs": np.array([traj[0]], dtype=np.float32),
            "dones": np.array([True])
        }
        current_rollout_states[j].append(obs_dict)
        rollout_history[j].append(current_rollout_states[j])
        current_rollout_states[j] = []

""" 3. 궤적 시각화 """
plot_rollout_trajectory(
    rollout_data, 
    n_trajs, 
    n_agents, 
    save_dir=base_dir, 
    map_size=map_size
    )

""" 4. Temporal Distance Update """
print("Temporal Distance Update 시작 시각: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
update_SD_model(rollout_history)
print("Temporal Distance Update 완료 시각: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

""" 5. Eval """
print("Eval 시작 시각: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

eval_coords_ls = []
eval_respawn_coords = np.zeros((n_agents, 2))
for agent_id in range(n_agents):
    eval_respawn_coords[agent_id] = get_const_pos([eval_respawn_coords[agent_id]], -0.8 * map_size, bounds)
eval_coords_ls.append(eval_respawn_coords)

# 1) x좌표: eval_respawn_coords[j][0], y좌표가 map_size - agent_size * 1.5
eval_first_coods = np.zeros((n_agents, 2))
for agent_id in range(n_agents):
    eval_first_coods[agent_id] = get_const_pos([[eval_respawn_coords[agent_id][0], map_size - agent_size * 1.5]], [eval_respawn_coords[agent_id][0], map_size - agent_size * 1.5], bounds)
eval_coords_ls.append(eval_first_coods)
# 2) x좌표가 map_size / 2 - agent_size / 2 - map_size / 10까지 오른쪽 직진 (y는 고정)
eval_second_coods = np.zeros((n_agents, 2))
for agent_id in range(n_agents):
    eval_second_coods[agent_id] = get_const_pos([[map_size / 2 - agent_size / 2 - map_size / 10, eval_first_coods[agent_id][1]]], [map_size / 2 - agent_size / 2 - map_size / 10, eval_first_coods[agent_id][1]], bounds)
eval_coords_ls.append(eval_second_coods)
# 3) y좌표가 -map_size + agent_size * 1.5까지 아래로 직진 (x는 고정)
eval_third_coods = np.zeros((n_agents, 2))
for agent_id in range(n_agents):
    eval_third_coods[agent_id] = get_const_pos([[eval_second_coods[agent_id][0], -map_size + agent_size * 1.5]], [eval_second_coods[agent_id][0], -map_size + agent_size * 1.5], bounds)
eval_coords_ls.append(eval_third_coods)
# 4) x좌표가 map_size - agent_size * 1.5까지 오른쪽 직진 (y는 고정)
eval_fourth_coods = np.zeros((n_agents, 2))
for agent_id in range(n_agents):
    eval_fourth_coods[agent_id] = get_const_pos([[map_size - agent_size * 1.5, eval_third_coods[agent_id][1]]], [map_size - agent_size * 1.5, eval_third_coods[agent_id][1]], bounds)
eval_coords_ls.append(eval_fourth_coods)
# 5) y좌표가 map_size - agent_size * 1.5까지 위로 직진 (x는 고정)
eval_fifth_coods = np.zeros((n_agents, 2))
for agent_id in range(n_agents):
    eval_fifth_coods[agent_id] = get_const_pos([[eval_fourth_coods[agent_id][0], map_size - agent_size * 1.5]], [eval_fourth_coods[agent_id][0], map_size - agent_size * 1.5], bounds)
eval_coords_ls.append(eval_fifth_coods)

for j in range(5):
    for agent_id in range(n_agents):
        start_pos = eval_coords_ls[j][agent_id]
        start_pos_tensor = torch.tensor(start_pos, device=device).float()
        plot_distance_map(
            start_pos, 
            map_size, 
            agent_id, 
            f"agent_{agent_id}_j_{j}_start_pos_{start_pos[0]}_{start_pos[1]}",
            step=j
        )

print("Eval 완료 시각: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
print("base_dir: ", base_dir)