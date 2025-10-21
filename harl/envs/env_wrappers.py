"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions, episode_step):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions, episode_step):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions, episode_step)    # reset때와 다르게 step_async()를 먼저 실행하고, step_wait()를 실행한다.
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()    # env_fn_wrapper.x()는 env_fn_wrapper의 x를 실행하는 것이다. x가 애초에 init_env()를 가리키고 있었으므로 env = init_env()와 같다.
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            if data[0].shape[0] != env.n_agents:
                raise ValueError(f"data[0].shape[0] must be equal to env.n_agents, but got {data[0].shape[0]} and {env.n_agents}")
            ob, s_ob, reward, done, info, available_actions = env.step(data[0])
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()

            # 100 스텝마다 랜드마크 위치와 개수 변경
            if data[1] is not None:  # episode_step이 전달된 경우
                episode_step = data[1]
                map_size = getattr(env.env.aec_env.env.env.env.env, 'map_size', 1.0)
                
                # 랜드마크 업데이트 호출
                if episode_step % 99 == 0 and episode_step > 0:
                    # 랜드마크 위치와 개수 변경
                    world = env.env.aec_env.env.env.env.env.world
                    
                    # 현재 랜드마크 개수 확인
                    current_landmarks = [entity for entity in world.landmarks if entity.name.startswith('landmark')]
                    
                    # wall 정보 가져오기
                    walls = [entity for entity in world.landmarks if entity.name.startswith('wall')]
                    
                    # 랜드마크 개수를 1-3개 사이에서 랜덤하게 변경
                    import random
                    new_num_landmarks = random.randint(1, 3)
                    
                    # 기존 랜드마크 제거
                    for landmark in current_landmarks:
                        world.landmarks.remove(landmark)
                    
                    # 새로운 랜드마크 생성 (wall과의 거리 고려)
                    from pettingzoo.mpe._mpe_utils.core import Landmark
                    min_distance_from_wall = 0.15  # wall과의 최소 거리 (조정 가능)
                    min_distance_between_landmarks = 0.4  # 랜드마크 간 최소 거리 (조정 가능)
                    max_attempts = 50  # 최대 시도 횟수 (랜드마크 위치 찾기 실패 시)
                    
                    if new_num_landmarks > env.n_agents:
                        raise ValueError(f"new_num_landmarks must be less than or equal to the number of current landmarks, but got {new_num_landmarks} and {env.n_agents}")
                    # if new_num_landmarks < env.n_agents:
                    #     print(f"new_num_landmarks is less than the number of agents, so some landmarks will be removed")
                    
                    for i in range(new_num_landmarks):
                        landmark = Landmark()
                        landmark.name = f'landmark_{i}'
                        landmark.collide = False
                        landmark.movable = False
                        landmark.size = 0.03  # landmark_size from config
                        landmark.weight = random.randint(1, 3)  # 1, 2, 3 중 하나의 값을 랜덤하게 선택
                        if landmark.weight == 1:
                            landmark.color = np.array([0.0, 0.0, 1.0])
                        elif landmark.weight == 2:
                            landmark.color = np.array([0.0, 1.0, 0.0])
                        elif landmark.weight == 3:
                            landmark.color = np.array([1.0, 0.0, 0.0])
                        else:
                            raise ValueError(f"landmark.weight must be 1, 2, or 3, but got {landmark.weight}")
                        
                        # wall과의 거리를 고려한 위치 설정
                        valid_position = False
                        attempts = 0
                        
                        while not valid_position and attempts < max_attempts:
                            # 랜덤 위치 생성 (map_size 범위 내)
                            candidate_pos = np.random.uniform(-map_size, map_size, 2)
                            
                            # wall과의 거리 확인
                            valid_position = True
                            for wall in walls:
                                wall_pos = wall.state.p_pos
                                wall_width = wall.width
                                wall_height = wall.height
                                
                                # wall의 중심에서 가장 가까운 점까지의 거리 계산
                                # wall이 직사각형이라고 가정하고 각 모서리까지의 거리 중 최소값 사용
                                
                                # wall의 경계까지의 거리 계산 (직사각형 wall 고려)
                                # wall의 경계 상자 내부에 있는지 확인
                                if (abs(candidate_pos[0] - wall_pos[0]) <= wall_width and 
                                    abs(candidate_pos[1] - wall_pos[1]) <= wall_height):
                                    # wall 내부에 있으면 무조건 거리 0
                                    valid_position = False
                                    break
                                
                                # wall 경계까지의 최단 거리 계산
                                dx = max(0, abs(candidate_pos[0] - wall_pos[0]) - wall_width)
                                dy = max(0, abs(candidate_pos[1] - wall_pos[1]) - wall_height)
                                distance_to_wall = np.sqrt(dx**2 + dy**2)
                                
                                if distance_to_wall < min_distance_from_wall:
                                    valid_position = False
                                    break
                            
                            # 기존 랜드마크들과의 거리 확인 (랜드마크 간 겹침 방지)
                            if valid_position:
                                for existing_landmark in world.landmarks:
                                    if existing_landmark.name.startswith('landmark'):
                                        distance_to_landmark = np.linalg.norm(candidate_pos - existing_landmark.state.p_pos)
                                        if distance_to_landmark < min_distance_between_landmarks:
                                            valid_position = False
                                            break
                            
                            if valid_position:
                                landmark.state.p_pos = candidate_pos
                                landmark.state.p_vel = np.zeros(world.dim_p)
                                world.landmarks.append(landmark)
                                break
                            
                            attempts += 1
                        
                        # 최대 시도 횟수 초과 시 기본 위치 사용
                        if not valid_position:
                            landmark.state.p_pos = np.random.uniform(-map_size * 0.5, map_size * 0.5, 2)
                            landmark.state.p_vel = np.zeros(world.dim_p)
                            world.landmarks.append(landmark)

            landmarks = []
            for entity in env.env.aec_env.env.env.env.env.world.landmarks:
                if entity.name.startswith('landmark'):
                    landmark_info = {
                        'position': entity.state.p_pos,
                        'size': entity.size
                    }
                    # weight 속성이 있으면 포함
                    if hasattr(entity, 'weight'):
                        landmark_info['weight'] = entity.weight
                    landmarks.append(landmark_info)
            
            """ weight 기반 추가 리워드 계산 """
            if len(landmarks) > 0 and len(ob) > 0:
                additional_reward = 0.0
                agent_positions = []
                
                # 설정 가능한 파라미터들
                landmark_sparse_reward = 5.0  # 랜드마크 sparse 리워드 (조정 가능)
                landmark_detection_radius_multiplier = 1.2  # 랜드마크 감지 반경 배수 (조정 가능)
                
                # 에이전트 위치 수집 (obs에서 에이전트 위치 추출)
                for agent_idx in range(len(ob)):
                    if len(ob[agent_idx]) >= 4:  # 최소 4차원 필요 (x, y, vx, vy)
                        agent_pos = ob[agent_idx][2:4]  # x, y 좌표
                        agent_positions.append(agent_pos)
                
                # 각 랜드마크에 대해 weight 기반 리워드 계산
                for landmark_idx, landmark in enumerate(landmarks):
                    if 'weight' in landmark:
                        landmark_pos = landmark['position']
                        landmark_weight = landmark['weight']
                        landmark_size = landmark['size']
                        
                        # 랜드마크 주변에 있는 에이전트 수 계산
                        agents_near_landmark = 0
                        # total_distance = 0.0
                        
                        for agent_pos in agent_positions:
                            distance = np.sqrt(np.sum((agent_pos - landmark_pos)**2))
                            detection_radius = landmark_size * landmark_detection_radius_multiplier
                            
                            if distance <= detection_radius:
                                agents_near_landmark += 1
                                # total_distance += distance
                        
                        # weight만큼의 에이전트가 모였을 때 추가 리워드
                        required_agents = int(landmark_weight)
                        if agents_near_landmark <= required_agents and agents_near_landmark > 0:
                            # 거리 기반 차등 리워드 (더 가까이 있을수록 더 많은 리워드)
                            # avg_distance = total_distance / agents_near_landmark if agents_near_landmark > 0 else 0
                            # distance_factor = max(0.5, 1.0 - (avg_distance / detection_radius))
                            
                            additional_reward += landmark_sparse_reward * landmark_weight
                            
                            # 디버깅을 위한 로그 (필요시 주석 해제)
                            # print(f"Landmark {landmark_idx}: weight={landmark_weight:.1f}, agents={agents_near_landmark}, reward={landmark_reward:.2f}")
                
                # 추가 리워드를 기존 리워드에 더함
                if additional_reward > 0:
                    for agent_idx in range(len(reward)):
                        reward[agent_idx] = [additional_reward + reward[agent_idx][0]]
            
            
            # 랜드마크와의 상대변위 처리 및 시야 범위 적용
            # if data[1] is not None:
            #     if len(landmarks) > 0:  # ob가 충분한 차원을 가지는지 확인
            #         # sum_min_dist_lm_wise = 0
            #         # for i in range(len(landmarks)):
            #         #     relative_pos_landmark_wise = []
            #         #     for agent_idx in range(len(ob)):
            #         #         relative_pos_landmark_wise.append(ob[agent_idx][4 + 2 * i:6 + 2 * i])
            #         #     sum_min_dist_lm_wise += np.min(np.sqrt(np.sum(np.array(relative_pos_landmark_wise)**2, axis=1)))
                        
            #         for agent_idx in range(len(ob)):
            #             # 5번째부터 10번째 차원까지 (인덱스 4부터 9까지) 처리
            #             for i in range(min(3, len(landmarks))):  # 최대 3개의 랜드마크 처리
            #                 start_idx = 4 + 2 * i  # 각 랜드마크의 시작 인덱스
            #                 end_idx = start_idx + 2  # 각 랜드마크의 끝 인덱스
                            
            #                 if end_idx <= len(ob[agent_idx]):  # 인덱스 범위 확인
            #                     # 현재 랜드마크와의 상대변위 (2차원)
            #                     relative_pos = ob[agent_idx][start_idx:end_idx]
                                
            #                     # 거리 계산 (유클리드 거리)
            #                     distance = np.sqrt(np.sum(relative_pos**2))
                                
            #                     # 거리가 현재 시야보다 크면 (최소 시야는 0.1), 0으로 마스킹하고 리워드는 충돌 리워드는 유지한채로 극단적 패널티
            #                     # if distance > max(0.1, (np.sqrt(2) * 2 * ((500 - data[1]) / env.max_cycles))):
            #                     #     ob[ob_idx][start_idx:end_idx] = 0.0
            #                     #     reward[ob_idx] += 0.5 * sum_min_dist_lm_wise * 3
            #                     #     reward[ob_idx] -= 0.5 * np.sqrt(2) * 2 * 3 * 3
                                
            #                     if distance > 0.1:
            #                         ob[agent_idx][start_idx:end_idx] = 0.0
            
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            
            # landmarks = []
            # for entity in env.env.aec_env.env.env.env.env.world.landmarks:
            #     if entity.name.startswith('landmark'):
            #         landmarks.append({
            #             'position': entity.state.p_pos,
            #             'size': entity.size
            #         })
            
            # 랜드마크와의 상대변위 처리 및 시야 범위 적용
            # if len(landmarks) > 0:  # ob가 충분한 차원을 가지는지 확인
            #     for ob_idx in range(len(ob)):
            #         # 5번째부터 10번째 차원까지 (인덱스 4부터 9까지) 처리
            #         for i in range(min(3, len(landmarks))):  # 최대 3개의 랜드마크 처리
            #             start_idx = 4 + 2 * i  # 각 랜드마크의 시작 인덱스
            #             end_idx = start_idx + 2  # 각 랜드마크의 끝 인덱스
                        
            #             if end_idx <= len(ob[ob_idx]):  # 인덱스 범위 확인
            #                 # 현재 랜드마크와의 상대변위 (2차원)
            #                 relative_pos = ob[ob_idx][start_idx:end_idx]
                            
            #                 # 거리 계산 (유클리드 거리)
            #                 distance = np.sqrt(np.sum(relative_pos**2))
                            
            #                 # 거리가 0.1보다 크면 해당 위치의 값들을 0으로 설정
            #                 if distance > 0.1:
            #                     ob[ob_idx][start_idx:end_idx] = 0.0
            
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
            else:
                fr = env.render()
                remote.send(fr)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "get_num_agents": 
            remote.send((env.n_agents))
        elif cmd == "get_landmarks_and_obstacles":
            landmarks = []
            obstacles = []
            for entity in env.env.aec_env.env.env.env.env.world.landmarks:
                if entity.name.startswith('wall'):
                    obstacles.append({
                        'position': entity.state.p_pos,
                        'size': entity.size
                    })
                else:
                    landmark_info = {
                        'position': entity.state.p_pos,
                        'size': entity.size
                    }
                    # weight 속성이 있으면 포함
                    if hasattr(entity, 'weight'):
                        landmark_info['weight'] = entity.weight
                    landmarks.append(landmark_info)
            remote.send((landmarks, obstacles))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False    # whether we are waiting for the result of step
        self.closed = False # whether the environment has been closed
        nenvs = len(env_fns)    
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])  # remotes: 메인 프로세스, work_remotes: 워커 프로세스, 즉 병렬로 돌아가는 프로세스
        # Pipe()는 양방향으로 데이터를 주고받을 수 있는 파이프를 생성한다.
        # nenvs만큼 Pipe()를 생성하는데 각각의 Pipe()는 (부모, 자식)으로 묶인다.
        # 이 때 *를 붙이면 리스트 내부의 튜플들을 해체하여 각각의 요소로 만들어준다.
        # 그걸 다시 zip()으로 묶어주면 각각의 같은 순서의 요소들끼리 묶인다.
        # 그래서 self.remotes는 모든 파이프들의 부모들로만 구성되고, self.work_remotes는 모든 파이프들의 자식들로만 구성된다.
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )   # 기껏 self.work_remotes와 self.remotes를 묶었는데 다시 풀어서 각각의 요소들을 묶어서 Process()에 넣어준다.
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for worker_remote in self.work_remotes:
            worker_remote.close()
        self.remotes[0].send(("get_num_agents", None))  # 메인프로세스인 self.remotes가 허공에 "get_num_agents"를 보내고, 이걸 받은 워커프로세스는 env.n_agents를 보내준다.
        self.n_agents = self.remotes[0].recv()  # 여기서 굳이 가장 첫번째 가상환경에서 받아도 되는 이유는 모든 가상환경이 동일한 n_agents를 가지기 때문이다.
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space    # 이거 대단한거 아니고 그냥 넘겨준 모든 정보들을 ShareVecEnv에 넘겨주는 것이다.
        )   # 심지어 ShareVecEnv는 이 파일 안에 있는 다른 클래스다.

    def step_async(self, actions, episode_step=None):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", (action, episode_step)))
        self.waiting = True # 환경 닫을때 이걸로 점검한다.

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)   
        # 리스트 내부에 2개의 튜플이 있다. 이것들을 풀어서 obs, share_obs, rews, dones, infos, available_actions로 만든다.
        # 그리고 zip을 통해서 다시 두 환경의 obs, share_obs, rews, dones, infos, available_actions로 묶어준다.
        return (
            np.stack(obs),  # 원래 길이가 2인 튜플로 구성된 리스트를 np.stack()을 통해 (2, 3, 18)인 튜플로 바꿔준다.
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
        except:
            pass

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results)
        )

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i]) # 이거는 0번째 에이전트의 obs_0, 1번째 에이전트의 obs_1이 저장 된다.
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i]) # 이거는 obs_0 + obs_1... 이 모든 에이전트에 저장된다.
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
