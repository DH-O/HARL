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
    def step_async(self, actions):
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

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)    # reset때와 다르게 step_async()를 먼저 실행하고, step_wait()를 실행한다.
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
            ob, s_ob, reward, done, info, available_actions = env.step(data)
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

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
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

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
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
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
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
