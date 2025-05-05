import gymnasium as gym
import numpy as np

class OldGymToGymnasium(gym.Wrapper):
    """Convert Gym-style (obs,r,done,info) envs to Gymnasium signature."""
    def reset(self, **kw):
        obs = self.env.reset(**kw)
        return (obs, {}) if not isinstance(obs, tuple) else obs
    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:                      # old: obs,r,done,info
            obs, r, done, info = out
            return obs, r, done, False, info   # truncated = False
        return out

class DictObsToArray(gym.ObservationWrapper):
    """Flatten nested-dict observations into a 1-D float32 array."""
    def __init__(self, env):
        super().__init__(env)
        vec = self._vec(env.reset()[0])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(len(vec),),
                                                dtype=np.float32)
    def observation(self, obs):
        return self._vec(obs)
    @staticmethod
    def _vec(obj):
        buf = []
        def walk(x):
            if isinstance(x, dict):
                for v in x.values(): walk(v)
            else:
                buf.extend(np.asarray(x, dtype=np.float32).ravel())
        walk(obj)
        return np.asarray(buf, dtype=np.float32)

class ScalarActionToDict(gym.ActionWrapper):
    """Turn scalar `a` → `{'tls_0': a}` so sumo-rl accepts it."""
    def __init__(self, env):
        super().__init__(env)
        self._tls_id = env.ts_ids[0]           # first (only) traffic light
    def action(self, act):
        return {self._tls_id: int(act)}
# add to compat.py
class DictRewardToFloat(gym.RewardWrapper):
    """
    Convert reward dict → float.
    Here we use the *sum* of all junction
    rewards; you can replace by np.mean, max, etc.
    """
    def reward(self, reward):
        if isinstance(reward, dict):
            return float(sum(reward.values()))
        return float(reward)

