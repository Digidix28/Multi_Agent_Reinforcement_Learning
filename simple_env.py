import gymnasium as gym
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

class SimpleGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: EnvContext = {}):
        super(SimpleGridEnv, self).__init__()
        self.grid_size = env_config.get("grid_size", 5)
        self.start_pos = env_config.get("start_pos", (0, 0))
        self.goal_pos = env_config.get("goal_pos", (self.grid_size - 1, self.grid_size - 1))

        # Observation space en float32 pour éviter les conflits de type
        # L'observation est la position (2 valeurs) entre 0 et grid_size-1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=float(self.grid_size - 1), shape=(2,), dtype=np.float32
        )

        # Actions : 0 = haut, 1 = bas, 2 = gauche, 3 = droite
        self.action_space = gym.spaces.Discrete(4)

        self.agent_pos = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        else:
            self._np_random = None

        self.agent_pos = np.array(self.start_pos, dtype=np.int32)
        obs = self._get_obs().astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        if action == 0:  # haut
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # bas
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 2:  # gauche
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # droite
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = False

        reward = -1.0
        if terminated:
            reward = 10.0

        obs = self._get_obs().astype(np.float32)
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.int32)

