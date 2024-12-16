from typing import Optional
import numpy as np
import gymnasium as gym
from ray.rllib.env.env_context import EnvContext


class GridWorldEnv(gym.Env):

    def __init__(self, env_config: EnvContext = {}):
        super(GridWorldEnv, self).__init__()
        # The size of the square grid
        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 60)

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are a 2D position encoded as floats in {0, ..., `size`-1}
        self.observation_space = gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }
        self.current_step = 0

    def _get_obs(self):
        # Return the agent's position as float32
        return self._agent_location.astype(np.float32)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0,0], dtype=np.int32)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )
        
        self._target_location = np.array([self.size-1, self.size-1], dtype=np.int32)
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        info = self._get_info()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps  # Troncature après max_steps
        if terminated:
            reward = 50
        elif truncated:
            reward = -100
        else:
            reward = -1
        observation = self._get_obs()
        
        

        return observation, reward, terminated, truncated, info
