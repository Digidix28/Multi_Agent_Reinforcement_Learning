from typing import Optional, Dict, Tuple
import numpy as np
import pygame
from pygame.locals import *
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray.rllib.utils.typing import AgentID

class MultiAgentGridWorldEnv(MultiAgentEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, env_config: Dict = {}):
        super().__init__()
        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 60)

        # Initialize agents
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.target_position = np.array([self.size - 1, self.size - 1], dtype=np.int32)

        # Initialize agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size-1], dtype=np.int32)}

        # Observation and action spaces
        self.observation_spaces = {
            agent_id: spaces.Box(0, self.size - 1, shape=(2,), dtype=np.float32)
            for agent_id in self.agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(4)
            for agent_id in self.agents
        }

        # Global spaces for RLlib
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

        # Define movement directions
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        self.current_step = 0

        # Initialize rendering
        self.window_size = 500
        self.cell_size = self.window_size // self.size
        self.screen = None
        self.clock = None
        self.render_mode = env_config.get("render_mode", None)

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Multi-Agent Grid World")
            self.clock = pygame.time.Clock()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        super().reset(seed=seed)

        # Reset agent positions to their initial positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size-1], dtype=np.int32)}

        # Reset agents to active
        self.agents = self.possible_agents.copy()
        self.current_step = 0

        # Return observations
        observations = {
            agent_id: self.agent_positions[agent_id].astype(np.float32)
            for agent_id in self.agents
        }

        # Render for human mode
        if self.render_mode == "human":
            self.render()

        return observations, {}

    def step(self, actions: Dict[AgentID, int]) -> Tuple[
        Dict[AgentID, np.ndarray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, dict]
    ]:
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}

        # Apply actions
        for agent_id, action in actions.items():
            direction = self._action_to_direction[action]
            self.agent_positions[agent_id] = np.clip(
                self.agent_positions[agent_id] + direction, 0, self.size - 1
            )

        # Increment step
        self.current_step += 1

        # Assign rewards and check termination
        for agent_id in self.agents:
            if np.array_equal(self.agent_positions[agent_id], self.target_position):
                rewards[agent_id] = 50
                terminated[agent_id] = True
            elif self.current_step >= self.max_steps:
                rewards[agent_id] = -100
                truncated[agent_id] = True
                terminated[agent_id] = False
            else:
                rewards[agent_id] = -1
                terminated[agent_id] = False
                truncated[agent_id] = False

        # Global termination flags
        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = self.current_step >= self.max_steps

        # Generate observations
        observations = {
            agent_id: self.agent_positions[agent_id].astype(np.float32)
            for agent_id in self.agents
        }

        # Filter active agents
        self.agents = [agent_id for agent_id in self.agents if not terminated.get(agent_id, False)]

        # Render for human mode
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminated, truncated, infos

    def render(self):
        # Consistent return for different render modes
        if self.render_mode is None:
            return None

        # Render frame
        frame = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8) + 255  # White background
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Draw grid
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, (200, 200, 200), rect, 1)

        # Draw target position
        target_rect = pygame.Rect(
            self.target_position[0] * self.cell_size,
            self.target_position[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(surface, (0, 255, 0), target_rect)

        # Draw agents
        for agent_id, position in self.agent_positions.items():
            color = (0, 0, 255) if agent_id == "agent_1" else (255, 0, 0)
            agent_rect = pygame.Rect(
                position[0] * self.cell_size,
                position[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.ellipse(surface, color, agent_rect)

        # Render for human mode
        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        # Return frame for rgb_array mode
        if self.render_mode == "rgb_array":
            return frame

        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def get_agent_ids(self):
        return set(self.agents)
