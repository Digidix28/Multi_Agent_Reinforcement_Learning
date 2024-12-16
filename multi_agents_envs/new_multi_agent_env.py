from typing import Optional, Dict, Tuple
import numpy as np
import pygame
from pygame.locals import *
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray.rllib.utils.typing import AgentID

class NewMultiAgentGridWorldEnv(MultiAgentEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, env_config: Dict = {}):
        super().__init__()
        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 800)

        # Initialize agents
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.target_position = np.array([self.size - 1, self.size - 1], dtype=np.int32)

        # Initialize grid (0 = empty, 1 = visited)
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        
        # Initialize agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size - 1], dtype=np.int32)}

        # Observation and action spaces
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=0,
                high=self.size - 1,
                shape=(19,),  # 2 (position) + 1 (current tile) + 16 (neighbor states)
                dtype=np.float32
            )
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

    def _get_observation(self, position: np.ndarray) -> np.ndarray:
        """Generate a flattened observation for a given position."""
        current_tile_state = self.grid[tuple(position)]
        neighbors_one_hot = self._get_one_hot_neighbors(position)
        observation = np.concatenate(([position[0], position[1], current_tile_state], neighbors_one_hot)).astype(np.float32)
        return observation
    
    def _get_one_hot_neighbors(self, position: np.ndarray) -> np.ndarray:
        """Get a one-hot encoded vector of neighbor states with two features per neighbor."""
        x, y = position
        neighbor_states = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # Skip the current position
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    # Inside bounds: Get state from the grid
                    state = self.grid[nx, ny]  # 0: unvisited, 1: visited
                    if state == 0:  # Unvisited
                        neighbor_states.append([1, 0])
                    elif state == 1:  # Visited
                        neighbor_states.append([0, 1])
                else:
                    # Out of bounds: [0, 0]
                    neighbor_states.append([0, 0])

        # Flatten the neighbor states
        return np.array(neighbor_states).flatten().astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        super().reset(seed=seed)

        # Reset grid
        self.grid.fill(0)

        # Reset agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size - 1], dtype=np.int32)}
        
        # Mark agent starting positions as visited
        self.grid[tuple(self.agent_positions["agent_1"])] = 1
        self.grid[tuple(self.agent_positions["agent_2"])] = 1

        # Reset agents to active
        self.agents = self.possible_agents.copy()
        self.current_step = 0

        # Generate observations
        observations = {
            agent_id: self._get_observation(self.agent_positions[agent_id])
            for agent_id in self.agents
        }

        # Render for human mode
        if self.render_mode == "human":
            self.render()

        return observations, {}

    def step(self, actions: Dict[AgentID, int]) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, dict]]:
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}

        # Apply actions
        for agent_id, action in actions.items():
            direction = self._action_to_direction[action]
            new_position = np.clip(self.agent_positions[agent_id] + direction, 0, self.size - 1)
            self.agent_positions[agent_id] = new_position

            # Reward for visiting a new cell
            if self.grid[tuple(new_position)] == 0:  # Cell was unvisited
                rewards[agent_id] = 5
            else:  # Penalize for revisiting a cell
                rewards[agent_id] = -1

            # Mark grid cell as visited
            self.grid[tuple(new_position)] = 1

        # Increment step count
        self.current_step += 1

        # Check if all tiles are visited
        all_tiles_visited = np.all(self.grid == 1)

        # Assign rewards and check termination
        for agent_id in self.agents:
            if all_tiles_visited and np.array_equal(self.agent_positions[agent_id], self.target_position):
                rewards[agent_id] += 200
                terminated[agent_id] = True
            elif self.current_step >= self.max_steps:
                rewards[agent_id] = -1000
                truncated[agent_id] = True
                terminated[agent_id] = False
            else:
                terminated[agent_id] = False
                truncated[agent_id] = False

        # Global termination flags
        terminated["__all__"] = all(terminated.values()) and all_tiles_visited
        truncated["__all__"] = self.current_step >= self.max_steps

        # Generate observations
        observations = {
            agent_id: self._get_observation(self.agent_positions[agent_id])
            for agent_id in self.agents
        }

        # Filter out agents that are terminated
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

        # Draw grid and cell states based on the grid object
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                # Set color based on grid state: 0 = empty, 1 = visited
                cell_color = (200, 200, 200) if self.grid[x, y] == 0 else (180, 255, 180)  # Gray for empty, green for visited
                pygame.draw.rect(surface, cell_color, rect)  # Fill cell
                pygame.draw.rect(surface, (50, 50, 50), rect, 1)  # Draw grid line

        # Draw target position
        target_rect = pygame.Rect(
            self.target_position[0] * self.cell_size,
            self.target_position[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(surface, (0, 255, 0), target_rect)  # Green for the target

        # Draw agents on the grid
        for agent_id, position in self.agent_positions.items():
            color = (0, 0, 255) if agent_id == "agent_1" else (255, 0, 0)  # Blue for agent_1, red for agent_2
            agent_rect = pygame.Rect(
                position[0] * self.cell_size,
                position[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.ellipse(surface, color, agent_rect)  # Draw agent as a circle

        # Render for human mode
        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        # Return frame for rgb_array mode
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(surface).swapaxes(0, 1)  # Convert back to NumPy array for consistency

        return None



    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    import numpy as np





