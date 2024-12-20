from typing import Optional, Dict, Tuple
import numpy as np
from gymnasium import Env, spaces
from ray.rllib.utils.typing import AgentID
import pygame
from pygame.locals import *
from random import randint

class SingleAgentGridWorldEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    def __init__(self, env_config: Dict = {}):
        super(SingleAgentGridWorldEnv, self).__init__()

        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 60)

        self.target_position = np.array([self.size - 1, self.size -1], dtype = np.int32)

        self.grid = np.zeros((self.size, self.size), dtype=np.int32)

        self.agent_positions = np.array([0, 0], dtype=np.int32)


        self.observation_space = spaces.Box(
            low = np.zeros((19,)),
            high = np.array(2 * [self.size - 1] + 17 * [1]),
            shape = (19,),
            dtype = np.float32
        )

        self.action_space = spaces.Discrete(4)

        # self.observation_space = spaces.Dict(self.observation_space)

        self._action_to_direction = {
            0: np.array([1, 0]), # right right
            1: np.array([0, 1]), # right down,
            2: np.array([-1, 0]), # right left
            3: np.array([0, -1]), # right up
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
    
    
    def _get_observation(self):
        # observation = {
        #     "agent_1_position": self.agent_positions["agent_1"],
        #     "agent_2_position": self.agent_positions["agent_2"],
        #     "grid": self.common_grid().flatten()
        # }
        observation = np.concatenate([self.agent_positions, 
                                      self._get_one_hot_neighbors()]).astype(np.float32)
        return observation
    
    def _get_one_hot_neighbors(self):
        x, y = self.agent_positions
        neighbor_states = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    neighbor_states.append(0) # 0 or 1 in MA
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    state = self.grid[nx, ny]
                    if state == 0:  # Unvisited
                        neighbor_states += [1, 0]
                    elif state == 1:  # Visited & Empty
                        neighbor_states += [0, 1]
                    else: # Visited and occupied
                        neighbor_states += [1, 1]
                else:
                    # Out of bounds: [0, 0]
                    neighbor_states += [0, 0]
        
        return np.array(neighbor_states).flatten()
        


    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        super().reset(seed=seed)

        self.grid.fill(0)

        if randint(0, 1) == 0:
            self.agent_positions = np.array([0, 0], dtype=np.int32)
        else:
            self.agent_positions = np.array([0, self.size - 1], dtype=np.int32)

        self.grid[tuple(self.agent_positions)] = 1

        self.current_step = 0

        observations = self._get_observation()

        # Render for human mode
        if self.render_mode == "human":
            self.render()
            self.close()

        return observations, {}
    
    def step(self, action) -> Tuple:
        reward = 0
        terminated = False
        truncated = False
        infos = {}

        directions = self._action_to_direction[action]

        new_position = self.agent_positions + directions
        if not (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size):
            reward = - 10
            new_position = np.clip(new_position, 0, self.size - 1)
        
        
        if self.grid[tuple(new_position)] == 0:
            reward += 5
            self.grid[tuple(new_position)] = 1
        else:
            reward += -1        
        
        self.agent_positions = new_position

        self.current_step += 1

        all_tiles_visited = np.all(self.grid == 1)

        terminated = all_tiles_visited and np.array_equal(self.agent_positions, self.target_position)
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward = 200
        elif truncated:
            reward = -1000

        observations = self._get_observation()

        return observations, reward, terminated, truncated, infos

    def render(self):
        """
    Affiche l'état actuel de l'environnement.

    Cette méthode gère le rendu graphique de la grille et des agents en utilisant
    Pygame. Elle prend en charge deux modes :
    - "human" : Affiche la grille dans une fenêtre interactive.
    - "rgb_array" : Retourne une représentation de la grille sous forme de tableau NumPy, 
    qui est utlisé pour enregistrer des vidéos de la run. 

    Returns:
        np.ndarray | None: Une image RGB de la grille si le mode est "rgb_array",
        sinon None pour le mode "human".
    """
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
                if self.grid[x, y] == 0:
                    cell_color = (200, 200, 200)
                else:
                    cell_color = (255, 180, 180)
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
        position = self.agent_positions
        agent_rect = pygame.Rect(
            position[0] * self.cell_size,
            position[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.ellipse(surface, (255, 0, 0), agent_rect)

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