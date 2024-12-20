from typing import Optional, Dict, Tuple
import numpy as np
from gymnasium import Env, spaces
from ray.rllib.utils.typing import AgentID
import pygame
from pygame.locals import *

class CentralizedAgentGridWorldEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, env_config: Dict = {}):
        super(CentralizedAgentGridWorldEnv, self).__init__()

        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 60)

        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.target_position = np.array([self.size - 1, self.size -1], dtype = np.int32)

        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.grids = {agent_id: np.zeros((self.size, self.size), dtype=np.int32) for agent_id in self.agents}

        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size-1], dtype=np.int32)}


        self.observation_space = spaces.Box(
            low = 0,
            high = np.array(2 * 2 * [self.size - 1] + 17 * 2 * [1]),
            shape = (38,),
            dtype = np.float32
        )

        self.action_space = spaces.Discrete(16)

        # self.observation_space = spaces.Dict(self.observation_space)

        self._action_to_direction = {
            0: np.array([1, 0, 1, 0]), # right right
            1: np.array([1, 0, 0, 1]), # right down,
            2: np.array([1, 0, -1, 0]), # right left
            3: np.array([1, 0, 0, -1]), # right up
            4: np.array([0, 1, 1, 0]), # down right
            5: np.array([0, 1, 0, 1]), # down down,
            6: np.array([0, 1, -1, 0]), # down left
            7: np.array([0, 1, 0, -1]), # down up
            8: np.array([-1, 0, 1, 0]), # left right
            9: np.array([-1, 0, 0, 1]), # left down,
            10: np.array([-1, 0, -1, 0]), # left left
            11: np.array([-1, 0, 0, -1]), # left up
            12: np.array([0, -1, 1, 0]), # up right
            13: np.array([0, -1, 0, 1]), # up down,
            14: np.array([0, -1, -1, 0]), # up left
            15: np.array([0, -1, 0, -1]), # up up
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

    def other(self, agent_id):
        if agent_id == self.possible_agents[0]:
            return self.possible_agents[1]
        else:
            return self.possible_agents[0]

    def common_grid(self):
        common_grid = np.zeros((self.size, self.size), dtype=np.int32)
        for grid in self.grids.values():
            common_grid = common_grid | grid
        return common_grid
    
    def _get_observation(self):
        # observation = {
        #     "agent_1_position": self.agent_positions["agent_1"],
        #     "agent_2_position": self.agent_positions["agent_2"],
        #     "grid": self.common_grid().flatten()
        # }
        observation = np.concatenate([self.agent_positions["agent_1"], 
                                      self.agent_positions["agent_2"],
                                      self._get_one_hot_neighbors("agent_1"),
                                      self._get_one_hot_neighbors("agent_2")]).astype(np.float32)
        return observation
    
    def _get_one_hot_neighbors(self, agent_id) -> np.ndarray:
        """
        Génère un vecteur one-hot des états des voisins (deux caractéristiques par voisin).

        Args:
            position (np.ndarray): Position actuelle de l'agent.

        Returns:
            np.ndarray: Vecteur aplati des états des voisins.
        """
        self.grid = self.common_grid()
        x, y = self.agent_positions[agent_id]
        neighbor_states = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    if tuple(self.agent_positions[self.other(agent_id)]) == (x,y):
                        neighbor_states.append(1) # 0 or 1 in MA
                    else:
                        neighbor_states.append(0)
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    state = self.grid[nx, ny]
                    if state == 0:  # Unvisited
                        neighbor_states += [1, 0]
                    elif state == 1:  # Visited & Empty
                        if tuple(self.agent_positions[self.other(agent_id)]) == (nx,ny):
                            neighbor_states += [1, 1] 
                        else:
                            neighbor_states += [0, 1]
                else:
                    # Out of bounds: [0, 0]
                    neighbor_states += [0, 0]
        
        return np.array(neighbor_states).flatten()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        super().reset(seed=seed)

        self.grids["agent_1"].fill(0)
        self.grids["agent_2"].fill(0)

        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size - 1], dtype=np.int32)}

        self.grids["agent_1"][tuple(self.agent_positions["agent_1"])] = 1
        self.grids["agent_2"][tuple(self.agent_positions["agent_2"])] = 1
        self.grid = self.common_grid()

        self.agents = self.possible_agents.copy()
        self.current_step = 0

        observations = self._get_observation()

        if self.render_mode == "human":
            self.render()
            self.close()

        return observations, {}
    
    def step(self, action) -> Tuple:
        rewards = 0
        terminated = {}
        truncated = {}
        infos = {}

        directions = self._action_to_direction[action]


        for agent_id in self.agents:
            if agent_id == self.possible_agents[0]:
                new_position = self.agent_positions[agent_id] + directions[0:2]
            else:
                new_position = self.agent_positions[agent_id] + directions[2:4]
            if not (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size):
                rewards -= 10
                new_position = np.clip(new_position, 0, self.size - 1)

            if self.grid[tuple(new_position)] == 0:
                rewards += 5
            else:
                rewards += -1
            
            self.agent_positions[agent_id] = new_position
            self.grids[agent_id][tuple(new_position)] = 1
    
        self.grid = self.common_grid()
        
        self.current_step += 1

        if len(self.agents) > 1 and np.array_equal(self.agent_positions["agent_1"], self.agent_positions["agent_2"]):
            # the two agents hit each other!
            rewards -= 40

        all_tiles_visited = np.all(self.common_grid() == 1)

        for agent_id in self.agents:
            if all_tiles_visited and np.array_equal(self.agent_positions[agent_id], self.target_position):
                rewards += 200
                terminated[agent_id] = True
            elif self.current_step >= self.max_steps:
                rewards = -1000
                truncated[agent_id] = True
                terminated[agent_id] = False
            else:
                terminated[agent_id] = False
                truncated[agent_id] = False

        terminated["__all__"] = all(terminated.values()) and all_tiles_visited
        truncated["__all__"] = self.current_step >= self.max_steps

        observations = self._get_observation()

        self.agents = [agent_id for agent_id in self.agents if not terminated.get(agent_id, False)]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminated["__all__"], truncated["__all__"], infos

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
        common_grid = self.common_grid()
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                # Set color based on grid state: 0 = empty, 1 = visited
                if common_grid[x, y] == 0:
                    cell_color = (200, 200, 200)
                else:
                    if self.grids["agent_1"][x, y] == 1:
                        cell_color = (180, 180, 255) if self.grids["agent_2"][x,y] == 0 else (255, 180, 255)
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