from typing import Optional, Dict, Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray.rllib.utils.typing import AgentID
import pygame
from pygame.locals import *

class MultiAgentGridWorldEnv(MultiAgentEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    
    def __init__(self, env_config: Dict = {}):
        super().__init__()
        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 800)
        self.share = env_config.get("share", False)
        # Initialize agents
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.target_position = np.array([self.size - 1, self.size -1], dtype = np.int32)

        # Initialize grid (0 = empty, 1 = visited)
        self.grids = {agent_id: np.zeros((self.size, self.size), dtype=np.int32) for agent_id in self.agents}

        # Initialize agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size-1], dtype=np.int32)}

        positions_shape = 4 if self.share else 2

        self.observation_spaces = {
            agent_id: spaces.Box(
                low = np.zeros((positions_shape + self.size ** 2)),
                high = np.array(positions_shape * [self.size - 1] + self.size ** 2 * [1]),
                shape = (positions_shape + self.size ** 2,),
                dtype = np.float32
            )
            for agent_id in self.agents
        }

        self.action_spaces = {
            agent_id: spaces.Discrete(4)
            for agent_id in self.agents
        }

        self.observation_spaces = spaces.Dict(self.observation_spaces)
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
            

    def _get_observation(self, agent_id):
        position = self.agent_positions[agent_id]
        grid = self.grids[agent_id].flatten()
        observation = np.concatenate([position, grid])
        if self.share:
            grid = self.common_grid().flatten()
            other_position = self.agent_positions[self.other(agent_id)]
            observation["other_position"] = other_position
            observation = np.concatenate([position, other_position, grid])
        return observation.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        super().reset(seed=seed)

        self.grids["agent_1"].fill(0)
        self.grids["agent_2"].fill(0)

        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size - 1], dtype=np.int32)}

        self.grids["agent_1"][tuple(self.agent_positions["agent_1"])] = 1
        self.grids["agent_2"][tuple(self.agent_positions["agent_2"])] = 1

        self.agents = self.possible_agents.copy()
        self.current_step = 0

        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        # Render for human mode
        if self.render_mode == "human":
            self.render()

        return observations, {}
    
    def step(self, actions: Dict[AgentID, int]) -> Tuple:
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}


        # 1. Appliquer les actions et mettre à jour les positions
        for agent_id, action in actions.items():
            common_grid = self.common_grid()
            direction = self._action_to_direction[action]
            new_position = np.clip(self.agent_positions[agent_id] + direction, 0, self.size - 1)
            self.agent_positions[agent_id] = new_position

            if common_grid[tuple(new_position)] == 0:
                rewards[agent_id] = 5
            else:
                rewards[agent_id] = -1
            
            self.grids[agent_id][tuple(new_position)] = 1
        
        self.current_step += 1 

        all_tiles_visited = np.all(self.common_grid() == 1)

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

        terminated["__all__"] = all(terminated.values()) and all_tiles_visited
        truncated["__all__"] = self.current_step >= self.max_steps

        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        # Filter out agents that are terminated
        self.agents = [agent_id for agent_id in self.agents if not terminated.get(agent_id, False)]
       
        # Render for human mode
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminated, truncated, infos

    

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