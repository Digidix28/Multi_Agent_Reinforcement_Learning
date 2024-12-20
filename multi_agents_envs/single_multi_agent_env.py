from typing import Optional, Dict, Tuple
import numpy as np
import pygame
from pygame.locals import *
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray.rllib.utils.typing import AgentID

class NewMultiAgentGridWorldEnv(MultiAgentEnv):
    """
    Environnement Multi-Agent de Grille pour RLlib.

    Cet environnement implique deux agents qui naviguent dans un monde basé sur
    une grille. Leur objectif est de visiter toutes les cases et d'atteindre une 
    position cible. Les agents reçoivent des récompenses pour visiter des cases
    non visitées et des pénalités pour revisiter des cases déjà visitées. L'épisode
    se termine lorsque toutes les cases sont visitées et que les agents atteignent 
    la position cible, ou lorsque le nombre maximal d'étapes est dépassé.
    
    IMPORTANT : La principale différence avec un environement single-agent est que 
    tout est retourné dans un dictionnaire avec comme clé l'identifiant de l'agent.
    
    Exemple : Dans la fonction step, Si deux agents, "agent_1" et "agent_2", interagissent dans l'environnement, 
    les sorties de la méthode `step()` seront structurées comme suit :
    
        observations = {
            "agent_1": np.array([...]),  # Observation pour l'agent 1
            "agent_2": np.array([...])   # Observation pour l'agent 2
        }

        rewards = {
            "agent_1": 5.0,  # Récompense reçue par l'agent 1
            "agent_2": -1.0  # Récompense reçue par l'agent 2
        }

        terminated = {
            "agent_1": False,  # Indique si l'agent 1 a terminé
            "agent_2": True,   # Indique si l'agent 2 a terminé
            "__all__": True    # Indique si tous les agents ont terminé l'épisode
        }

        truncated = {
            "agent_1": False,  # Indique si l'agent 1 a été tronqué
            "agent_2": False,  # Indique si l'agent 2 a été tronqué
            "__all__": False   # Indique si l'épisode a été tronqué globalement
        }

        infos = {
            "agent_1": {},  # Informations supplémentaires pour l'agent 1
            "agent_2": {}   # Informations supplémentaires pour l'agent 2
        }

    Au lieu de retourner simplement un de chaque. Ainsi, chaque agent reçoit des observations, récompenses, et indicateurs de statut
    individuellement, permettant une gestion multi-agent indépendante.

    Attributs:
        - size (int): Taille de la grille (par défaut: 5x5).
        - max_steps (int): Nombre maximal d'étapes avant la fin de l'épisode.
        - possible_agents (list): Liste des agents possibles dans l'environnement.
        - agents (list): Liste des agents actuellement actifs dans l'épisode.
        - grid (np.ndarray): Matrice représentant l'état de la grille (0 = non visité, 1 = visité).
        - agent_positions (dict): Positions actuelles de chaque agent sur la grille.
        - target_position (np.ndarray): Position cible que les agents doivent atteindre.
        - current_step (int): Compteur du nombre d'étapes effectuées dans l'épisode.
        - render_mode (str): Mode de rendu ("human" ou "rgb_array").
        TRES IMPORTANT :
        - observation_spaces (dict): Espaces d'observation pour chaque agent. Doit correspondre au format de l'observation. 
        C'est le paramètre utilisé pour déterminer la taille de l'imput l'ayer des réseaux de neurones. 
        - action_spaces (dict): Espaces d'action pour chaque agent. Doit correspondre au format de l'action.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, env_config: Dict = {}):
        """
        Initialise l'environnement avec les paramètres donnés.

        Args:
            env_config (dict): Configuration de l'environnement. Peut contenir:
                - "size" (int): Taille de la grille (par défaut: 5).
                - "max_steps" (int): Nombre maximal d'étapes (par défaut: 800).
                - "render_mode" (str): Mode de rendu ("human" ou "rgb_array").
        """
        super().__init__()
        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 800)

        # Initialize agents
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()
        self.target_position = np.array([self.size - 1, self.size - 1], dtype=np.int32)

        # Initialize grid (0 = empty, 1 = visited)
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.grids = {agent_id: np.zeros((self.size, self.size), dtype=np.int32) for agent_id in self.agents}
        
        # Initialize agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size - 1], dtype=np.int32)}

        # Observation and action spaces
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=0,
                high=np.array(2 * [self.size - 1] + 17 * [1]),
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

    def _get_observation(self, agent_id) -> np.ndarray:
        """
        Génère une observation locale que l'agent va utiliser pour prendre une décision. 
        L'observation contient la position actuelle de l'agent, l'état de la case actuelle, 
        et l'état des voisins de l'agent. Celle-ci est ensuite aplatie pour être compatible
        avec les réseaux de neurones utilisés par les policy.

        Args:
            position (np.ndarray): Position actuelle de l'agent.

        Returns:
            np.ndarray: Observation flattened contenant la position, l'état de la case actuelle,
            et l'état des voisins.
        """
        try:
            position = self.agent_positions[agent_id]
        except: 
            print(self.agent_positions)
            print(agent_id)
        observation = np.concatenate([self.agent_positions[agent_id],
                                      self._get_one_hot_neighbors(agent_id)]).astype(np.float32)
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
                    # if tuple(self.agent_positions[self.other(agent_id)]) == (x,y):
                    #     neighbor_states.append(1) # 0 or 1 in MA
                    # else:
                    #     neighbor_states.append(0)
                    neighbor_states.append(0)
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    state = self.grid[nx, ny]
                    if state == 0:  # Unvisited
                        neighbor_states += [1, 0]
                    elif state == 1:  # Visited & Empty
                        # if tuple(self.agent_positions[self.other(agent_id)]) == (nx,ny):
                        #     neighbor_states += [1, 1] 
                        # else:
                        #     neighbor_states += [0, 1]
                        neighbor_states += [0, 1]
                else:
                    # Out of bounds: [0, 0]
                    neighbor_states += [0, 0]
        
        return np.array(neighbor_states).flatten()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        """
        Réinitialise l'environnement à son état initial. 
        Remet les agents à leurs positions et regénère la grille, 

        Returns:
            Tuple[Dict[AgentID, np.ndarray], Dict]: Observations initiales pour chaque agent
            et dictionnaire d'informations.
        """
        super().reset(seed=seed)

        # Reset grid
        self.grids["agent_1"].fill(0)
        self.grids["agent_2"].fill(0)

        # Reset agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size - 1], dtype=np.int32)}
        
        # Mark agent starting positions as visited
        self.grids["agent_1"][tuple(self.agent_positions["agent_1"])] = 1
        self.grids["agent_2"][tuple(self.agent_positions["agent_2"])] = 1
        self.grid = self.common_grid()

        # Reset agents to active
        self.agents = self.possible_agents.copy()
        self.current_step = 0

        # Generate observations
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.agents
        }

        # Render for human mode
        if self.render_mode == "human":
            self.render()
            self.close()

        return observations, {}

    def step(self, actions: Dict[AgentID, int]) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, dict]]:
        """
    Exécute une étape de simulation en appliquant les actions des agents.

    Cette méthode gère :
    - Le déplacement des agents en fonction de leurs actions.
    - L'attribution des récompenses en fonction des cellules visitées ou revisitées.
    - La vérification des conditions de terminaison de l'épisode.
    - La génération des nouvelles observations pour chaque agent.
    - La gestion des agents terminés et tronqués : Si un agent termine avant l'autre, 
        il est retiré de la liste des agents actifs pour éviter de le traiter à l'étape suivante.

    Args:
        actions (Dict[AgentID, int]): Dictionnaire contenant les actions de chaque agent actif.

    Returns:
        Tuple:
            - Dict[AgentID, np.ndarray]: Nouvelles observations pour chaque agent.
            - Dict[AgentID, float]: Récompenses attribuées à chaque agent.
            - Dict[AgentID, bool]: Indicateurs de terminaison pour chaque agent.
            - Dict[AgentID, bool]: Indicateurs de troncature pour chaque agent.
            - Dict[AgentID, dict]: Informations additionnelles (vide par défaut).
    """
        
        rewards = {agent_id: 0 for agent_id in self.agents}
        terminated = {}
        truncated = {}
        infos = {}

        # Apply actions
        for agent_id, action in actions.items():
            direction = self._action_to_direction[action]
            new_position = self.agent_positions[agent_id] + direction
            if not (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size):
                rewards[agent_id] = - 10
                new_position = np.clip(new_position, 0, self.size - 1)

            # Reward for visiting a new cell
            if self.grid[tuple(new_position)] == 0:  # Cell was unvisited
                rewards[agent_id] += 5
            else:  # Penalize for revisiting a cell
                rewards[agent_id] += -1

            self.agent_positions[agent_id] = new_position
            # Mark grid cell as visited
            self.grids[agent_id][tuple(new_position)] = 1

        self.grid = self.common_grid()
        # Increment step count
        self.current_step += 1

        if np.array_equal(self.agent_positions["agent_1"], self.agent_positions["agent_2"]):
            # the two agents hit each other!
            rewards["agent_1"] -= 20
            rewards["agent_2"] -= 20

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

    import numpy as np





