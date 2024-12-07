import functools
from copy import copy
import numpy as np
import random

from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv


class GridworldEnv(ParallelEnv):
    metadata = {
        "name": "gridworld_v0",
    }

    def __init__(self, grid_size=5, num_agents=2, max_steps=200):
        """
        Paramètres :
        - grid_size : dimension de la grille (grid_size x grid_size)
        - _num_agents : nombre d'agents
        - max_steps : limite de pas pour troncature
        """
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.max_steps = max_steps

        # Liste des agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]

        # Attributs définis dans reset()
        self.agents = []
        self.timestep = 0

        # Positions des agents
        self.agent_positions = None

        # Etat du goal
        self.goal_position = None

        # Grille de suivi des visites
        # True/False pour chaque case de la grille
        self.visited = None

    def reset(self, seed=None, options=None):
        # Optionnel: gérer le seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Initialisation des positions des agents
        # Par exemple, les agents commencent tous dans le coin supérieur gauche
        # ou dans des positions différentes
        self.agent_positions = []
        for _ in range(self.num_agents):
            # Placer chaque agent, par exemple, dans des coins distincts
            # Ici, simplifions: mettons tous les agents en haut à gauche
            self.agent_positions.append([0, 0])

        # Définir la position du goal, par exemple en bas à droite
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]

        # Réinitialiser la grille de visites
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for (x, y) in self.agent_positions:
            self.visited[y, x] = True

        # Observations initiales et infos
        observations = self._get_observations()
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        # actions est un dict {agent_name: action}, avec action dans {0: gauche, 1: droite, 2: haut, 3: bas}

        # Mettre à jour la position de chaque agent selon son action
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            x, y = self.agent_positions[i]

            # Up:2, Down:3, Left:0, Right:1 (par exemple)
            if action == 0 and x > 0:        # gauche
                x -= 1
            elif action == 1 and x < self.grid_size - 1:  # droite
                x += 1
            elif action == 2 and y > 0:      # haut
                y -= 1
            elif action == 3 and y < self.grid_size - 1:  # bas
                y += 1

            self.agent_positions[i] = [x, y]

        # Marquer les cases visitées
        for (x, y) in self.agent_positions:
            self.visited[y, x] = True

        # Calculer récompenses, terminations, truncations
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Condition de victoire : toutes les cases visitées ET un agent atteint le goal
        all_visited = np.all(self.visited)
        goal_reached = any((pos[0] == self.goal_position[0] and pos[1] == self.goal_position[1]) for pos in self.agent_positions)

        if all_visited and goal_reached:
            # Donner une grande récompense
            for a in self.agents:
                rewards[a] = 1.0
            terminations = {a: True for a in self.agents}
        else:
            # On peut donner une petite récompense à chaque nouvelle case visitée à ce tour
            # Pour simplifier, ignorons pour l'instant ce bonus
            pass

        # Vérifier la limite de pas (troncature)
        self.timestep += 1
        if self.timestep >= self.max_steps and not any(terminations.values()):
            truncations = {a: True for a in self.agents}

        # Si terminé ou truncation, plus d'agents actifs
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        observations = self._get_observations()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # Représentation textuelle : P pour agents, G pour goal, . pour visités, _ pour non visités
        grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
        # Mark visited
        grid[self.visited] = '.'

        # Mark goal
        gx, gy = self.goal_position
        grid[gy, gx] = 'G'

        # Mark agents
        for i, a in enumerate(self.agents):
            x, y = self.agent_positions[i]
            grid[y, x] = 'A'

        print(grid)

    def _get_observations(self):
        # Exemple d’observation : pour chaque agent, on donne :
        # - la position x,y de tous les agents
        # - la position du goal
        # - la matrice des cases visitées (sous forme flatten ?)
        # Attention, il faut être cohérent avec l’espace d’observation défini.
        #
        # Par simplicité, supposons :
        # Observation = (positions de tous les agents, position du goal, matrice visited aplatie)
        #
        # Nombre d’agents * 2 (x,y) + 2 (goal_x, goal_y) + grid_size*grid_size (visited)
        # On utilise un MultiDiscrete ou Box, selon le besoin.
        #
        # ICI : On va juste rendre un vecteur entier.
        # Par exemple :
        # obs = [agent_0_x, agent_0_y, agent_1_x, agent_1_y, goal_x, goal_y, visited(à plat: 0/1 pour chaque case)]
        #
        # ATTENTION : Pour PettingZoo, chaque agent reçoit la même taille d’obs.
        
        obs_dict = {}
        flat_visited = self.visited.flatten().astype(int)

        for i, agent in enumerate(self.agents):
            positions = []
            for (x, y) in self.agent_positions:
                positions += [x, y]
            gx, gy = self.goal_position
            obs = positions + [gx, gy] + flat_visited.tolist()
            obs_dict[agent] = np.array(obs, dtype=int)

        return obs_dict

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Définir l’espace d’observation
        # Nombre total d’éléments = num_agents*2 + 2 (goal) + grid_size*grid_size
        size = self.num_agents*2 + 2 + self.grid_size*self.grid_size
        # Chaque valeur : positions sont <= grid_size, visited est binaire (0 ou 1)
        # Pour simplifier, utiliser MultiDiscrete
        # Positions agent/goal : max = grid_size - 1
        # visited : max = 1
        #
        # On peut mettre une grande MultiDiscrete combinant pour chaque variable son max :
        # Agents x,y : max = grid_size
        # goal x,y : max = grid_size
        # visited[i] : max = 1
        #
        # Il faut construire une liste de max_value pour chaque élément
        max_values = []
        # Pour chaque agent: (x,y)
        for _ in range(self.num_agents):
            max_values.append(self.grid_size)  # x
            max_values.append(self.grid_size)  # y

        # goal (x,y)
        max_values.append(self.grid_size)
        max_values.append(self.grid_size)

        # visited
        for _ in range(self.grid_size*self.grid_size):
            max_values.append(2)  # 0 ou 1

        return MultiDiscrete(max_values)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 4 actions : gauche(0), droite(1), haut(2), bas(3)
        return Discrete(4)
