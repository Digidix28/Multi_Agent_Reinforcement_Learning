from typing import Optional, Dict, Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray.rllib.utils.typing import AgentID


class MultiAgentGridWorldEnv(MultiAgentEnv):
    def __init__(self, env_config: Dict = {}):
        super().__init__()
        self.size = env_config.get("size", 5)
        self.max_steps = env_config.get("max_steps", 60)

        # Initialize agents
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents.copy()  # All agents start active
        self.target_position = np.array([self.size - 1, self.size - 1], dtype=np.int32)

        # Initialize agent positions
        self.agent_positions = {"agent_1": np.array([0, 0], dtype=np.int32), "agent_2": np.array([0, self.size-1], dtype=np.int32)}

        # Observation and action spaces (same for all agents)
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
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[AgentID, np.ndarray], Dict]:
        super().reset(seed=seed, options=options)

        # Reset agent positions to start position
        for agent_id in self.possible_agents:
            self.agent_positions[agent_id] = np.array([0, 0], dtype=np.int32)

        # Reset agents to active
        self.agents = self.possible_agents.copy()
        self.current_step = 0

        # Return observations for all agents
        observations = {
            agent_id: self.agent_positions[agent_id].astype(np.float32)
            for agent_id in self.agents
        }
        return observations, {}

    def step(self, actions: Dict[AgentID, int]) -> Tuple[
        Dict[AgentID, np.ndarray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], Dict[AgentID, dict]
    ]:
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}

        # 1. Appliquer les actions et mettre à jour les positions
        for agent_id, action in actions.items():
            direction = self._action_to_direction[action]
            self.agent_positions[agent_id] = np.clip(
                self.agent_positions[agent_id] + direction, 0, self.size - 1
            )

        # 2. Incrémenter l'étape actuelle
        self.current_step += 1

        # 3. Attribuer les récompenses et vérifier les conditions de terminaison
        for agent_id in self.agents:
            if np.array_equal(self.agent_positions[agent_id], self.target_position):
                rewards[agent_id] = 50  # Récompense pour avoir atteint la cible
                terminated[agent_id] = True
            elif self.current_step >= self.max_steps:
                rewards[agent_id] = -100  # Pénalité pour dépassement des étapes max
                truncated[agent_id] = True
                terminated[agent_id] = False
            else:
                rewards[agent_id] = -1  # Pénalité pour chaque étape
                terminated[agent_id] = False
                truncated[agent_id] = False

        # 4. Terminaison globale
        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = self.current_step >= self.max_steps

        # 5. Générer les observations avant de filtrer les agents
        observations = {
            agent_id: self.agent_positions[agent_id].astype(np.float32)
            for agent_id in self.agents
        }

        # 6. Filtrer les agents actifs
        self.agents = [agent_id for agent_id in self.agents if not terminated.get(agent_id, False)]

        return observations, rewards, terminated, truncated, infos



    def get_observation_space(self, agent_id: AgentID) -> spaces.Space:
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id: AgentID) -> spaces.Space:
        return self.action_spaces[agent_id]
