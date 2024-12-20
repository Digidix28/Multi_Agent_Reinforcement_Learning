from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.typing import SampleBatchType, TensorType
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium
import numpy as np
import torch
import torch.nn as nn


class CentralizedCriticModel(FullyConnectedNetwork):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(input_dim = input_dim)
    
    def forward_central_value(self, joint_obs):
        # compute he centralized value using joint observations
        return self.value_branch(joint_obs)

class MAPPOTorchPolicy(TorchPolicyV2):
    def __init__(self, observation_space, action_space, config):
        self.model = FullyConnectedNetwork(
            input_dim = observation_space.shape[0],
            action_space = action_space,
            model_config_dict = config["model"],
        )
        super().__init__(observation_space, action_space, config, model = self.model)

        self.central_critic = CentralizedCriticModel(
            input_dim=config["joint_obs_dim"],
            action_space = action_space,
            model_config_dict = config["model"]
        )

    def compute_central_value(self, joint_obs):
        joint_obs_tensor = torch.tensor(joint_obs, dtype=torch.float32)
        return self.central_critic.forward_central_value(joint_obs_tensor)
    
    def loss(self, model, dist_class, train_batch):
        policy_loss = super().loss(model, dist_class, train_batch)

        joint_obs = train_batch["joint_obs"]
        values = self.compute_central_value(joint_obs)
        value_loss = torch.mean((values - train_batch[SampleBatch.REWARDS]) ** 2)

        return policy_loss + self.coonfig["vf_coeff"] * value_loss
    