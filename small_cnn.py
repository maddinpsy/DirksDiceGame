
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))
