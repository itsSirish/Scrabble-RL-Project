# scrabble_rl/model_cnnppo_better.py

import torch
import torch.nn as nn

class ActorCriticCNNBetter(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.board_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        self.board_output_dim = 64 * 15 * 15

        self.shared = nn.Sequential(
            nn.Linear(self.board_output_dim + 27 + 27 + 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        board = obs["board"].reshape(1, 2, 15, 15)
        board_feat = self.board_cnn(board)
        rack = obs["rack"]
        bag = obs["bag"]
        score = obs["score"]
        x = torch.cat([board_feat.squeeze(), rack, bag, score], dim=0)
        x = self.shared(x)
        return self.actor(x), self.critic(x)
