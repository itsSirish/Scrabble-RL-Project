 
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2*15*15 + 27 + 27 + 2, 256),
            nn.ReLU(),
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
        board = obs["board"].reshape(-1)
        rack = obs["rack"]
        bag = obs["bag"]
        score = obs["score"]
        x = torch.cat([board, rack, bag, score], dim=0)
        x = self.shared(x)
        return self.actor(x), self.critic(x)
