# scrabble_rl/q_network.py
import torch
import torch.nn as nn

class Scrabbler(nn.Module):
    def __init__(self, input_dim=252, hidden_dims=[256, 128, 64]):
        super(Scrabbler, self).__init__()

        self.board_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.rack_dn = nn.Sequential(
            nn.Linear(27, 32),
            nn.ReLU()
        )

        self.bag_dn = nn.Sequential(
            nn.Linear(86, 32),
            nn.ReLU()
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(32*32 + 32 + 32 + 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)  # Output: Q-value
        )

    def forward(self, state):
        board = state['board'].squeeze(1)    # 🛠 fix
        board = self.board_cnn(board)    # [batch_size, 2, 15, 15] → flatten
        rack = self.rack_dn(state['rack'])         # [batch_size, 27]
        bag = self.bag_dn(state['bag'])            # [batch_size, 86]
        score = state['score']                     # [batch_size, 2]

        x = torch.cat((board, rack, bag, score), dim=1)
        q_value = self.final_mlp(x)
        return q_value
