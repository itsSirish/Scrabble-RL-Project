# scrabble_rl/deep_q_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from scrabble_rl.q_network import Scrabbler

class DeepQAgent:
    def __init__(
        self,
        action_dim=10,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.action_dim = action_dim
        self.device = device

        self.q_network = Scrabbler().to(device)
        self.target_network = Scrabbler().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, moves_available):
        if np.random.rand() < self.epsilon:
            return random.randint(0, moves_available-1)
        else:
            with torch.no_grad():
                q_values = []
                for idx in range(moves_available):
                    q = self.q_network(state)
                    q_values.append(q.item())
                return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        # Prepare batches
        state_batch = {k: torch.stack([s[k] for s in states]).to(self.device) for k in states[0]}
        next_state_batch = {k: torch.stack([s[k] for s in next_states]).to(self.device) for k in next_states[0]}
        actions = torch.tensor(actions).to(self.device).long()
        rewards = torch.tensor(rewards).to(self.device).float()
        dones = torch.tensor(dones).to(self.device).float()

        # Compute current Q-values
        current_q = self.q_network(state_batch).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_state_batch).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
