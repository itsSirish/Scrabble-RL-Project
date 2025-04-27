import numpy as np
import torch

class QAgent:
    def __init__(self, action_dim, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, alpha=0.1, gamma=0.99):
        self.q_table = {}  # Key: state tuple, Value: action-value array
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        # Simplify state (for now) to bag+score differential+rack
        bag = tuple(state['bag'])
        rack = tuple(state['rack'])
        score = tuple(state['score'])
        return (bag, rack, score)

    def select_action(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state, done):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_dim)

        best_next = np.max(self.q_table[next_key])

        target = reward + (0 if done else self.gamma * best_next)
        self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])

        # Decay exploration
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
