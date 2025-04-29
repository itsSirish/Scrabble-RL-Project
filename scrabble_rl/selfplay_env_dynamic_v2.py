# scrabble_rl/selfplay_env_dynamic_better.py

import gym
from gym import spaces
import numpy as np
from Game import Game
from utils import mapBoardToState

class SelfPlayScrabbleEnv(gym.Env):
    def __init__(self, agent1, agent2, max_moves=300, relaxation_prob=0.3):
        super().__init__()
        self.game = Game()
        self.max_moves = max_moves
        self.current_step = 0
        self.agent1 = agent1
        self.agent2 = agent2
        self.current_agent = 0
        self.relaxation_prob = relaxation_prob
        self.moves = []

    def reset(self):
        self.game = Game()
        self.current_step = 0
        self.current_agent = 0
        self.moves = self.game.find_best_moves(self.game.players[self.current_agent].rack, num=50)
        return self._get_obs()

    def _get_obs(self):
        state = mapBoardToState(self.game, self.game.players[self.current_agent].rack)
        return {
            "board": np.array(state["board"]).transpose(2, 0, 1),
            "rack": np.array(state["rack"]),
            "bag": np.array(state["bag"]),
            "score": np.array(state["score"])
        }

    def step(self, action_idx):
        if len(self.moves) == 0:
            return self._get_obs(), 0, True, {}

        if np.random.rand() < self.relaxation_prob:
            action_idx = np.random.randint(len(self.moves))

        action_idx = min(action_idx, len(self.moves) - 1)
        move = self.moves[action_idx]

        pre_score = self.game.players[self.current_agent].score
        self.game.play(move[2], move[0], move[3])
        post_score = self.game.players[self.current_agent].score
        reward = post_score - pre_score

        self.current_step += 1
        done = self.game.numMoves == -1 or self.current_step >= self.max_moves

        self.current_agent = not self.current_agent
        self.moves = self.game.find_best_moves(self.game.players[self.current_agent].rack, num=50)

        return self._get_obs(), reward, done, {}

    def render(self):
        print(self.game.board)
