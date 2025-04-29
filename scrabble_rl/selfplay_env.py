# scrabble_rl/selfplay_env.py

import gym
from gym import spaces
import numpy as np
from Game import Game
from utils import mapBoardToState

class SelfPlayScrabbleEnv(gym.Env):
    def __init__(self, agent1, agent2, max_moves=300):
        super().__init__()
        self.game = Game()
        self.max_moves = max_moves
        self.current_step = 0
        self.agent1 = agent1
        self.agent2 = agent2
        self.current_agent = 0  # 0: agent1, 1: agent2
        self.moves = []

        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=26, shape=(2, 15, 15), dtype=np.int32),
            "rack": spaces.Box(low=0, high=10, shape=(27,), dtype=np.int32),
            "bag": spaces.Box(low=0, high=10, shape=(27,), dtype=np.int32),
            "score": spaces.Box(low=-500, high=500, shape=(2,), dtype=np.int32)
        })

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

        action_idx = min(action_idx, len(self.moves) - 1)
        move = self.moves[action_idx]

        pre_score = self.game.players[self.current_agent].score
        self.game.play(move[2], move[0], move[3])
        post_score = self.game.players[not self.current_agent].score

        move_score = move[1]
        reward = move_score

        self.current_step += 1
        done = self.game.numMoves == -1 or self.current_step >= self.max_moves

        # Switch agent
        self.current_agent = not self.current_agent
        self.moves = self.game.find_best_moves(self.game.players[self.current_agent].rack, num=50)

        return self._get_obs(), reward, done, {}

    def render(self):
        print(self.game.board)
