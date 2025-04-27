 
import gym
from gym import spaces
import numpy as np
from Game import Game
from utils import mapBoardToState

  # your existing state encoder

class ScrabbleEnv(gym.Env):
    def __init__(self, max_moves=100):
        super().__init__()
        self.game = Game()
        self.max_moves = max_moves
        self.current_step = 0
        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=26, shape=(2, 15, 15), dtype=np.int32),
            "rack": spaces.Box(low=0, high=10, shape=(27,), dtype=np.int32),
            "bag": spaces.Box(low=0, high=10, shape=(27,), dtype=np.int32),
            "score": spaces.Box(low=-500, high=500, shape=(2,), dtype=np.int32)
        })

    def reset(self):
        self.game = Game()
        self.current_step = 0
        self.moves = self.game.find_best_moves(self.game.players[0].rack, num=10)
        return self._get_obs()

    def _get_obs(self):
        state = mapBoardToState(self.game, self.game.players[self.game.currentPlayer].rack)
        return {
            "board": np.array(state["board"]).transpose(2, 0, 1),
            "rack": np.array(state["rack"]),
            "bag": np.array(state["bag"]),
            "score": np.array(state["score"])
        }

    def step(self, action_idx):

        if len(self.moves) == 0:
            # No valid moves = skip turn
            print("⚠️ No valid moves available. Skipping turn.")
            return self._get_obs(), 0, True, {}

        action_idx = min(action_idx, len(self.moves) - 1)  # ✅ clamp safely
        move = self.moves[action_idx]
        print(f"Move selected: {move[0]}  Score: {move[1]}")

        print(f"Available moves: {len(self.moves)}")

        pre_score = self.game.players[self.game.currentPlayer].score
        self.game.play(move[2], move[0], move[3])
        post_score = self.game.players[not self.game.currentPlayer].score

        reward = move[1]  # ✅ Use the move's score directly

        self.current_step += 1
        done = self.game.numMoves == -1 or self.current_step >= self.max_moves
        self.moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, num=10)
        return self._get_obs(), reward, done, {}

    def render(self):
        print(self.game.board)
