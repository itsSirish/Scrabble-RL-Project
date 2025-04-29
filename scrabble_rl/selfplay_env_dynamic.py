# scrabble_rl/selfplay_env_dynamic.py

import gym
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
        self.current_agent = 0  # 0: agent1, 1: agent2
        self.relaxation_prob = relaxation_prob
        self.moves = []
        self.moves_played = []  # ✅ Track moves played
        self.skipped_turns = 0

    def reset(self):
        self.game = Game()
        self.current_step = 0
        self.current_agent = 0
        self.skipped_turns = 0
        self.moves_played = []
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
            print("⚠️ No valid moves. Ending turn.")
            self.current_step += 1
            self.skipped_turns += 1
            done = (self.skipped_turns >= 2) or (self.current_step >= self.max_moves)
            self.current_agent = not self.current_agent
            self.moves = self.game.find_best_moves(self.game.players[self.current_agent].rack, num=50)
            return self._get_obs(), -1, done, {}  # 🔥 penalty for skipping

        self.skipped_turns = 0

        # Relaxation
        if np.random.rand() < self.relaxation_prob:
            action_idx = np.random.randint(len(self.moves))
        else:
            action_idx = min(action_idx, len(self.moves) - 1)

        move = self.moves[action_idx]

        pre_score = self.game.players[self.current_agent].score
        self.game.play(move[2], move[0], move[3])
        post_score = self.game.players[self.current_agent].score

        move_score = move[1]
        reward = move_score

        # ✅ Save move played
        self.moves_played.append((self.current_agent, move))

        # ✅ PRINT after move played
        agent_str = "Agent1" if self.current_agent == 0 else "Agent2"
        print(f"\n🧩 {agent_str} played '{move[0]}' at {move[2]} direction {'HORIZ' if move[3]==0 else 'VERT'} | Move Score: {move[1]}")
        print("📋 Board after move:")
        self.render()
        print("-" * 50)

        self.current_step += 1
        done = (self.game.numMoves == -1) or (self.current_step >= self.max_moves)

        # Switch agent
        self.current_agent = not self.current_agent
        self.moves = self.game.find_best_moves(self.game.players[self.current_agent].rack, num=50)

        return self._get_obs(), reward, done, {}

    def render(self):
        print(self.game.board)
