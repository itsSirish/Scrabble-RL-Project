from scrabble_rl.model import ActorCritic
from scrabble_rl.q_agent import QAgent
from Game import Game
from utils import mapBoardToState
import torch
import numpy as np
import random


class TrainedQOpponent:
    def __init__(self, game, q_table):
        self.game = game
        self.q_table = q_table

    def get_state_key(self, state):
        bag = tuple(state['bag'])
        rack = tuple(state['rack'])
        score = tuple(state['score'])
        return (bag, rack, score)

    def play_move(self):
        state = mapBoardToState(self.game, self.game.players[self.game.currentPlayer].rack)
        key = self.get_state_key(state)
        if key not in self.q_table:
            moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 10)
            if moves:
                move = random.choice(moves)

                self.game.play(move[2], move[0], move[3])
            else:
                self.game.numMoves = -1
            return

        action = np.argmax(self.q_table[key])
        moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 10)
        if moves:
            move = moves[min(action, len(moves)-1)]
            self.game.play(move[2], move[0], move[3])
        else:
            self.game.numMoves = -1

def play_battle():
    game = Game()
    ppo_model = ActorCritic(action_dim=10)
    ppo_model.load_state_dict(torch.load("scrabble_ppo.pt"))
    ppo_model.eval()

    q_table = torch.load("q_agent.pt")
    q_agent = TrainedQOpponent(game, q_table)

    agent_id = 0
    opponent_id = 1

    while game.numMoves >= 0:
        if game.currentPlayer == agent_id:
            obs = mapBoardToState(game, game.players[agent_id].rack)
            obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}
            logits, _ = ppo_model(obs_tensor)
            action = torch.argmax(logits).item()
            moves = game.find_best_moves(game.players[agent_id].rack, 10)
            if moves:
                move = moves[min(action, len(moves)-1)]
                game.play(move[2], move[0], move[3])
            else:
                game.numMoves = -1
        else:
            q_agent.play_move()

    print(f"🏁 Final scores → PPO Agent: {game.players[agent_id].score}, QAgent: {game.players[opponent_id].score}")

if __name__ == "__main__":
    play_battle()
