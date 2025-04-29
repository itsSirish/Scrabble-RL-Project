# scrabble_rl/evaluate_ppo_cnn.py

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scrabble_rl.model_cnnppo import ActorCriticCNN
from Game import Game
from utils import mapBoardToState

class RandomOpponent:
    def __init__(self, game):
        self.game = game

    def play_move(self):
        moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 50)
        if moves:
            move = random.choice(moves)
            self.game.play(move[2], move[0], move[3])
        else:
            self.game.numMoves = -1

class GreedyOpponent:
    def __init__(self, game):
        self.game = game

    def play_move(self):
        moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 50)
        if moves:
            move = max(moves, key=lambda m: m[1])
            self.game.play(move[2], move[0], move[3])
        else:
            self.game.numMoves = -1

def prepare_obs_tensor(obs):
    board = torch.tensor(np.array(obs["board"])).float()
    rack = torch.tensor(np.array(obs["rack"])).float().flatten()
    bag = torch.tensor(np.array(obs["bag"])).float().flatten()
    score = torch.tensor(np.array(obs["score"])).float().flatten()
    return {"board": board.unsqueeze(0), "rack": rack, "bag": bag, "score": score}

def print_board(board_obj):
    board = board_obj.board
    print("   " + " ".join(f"{i:2}" for i in range(15)))
    for i, row in enumerate(board):
        row_str = ""
        for tile in row:
            row_str += f" {tile.letter if tile.letter else '.'} "
        print(f"{i:2} {row_str}")

def play_game(agent1, opponent_class, selfplay=False, agent2=None):
    game = Game()
    agent1.eval()
    if agent2:
        agent2.eval()

    moves_played = []

    while game.numMoves >= 0:
        current_id = game.currentPlayer
        if current_id == 0:
            model = agent1
        elif selfplay:
            model = agent2
        else:
            opponent_class(game).play_move()
            continue

        obs = mapBoardToState(game, game.players[current_id].rack)
        obs_tensor = prepare_obs_tensor(obs)
        logits, _ = model(obs_tensor)
        action = torch.argmax(logits, dim=0).item()

        moves = game.find_best_moves(game.players[current_id].rack, 50)
        print(f"\n🤖 Player {current_id+1} move options: {len(moves)}")

        if moves:
            move = moves[min(action, len(moves) - 1)]
            print(f"➡️ Player {current_id+1} playing: {move[0]} | Score: {move[1]}")
            game.play(move[2], move[0], move[3])
            moves_played.append((current_id, move))
            print_board(game.board)
        else:
            print(f"⚠️ Player {current_id+1} has no valid moves.")
            game.numMoves = -1

    return game.players[0].score, game.players[1].score, moves_played

def evaluate(agent_path, opponent_class=None, games=3, label="", selfplay=False):
    model1 = ActorCriticCNN(action_dim=50)
    model1.load_state_dict(torch.load(agent_path))
    model2 = None
    if selfplay:
        model2 = ActorCriticCNN(action_dim=50)
        model2.load_state_dict(torch.load(agent_path))

    agent1_scores = []
    agent2_scores = []

    for i in range(games):
        print(f"\n🎮 Starting Game {i+1} vs {label if label else 'Agent2'}")
        s1, s2, moves = play_game(model1, opponent_class, selfplay, model2)
        agent1_scores.append(s1)
        agent2_scores.append(s2)
        print(f"🏁 Final Scores: Player 1 = {s1}, Player 2 = {s2}")

    win_rate = sum(s1 > s2 for s1, s2 in zip(agent1_scores, agent2_scores)) / games * 100
    score_diffs = [a - b for a, b in zip(agent1_scores, agent2_scores)]

    print("\n======= Evaluation Summary =======")
    print(f"✅ Win Rate: {win_rate:.2f}%")
    print(f"✅ Avg Score Agent1: {np.mean(agent1_scores):.2f}")
    print(f"✅ Avg Score Opponent: {np.mean(agent2_scores):.2f}")
    print("=================================")

    plt.figure()
    plt.bar(range(games), score_diffs, color="purple")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"Score Difference (Agent1 - {label or 'Agent2'})")
    plt.xlabel("Game")
    plt.ylabel("Score Difference")
    plt.grid()
    plt.savefig(f"scrabble_rl/Tests/simple_score_diff_vs_{label.lower().replace(' ', '_') if label else 'agent2'}.png")
    # plt.show()

    plt.figure()
    plt.pie([sum(s1 > s2 for s1, s2 in zip(agent1_scores, agent2_scores)),
             sum(s2 > s1 for s1, s2 in zip(agent1_scores, agent2_scores))],
            labels=["Agent1 Wins", "Opponent Wins"], autopct='%1.1f%%', colors=["green", "red"])
    plt.title(f"Win Rate vs {label or 'Agent2'}")
    plt.savefig(f"scrabble_rl/Tests/simple_winrate_vs_{label.lower().replace(' ', '_') if label else 'agent2'}.png")
    # plt.show()

if __name__ == "__main__":
    print("\nEvaluating CNN-PPO Agent vs RandomOpponent...")
    evaluate("scrabble_rl/Outputs/Best_agent1_selfplay.pt", opponent_class=RandomOpponent, games=10, label="RandomOpponent")

    print("\nEvaluating CNN-PPO Agent vs GreedyOpponent...")
    evaluate("scrabble_rl/Outputs/Best_agent1_selfplay.pt", opponent_class=GreedyOpponent, games=10, label="GreedyOpponent")

    print("\nEvaluating CNN-PPO Agent vs Agent2 (SelfPlay)...")
    evaluate("scrabble_rl/Outputs/Best_agent1_selfplay.pt", games=10, label="Agent2", selfplay=True)