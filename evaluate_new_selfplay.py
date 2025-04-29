# scrabble_rl/evaluate_selfplay.py

import torch
import random
from scrabble_rl.model_cnnppo import ActorCriticCNN
from Game import Game
from utils import mapBoardToState
import os
import matplotlib.pyplot as plt

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
            move = max(moves, key=lambda m: m[1])  # highest scoring move
            self.game.play(move[2], move[0], move[3])
        else:
            self.game.numMoves = -1

def play_game(agent_model, opponent_class):
    game = Game()
    agent_model.eval()

    agent_id = 0
    opponent_id = 1

    while game.numMoves >= 0:
        if game.currentPlayer == agent_id:
            obs = mapBoardToState(game, game.players[agent_id].rack)
            obs_tensor = {k: torch.tensor(v).float().unsqueeze(0) for k, v in obs.items()}
            obs_tensor = {k: v if v.ndim > 1 else v.unsqueeze(0) for k, v in obs_tensor.items()}
            
            logits, _ = agent_model(obs_tensor)
            action = torch.argmax(logits, dim=1).item()

            moves = game.find_best_moves(game.players[agent_id].rack, 50)
            if moves:
                move = moves[min(action, len(moves)-1)]
                game.play(move[2], move[0], move[3])
            else:
                game.numMoves = -1
        else:
            opponent_class(game).play_move()

    agent_score = game.players[agent_id].score
    opponent_score = game.players[opponent_id].score
    return agent_score, opponent_score

def evaluate(agent_path, opponent_class, games=10, label=""):
    os.makedirs("test", exist_ok=True)

    model = ActorCriticCNN(action_dim=50)
    model.load_state_dict(torch.load(agent_path))
    model.eval()

    agent_wins = 0
    agent_scores = []
    opponent_scores = []
    score_diffs = []

    for i in range(games):
        a_score, o_score = play_game(model, opponent_class)
        agent_scores.append(a_score)
        opponent_scores.append(o_score)
        score_diffs.append(a_score - o_score)
        if a_score > o_score:
            agent_wins += 1
        print(f"Game {i+1}: Agent {a_score} - Opponent {o_score}")

    win_rate = agent_wins / games * 100

    print("\n=============================")
    print(f"✅ Agent win rate: {win_rate:.2f}%")
    print(f"✅ Avg Agent score: {sum(agent_scores)/games:.2f}")
    print(f"✅ Avg Opponent score: {sum(opponent_scores)/games:.2f}")
    print("=============================")

    # 🔍 Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Evaluation vs {label}")

    # Game-by-game scores
    axs[0, 0].plot(agent_scores, label='Agent Score', marker='o')
    axs[0, 0].plot(opponent_scores, label='Opponent Score', marker='x')
    axs[0, 0].set_title("Scores per Game")
    axs[0, 0].set_xlabel("Game")
    axs[0, 0].set_ylabel("Score")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Score difference
    axs[0, 1].bar(range(games), score_diffs, color='purple')
    axs[0, 1].set_title("Score Difference (Agent - Opponent)")
    axs[0, 1].set_xlabel("Game")
    axs[0, 1].set_ylabel("Score Diff")
    axs[0, 1].axhline(0, color='black', linestyle='--')
    axs[0, 1].grid()

    # Score histogram
    axs[1, 0].hist(agent_scores, bins=10, alpha=0.7, label='Agent')
    axs[1, 0].hist(opponent_scores, bins=10, alpha=0.7, label='Opponent')
    axs[1, 0].set_title("Score Histogram")
    axs[1, 0].set_xlabel("Score")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].legend()

    # Win rate pie chart
    axs[1, 1].pie([agent_wins, games - agent_wins], labels=["Agent Wins", "Opponent Wins"], autopct='%1.1f%%', colors=['green', 'red'])
    axs[1, 1].set_title("Win Rate")

    plt.tight_layout()
    plot_path = f"test/eval_vs_{label.lower()}.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Plot saved to {plot_path}")



if __name__ == "__main__":
    print("Evaluating agent vs RandomOpponent...")
    evaluate("Best_agent1_selfplay.pt", RandomOpponent, games=10, label="RandomOpponent")

    print("\nEvaluating agent vs GreedyOpponent...")
    evaluate("Best_agent2_selfplay.pt", GreedyOpponent, games=10, label="GreedyOpponent")
