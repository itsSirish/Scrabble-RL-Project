# scrabble_rl/evaluate_ppo_cnn.py

import torch
import random
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

def play_game(agent_model, opponent_class):
    game = Game()
    agent_model.eval()

    agent_id = 0
    opponent_id = 1

    agent_moves = 0
    opponent_moves = 0

    while game.numMoves >= 0:
        if game.currentPlayer == agent_id:
            obs = mapBoardToState(game, game.players[0].rack)
            obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}
            logits, _ = agent_model(obs_tensor)
            action = torch.argmax(logits).item()

            moves = game.find_best_moves(game.players[0].rack, 50)
            print(f"🤖 Agent move options: {len(moves)}")

            if moves:
                move = moves[min(action, len(moves) - 1)]
                print(f"➡️ Agent playing: {move[0]} | Score: {move[1]}")
                game.play(move[2], move[0], move[3])
                agent_moves += 1
                print_board(game.board)
            else:
                print("⚠️ Agent has no valid moves.")
                game.numMoves = -1
        else:
            opponent_class(game).play_move()
            opponent_moves += 1

    agent_score = game.players[agent_id].score
    opponent_score = game.players[opponent_id].score
    print(f"🏁 Final scores → Agent: {agent_score}, Opponent: {opponent_score}")
    print(f"🧩 Tiles left in rack (Agent): {game.players[0].rack}")
    print(f"👜 Bag size: {len(game.bag)}")
    print(f"🎮 Agent moves played: {agent_moves}")
    print(f"🎮 Opponent moves played: {opponent_moves}")
    return agent_score, opponent_score


def evaluate(agent_path, games=5):
    model = ActorCriticCNN(action_dim=50)
    model.load_state_dict(torch.load(agent_path))
    model.eval()

    agent_wins = 0
    agent_scores = []
    opponent_scores = []

    for i in range(games):
        print(f"\n🎮 Starting Game {i+1}")
        a_score, o_score = play_game(model, RandomOpponent)
        agent_scores.append(a_score)
        opponent_scores.append(o_score)
        if a_score > o_score:
            agent_wins += 1

    print("\n================ Evaluation Summary ================")
    print(f"✅ Agent win rate: {agent_wins / games * 100:.2f}%")
    print(f"✅ Avg Agent score: {sum(agent_scores)/games:.2f}")
    print(f"✅ Avg Opponent score: {sum(opponent_scores)/games:.2f}")
    print("====================================================")

def print_board(board_obj):
    board = board_obj.board
    print("   " + " ".join(f"{i:2}" for i in range(15)))
    for i, row in enumerate(board):
        row_str = ""
        for tile in row:
            row_str += f" {tile.letter if tile.letter else '.'} "
        print(f"{i:2} {row_str}")

if __name__ == "__main__":
    print("Evaluating CNN-PPO Agent vs RandomOpponent...")
    evaluate("agent2_selfplay.pt", games=3)
