# evaluate.py
import torch
import random
from scrabble_rl.model import ActorCritic
from Game import Game
from utils import mapBoardToState

class RandomOpponent:
    def __init__(self, game):
        self.game = game

    def play_move(self):
        moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 10)
        if moves:
            move = random.choice(moves)
            self.game.play(move[2], move[0], move[3])
        else:
            self.game.numMoves = -1

class GreedyOpponent:
    def __init__(self, game):
        self.game = game

    def play_move(self):
        moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 10)
        print(f"🎲 Opponent move options: {len(moves)}")
        if moves:
            move = random.choice(moves)
            print(f"🎯 Opponent plays: {move[0]} | Score: {move[1]}")
            self.game.play(move[2], move[0], move[3])
        else:
            print("⚠️ Opponent has no valid moves.")
            self.game.numMoves = -1

def play_game(agent_model, opponent_class):

    game = Game()
    agent_model.eval()

    agent_id = 0
    opponent_id = 1

    while game.numMoves >= 0:
        if game.currentPlayer == agent_id:
            obs = mapBoardToState(game, game.players[0].rack)
            obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}
            logits, _ = agent_model(obs_tensor)
            action = torch.argmax(logits).item()
            # moves = game.find_best_moves(game.players[0].rack, 10)
            # if moves:
            #     move = moves[min(action, len(moves) - 1)]
            #     game.play(move[2], move[0], move[3])
            # else:
            #     game.numMoves = -1

            moves = game.find_best_moves(game.players[0].rack, 10)
            print(f"🤖 Agent move options: {len(moves)}")

            if moves:
                move = moves[min(action, len(moves) - 1)]
                print(f"➡️ Agent playing: {move[0]} | Score: {move[1]}")
                game.play(move[2], move[0], move[3])

                print_board(game.board)

            else:
                print("⚠️ Agent has no valid moves.")
                game.numMoves = -1
        else:
            opponent_class(game).play_move()

    agent_score = game.players[agent_id].score
    opponent_score = game.players[opponent_id].score
    print(f"🏁 Final scores → Agent: {agent_score}, Opponent: {opponent_score}")
    return game.players[agent_id].score, game.players[opponent_id].score


    # return game.players[0].score, game.players[1].score

def evaluate(agent_path, opponent_class, games=5):
    model = ActorCritic(action_dim=10)
    model.load_state_dict(torch.load(agent_path))
    model.eval()

    agent_wins = 0
    agent_scores = []
    opponent_scores = []

    for i in range(games):
        a_score, o_score = play_game(model, opponent_class)
        agent_scores.append(a_score)
        opponent_scores.append(o_score)
        if a_score > o_score:
            agent_wins += 1
        print(f"Game {i+1}: Agent {a_score} - Opponent {o_score}")

    print("\n===========================")
    print(f"Agent win rate: {agent_wins / games * 100:.2f}%")
    print(f"Avg agent score: {sum(agent_scores)/games:.2f}")
    print(f"Avg opponent score: {sum(opponent_scores)/games:.2f}")
    print("===========================")


def print_board(board_obj):
    board = board_obj.board
    print("   " + " ".join(f"{i:2}" for i in range(15)))
    for i, row in enumerate(board):
        row_str = ""
        for tile in row:
            row_str += f" {tile.letter if tile.letter else '.'} "
        print(f"{i:2} {row_str}")


if __name__ == "__main__":
    # print("Evaluating agent vs RandomOpponent...")
    # evaluate("scrabble_ppo.pt", RandomOpponent, games=20)

    print("\nEvaluating agent vs GreedyOpponent...")
    evaluate("scrabble_ppo.pt", GreedyOpponent, games=1)