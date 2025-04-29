# evaluate_greedy_vs_greedy.py
import random
from Game import Game

class GreedyOpponent:
    def __init__(self, game):
        self.game = game

    def play_move(self):
        moves = self.game.find_best_moves(self.game.players[self.game.currentPlayer].rack, 50)
        if moves:
            moves.sort(key=lambda x: x[1], reverse=True)  # Sort by highest score
            move = moves[0]  # Pick best move
            self.game.play(move[2], move[0], move[3])
        else:
            self.game.numMoves = -1

def play_game_greedy_vs_greedy():
    game = Game()

    agent_id = 0
    opponent_id = 1

    agent_moves = 0
    opponent_moves = 0

    while game.numMoves >= 0:
        if game.currentPlayer == agent_id:
            GreedyOpponent(game).play_move()
            agent_moves += 1
        else:
            GreedyOpponent(game).play_move()
            opponent_moves += 1

    agent_score = game.players[agent_id].score
    opponent_score = game.players[opponent_id].score

    print(f"🏁 Final scores → Agent: {agent_score}, Opponent: {opponent_score}")
    print(f"🧩 Tiles left in rack (Agent): {game.players[agent_id].rack}")
    print(f"🧩 Tiles left in rack (Opponent): {game.players[opponent_id].rack}")
    print(f"👜 Bag size: {len(game.bag)}")
    print(f"🎮 Agent moves played: {agent_moves}")
    print(f"🎮 Opponent moves played: {opponent_moves}")
    print(f"🔁 Total moves: {agent_moves + opponent_moves}")

def evaluate(games=5):
    agent_wins = 0
    agent_total_score = 0
    opponent_total_score = 0

    for i in range(games):
        print(f"\n🎲 Game {i+1}")
        a_score, o_score = play_game_greedy_vs_greedy()
        agent_total_score += a_score
        opponent_total_score += o_score
        if a_score > o_score:
            agent_wins += 1

    print("\n===========================")
    print(f"🏆 Greedy win rate: {agent_wins / games * 100:.2f}%")
    print(f"📈 Avg agent score: {agent_total_score / games:.2f}")
    print(f"📈 Avg opponent score: {opponent_total_score / games:.2f}")
    print("===========================")

if __name__ == "__main__":
    evaluate(games=3)
