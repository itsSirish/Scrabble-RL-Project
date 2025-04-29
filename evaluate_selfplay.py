# scrabble_rl/evaluate_selfplay.py

import torch
from scrabble_rl.model_cnnppo import ActorCriticCNN
from Game import Game
from utils import mapBoardToState

def play_selfplay_game(agent1, agent2):
    game = Game()
    current_agent = 0  # 0: agent1, 1: agent2

    while game.numMoves >= 0:
        obs = mapBoardToState(game, game.players[current_agent].rack)
        obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}
        
        if current_agent == 0:
            logits, _ = agent1(obs_tensor)
        else:
            logits, _ = agent2(obs_tensor)

        probs = torch.softmax(logits, dim=0)
        action = torch.argmax(probs).item()

        moves = game.find_best_moves(game.players[current_agent].rack, 50)
        if moves:
            move = moves[min(action, len(moves)-1)]
            game.play(move[2], move[0], move[3])
        else:
            game.numMoves = -1

        current_agent = not current_agent

    return game.players[0].score, game.players[1].score

def evaluate_selfplay(agent1_path, agent2_path, games=20):
    agent1 = ActorCriticCNN(action_dim=50)
    agent2 = ActorCriticCNN(action_dim=50)
    agent1.load_state_dict(torch.load(agent1_path))
    agent2.load_state_dict(torch.load(agent2_path))
    agent1.eval()
    agent2.eval()

    agent1_wins = 0
    agent2_wins = 0

    for i in range(games):
        score1, score2 = play_selfplay_game(agent1, agent2)
        print(f"Game {i+1}: Agent1 {score1} - Agent2 {score2}")
        if score1 > score2:
            agent1_wins += 1
        else:
            agent2_wins += 1

    print("\n=========================")
    print(f"Agent1 win rate: {agent1_wins/games*100:.2f}%")
    print(f"Agent2 win rate: {agent2_wins/games*100:.2f}%")
    print("=========================")

if __name__ == "__main__":
    evaluate_selfplay("agent1_selfplay.pt", "agent2_selfplay.pt", games=20)
