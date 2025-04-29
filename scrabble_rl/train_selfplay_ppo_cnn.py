import torch
import torch.optim as optim
from scrabble_rl.selfplay_env_dynamic import SelfPlayScrabbleEnv
from scrabble_rl.model_cnnppo import ActorCriticCNN
from scrabble_rl.ppo_utils import compute_gae
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
import os

class PPOTrainerSelfPlay:
    def __init__(self, agent1, agent2, env, lr=3e-4, clip_epsilon=0.2, gamma=0.99, lam=0.95):
        self.agent1 = agent1
        self.agent2 = agent2
        self.env = env
        self.optimizer1 = optim.Adam(agent1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(agent2.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam
        self.agent1_rewards = []
        self.agent2_rewards = []
        self.agent1_wins = []
        self.agent2_wins = []
        self.score_diffs = []
        self.word_lengths = []
        self.logs = []
        os.makedirs("scrabble_rl/Outputs", exist_ok=True)

    def train(self, epochs=1000):
        for epoch in range(epochs):
            print(f"\n\U0001F3AE Epoch {epoch} | Self-Play Starting...")

            obs = self.env.reset()
            done = False
            trajectory1, trajectory2 = [], []
            current_agent = 0

            cumulative_reward1 = 0
            cumulative_reward2 = 0

            while not done:
                obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}

                if current_agent == 0:
                    logits, value = self.agent1(obs_tensor)
                else:
                    logits, value = self.agent2(obs_tensor)

                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, done, _ = self.env.step(action.item())

                if current_agent == 0:
                    trajectory1.append((obs_tensor, action, reward, value, log_prob))
                    cumulative_reward1 += reward
                else:
                    trajectory2.append((obs_tensor, action, -reward, value, log_prob))
                    cumulative_reward2 += reward

                obs = next_obs
                current_agent = not current_agent

            self.update(self.agent1, self.optimizer1, trajectory1)
            self.update(self.agent2, self.optimizer2, trajectory2)

            self.agent1_rewards.append(cumulative_reward1)
            self.agent2_rewards.append(cumulative_reward2)

            score1 = self.env.game.players[0].score
            score2 = self.env.game.players[1].score
            self.score_diffs.append(abs(score1 - score2))
            self.agent1_wins.append(int(score1 > score2))
            self.agent2_wins.append(int(score2 > score1))
            self.word_lengths.extend([len(move[1][0]) for move in self.env.moves_played])

            self.env.relaxation_prob = max(0.05, self.env.relaxation_prob * np.exp(-0.01 * epoch))

            board_state_str = str(self.env.game.board)
            log = {
                "epoch": epoch,
                "final_score": {"agent1": score1, "agent2": score2},
                "winner": "Agent1" if score1 > score2 else "Agent2",
                "board_state": board_state_str,
                "moves_played": [
                    {
                        "agent": "Agent1" if p == 0 else "Agent2",
                        "word": m[0],
                        "position": m[2],
                        "direction": "HORIZ" if m[3] == 0 else "VERT",
                        "score": m[1]
                    } for p, m in self.env.moves_played
                ]
            }
            self.logs.append(log)

            print("\n\U0001F3C1 Game Over!")
            self.env.render()
            print("\n\U0001F3AF Moves Played:")
            for player, move in self.env.moves_played:
                player_str = "Agent1" if player == 0 else "Agent2"
                print(f"{player_str} played {move[0]} at {move[2]} direction {'HORIZ' if move[3]==0 else 'VERT'} | Score: {move[1]}")

            print(f"\n\U0001F535 Final Score: Agent1 = {score1} | Agent2 = {score2}")
            print(f"\U0001F504 Relaxation now: {self.env.relaxation_prob:.4f}")
            print(f"\u2705 Epoch {epoch} complete | Agent1 Reward: {cumulative_reward1} | Agent2 Reward: {cumulative_reward2}")

        torch.save(self.agent1.state_dict(), "scrabble_rl/Outputs/Best_agent1_selfplay.pt")
        torch.save(self.agent2.state_dict(), "scrabble_rl/Outputs/Best_agent2_selfplay.pt")
        with open("scrabble_rl/Outputs/selfplay_logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)
        print("\u2705 Models and logs saved to scrabble_rl/Outputs.")

        self.plot_all()

    def update(self, agent, optimizer, trajectory):
        states, actions, rewards, values, log_probs = zip(*trajectory)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        returns, advantages = compute_gae(rewards, values, self.gamma, self.lam)

        for _ in range(4):
            logits_batch, value_batch = zip(*[agent(s) for s in states])
            logits_batch = torch.stack(logits_batch)
            values_new = torch.stack(value_batch).squeeze()

            dist = torch.distributions.Categorical(logits=logits_batch)
            log_probs_new = dist.log_prob(torch.stack(actions))

            ratio = torch.exp(log_probs_new - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(values_new, returns)
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def plot_all(self):
        self.plot_rewards()
        self.plot_win_rate()
        self.plot_score_diffs()
        self.plot_word_lengths()

    def plot_rewards(self):
        plt.plot(self.agent1_rewards, label="Agent 1")
        plt.plot(self.agent2_rewards, label="Agent 2")
        plt.title("Total Reward per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.grid()
        plt.legend()
        plt.savefig("scrabble_rl/Outputs/selfplay_reward_curve.png")
        # plt.show()

    def plot_win_rate(self):
        epochs = np.arange(1, len(self.agent1_wins) + 1)
        win1 = np.cumsum(self.agent1_wins) / epochs
        win2 = np.cumsum(self.agent2_wins) / epochs
        plt.plot(win1, label="Agent 1 Win Rate")
        plt.plot(win2, label="Agent 2 Win Rate")
        plt.title("Win Rate Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Win Rate")
        plt.grid()
        plt.legend()
        plt.savefig("scrabble_rl/Outputs/win_rate_curve.png")
        # plt.show()

    def plot_score_diffs(self):
        plt.plot(self.score_diffs)
        plt.title("Score Difference Per Game")
        plt.xlabel("Epoch")
        plt.ylabel("Score Diff")
        plt.grid()
        plt.savefig("scrabble_rl/Outputs/score_diff_curve.png")
        # plt.show()

    def plot_word_lengths(self):
        counts = Counter(self.word_lengths)
        lengths, freqs = zip(*sorted(counts.items()))
        plt.bar(lengths, freqs)
        plt.title("Word Length Distribution")
        plt.xlabel("Word Length")
        plt.ylabel("Frequency")
        plt.savefig("scrabble_rl/Outputs/word_length_histogram.png")
        # plt.show()

def main():
    agent1 = ActorCriticCNN(action_dim=50)
    agent2 = ActorCriticCNN(action_dim=50)
    env = SelfPlayScrabbleEnv(agent1, agent2)

    trainer = PPOTrainerSelfPlay(agent1, agent2, env)
    trainer.train(epochs=25)

if __name__ == "__main__":
    main()
