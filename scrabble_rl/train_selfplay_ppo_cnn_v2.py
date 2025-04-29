# scrabble_rl/train_selfplay_ppo_cnn_better.py

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scrabble_rl.selfplay_env_dynamic_v2 import SelfPlayScrabbleEnv
from scrabble_rl.model_cnnppo_v2 import ActorCriticCNNBetter
from scrabble_rl.ppo_utils import compute_gae

class PPOTrainerSelfPlayBetter:
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

    def train(self, epochs=25):
        for epoch in range(epochs):
            print(f"\n🎮 Epoch {epoch} | Self-Play Starting... | Relaxation: {self.env.relaxation_prob:.2f}")

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

            print(f"✅ Epoch {epoch} complete | Agent1 Reward: {cumulative_reward1:.1f} | Agent2 Reward: {cumulative_reward2:.1f}")

            # Relaxation decay
            self.env.relaxation_prob = max(0.05, self.env.relaxation_prob * 0.95)

        torch.save(self.agent1.state_dict(), "Better_agent1_selfplay.pt")
        torch.save(self.agent2.state_dict(), "Better_agent2_selfplay.pt")
        print("✅ Models saved.")

        self.plot_rewards()

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

    def plot_rewards(self):
        plt.plot(self.agent1_rewards, label="Agent 1")
        plt.plot(self.agent2_rewards, label="Agent 2")
        plt.title("Self-Play Training Rewards")
        plt.xlabel("Epoch")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid()
        plt.savefig("better_selfplay_rewards.png")
        plt.show()

def main():
    agent1 = ActorCriticCNNBetter(action_dim=50)
    agent2 = ActorCriticCNNBetter(action_dim=50)
    env = SelfPlayScrabbleEnv(agent1, agent2, relaxation_prob=0.3)

    trainer = PPOTrainerSelfPlayBetter(agent1, agent2, env)
    trainer.train(epochs=25)

if __name__ == "__main__":
    main()
