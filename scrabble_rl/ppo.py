 
import torch
import torch.nn as nn
import torch.optim as optim
from scrabble_rl.ppo_utils import compute_gae

class PPOTrainer:
    def __init__(self, env, model, clip_epsilon=0.2, lr=3e-4, gamma=0.99, lam=0.95):
        self.env = env
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam

    def train(self, epochs=1000):
        for epoch in range(epochs):
            print(f"\n🔁 Starting Epoch {epoch}")

            obs = self.env.reset()
            done = False
            trajectory = []
            print("🧠 Playing episode...")

            while not done:
                obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}
                logits, value = self.model(obs_tensor)
                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, done, _ = self.env.step(action.item())
                trajectory.append((obs_tensor, action, reward, value, log_prob))
                obs = next_obs

            episode_reward = sum(r for (_, _, r, _, _) in trajectory)
            print(f"✅ Episode complete. Total reward: {episode_reward}")

            self.update(trajectory)
            print("🔧 PPO update complete.\n" + "-" * 50)

    def update(self, trajectory):
        states, actions, rewards, values, log_probs = zip(*trajectory)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        returns, advantages = compute_gae(rewards, values, self.gamma, self.lam)

        for _ in range(4):
            logits_batch, value_batch = zip(*[self.model(s) for s in states])
            logits_batch = torch.stack(logits_batch)
            values_new = torch.stack(value_batch).squeeze()

            dist = torch.distributions.Categorical(logits=logits_batch)
            log_probs_new = dist.log_prob(torch.stack(actions))

            ratio = torch.exp(log_probs_new - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values_new, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
