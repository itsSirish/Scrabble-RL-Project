# scrabble_rl/train_deep_q.py

import torch
from scrabble_rl.deep_q_agent import DeepQAgent
from scrabble_rl.env import ScrabbleEnv
from utils import mapBoardToState
import random
import os

def main():
    env = ScrabbleEnv()
    agent = DeepQAgent()
    episodes = 5  # You can increase to 1000+ later
    target_update_freq = 10
    save_path = "scrabble_deepq.pt"

    all_rewards = []

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Prepare input for Q-network
            obs_tensor = {k: torch.tensor(v).float().unsqueeze(0).to(agent.device) for k, v in obs.items()}

            moves_available = 10  # Always top-10 moves
            action_idx = agent.select_action(obs_tensor, moves_available)

            next_obs, reward, done, _ = env.step(action_idx)
            next_obs_tensor = {k: torch.tensor(v).float().unsqueeze(0).to(agent.device) for k, v in next_obs.items()}

            agent.store_transition(obs_tensor, action_idx, reward, next_obs_tensor, done)

            obs = next_obs
            total_reward += reward

            agent.train_step()

        agent.decay_epsilon()

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode+1}/{episodes} | Total reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
        all_rewards.append(total_reward)

    # Save model
    torch.save(agent.q_network.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")

if __name__ == "__main__":
    main()
