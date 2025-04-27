from scrabble_rl.env import ScrabbleEnv
from scrabble_rl.q_agent import QAgent
import torch

def train_q_agent(episodes=5):
    env = ScrabbleEnv()
    agent = QAgent(action_dim=10)

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        print(f"Episode {ep}: Total Reward = {total_reward}")

    torch.save(agent.q_table, "q_agent.pt")
    print("✅ Q-learning agent saved.")

if __name__ == "__main__":
    train_q_agent()
