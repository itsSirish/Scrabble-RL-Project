from scrabble_rl.env import ScrabbleEnv
from scrabble_rl.model import ActorCritic
from scrabble_rl.ppo import PPOTrainer
import torch

def main():
    env = ScrabbleEnv()
    model = ActorCritic(action_dim=10)
    trainer = PPOTrainer(env, model)
    trainer.train(epochs=100)
    torch.save(model.state_dict(), "scrabble_ppo100.pt")
    print("✅ Model saved to scrabble_ppo.pt")


if __name__ == "__main__":
    main()
 
