import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    values = torch.cat((values, torch.tensor([0.0])))
    gae = 0
    returns = []
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    return torch.tensor(returns), torch.tensor(advantages)
