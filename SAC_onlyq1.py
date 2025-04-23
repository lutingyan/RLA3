# Soft Actor-Critic (SAC) for CartPole (discrete action version)
import gymnasium as gym
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import pandas as pd

# Set up environment and hyperparameters
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr_actor = 3e-4
lr_critic = 3e-4
lr_alpha = 3e-4
alpha = 0.2
hidden_dim = 128
gamma = 0.99
tau = 0.005
batch_size = 64
replay_size = int(1e5)
max_steps = int(1e6)
NUM_RUNS = 5
eval_interval = 250

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def push(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)

# Networks
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)  # Log probs

    def sample(self, state):
        log_probs = self.forward(state)
        probs = log_probs.exp()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, log_probs.gather(1, action.unsqueeze(1))

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Outputs Q-values for each action

def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

# Main SAC Training Loop
def run_sac(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=lr_actor)
    q1_optim = optim.Adam(q1.parameters(), lr=lr_critic)
    q2_optim = optim.Adam(q2.parameters(), lr=lr_critic)

    replay_buffer = ReplayBuffer(replay_size)
    total_steps = 0
    episode_rewards = []
    eval_scores = []
    eval_steps = []

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = actor.sample(state_tensor)
            action = action.item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = [x.to(device) for x in batch]

                with torch.no_grad():
                    next_log_probs = actor(next_states)
                    next_probs = next_log_probs.exp()

                    q1_next = q1_target(next_states)
                    q2_next = q2_target(next_states)
                    min_q_next = q1_next

                    next_v = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=1)
                    target_q = rewards + (1 - dones) * gamma * next_v

                q1_preds = q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                q2_preds = q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                q1_loss = F.mse_loss(q1_preds, target_q)
                q2_loss = F.mse_loss(q2_preds, target_q)

                q1_optim.zero_grad()
                q1_loss.backward()
                q1_optim.step()

                q2_optim.zero_grad()
                q2_loss.backward()
                q2_optim.step()

                log_probs = actor(states)
                probs = log_probs.exp()
                q1_vals = q1(states)
                q2_vals = q2(states)
                min_q = q1_vals

                actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=1).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                soft_update(q1_target, q1, tau)
                soft_update(q2_target, q2, tau)

            if total_steps >= 1250 and total_steps % eval_interval == 0:
                eval_reward = 0
                eval_state, _ = env.reset(seed=seed)
                done_eval = False
                while not done_eval:
                    eval_tensor = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        log_probs = actor(eval_tensor)
                        probs = log_probs.exp()
                        action_eval = torch.argmax(probs, dim=-1).item()
                    eval_state, reward, terminated, truncated, _ = env.step(action_eval)
                    eval_reward += reward
                    done_eval = terminated or truncated
                eval_scores.append(eval_reward)
                eval_steps.append(total_steps)
                episode_rewards.append(episode_reward)
                print(f"[Eval @ Step {total_steps}] Reward: {eval_reward}")

    return episode_rewards, eval_scores, eval_steps

if __name__ == "__main__":
    all_scores = []
    all_eval_scores = []
    all_eval_steps = []

    for run in range(NUM_RUNS):
        scores, eval_scores, eval_steps = run_sac(seed=run)
        all_scores.append(scores)
        all_eval_scores.append(eval_scores)
        all_eval_steps.append(eval_steps)

    max_len = max(len(s) for s in all_scores)
    all_scores = [s + [np.nan] * (max_len - len(s)) for s in all_scores]
    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    max_eval_len = max(len(s) for s in all_eval_scores)
    all_eval_scores = [s + [np.nan] * (max_eval_len - len(s)) for s in all_eval_scores]
    all_eval_steps = [s + [np.nan] * (max_eval_len - len(s)) for s in all_eval_steps]

    avg_eval_scores = np.nanmean(all_eval_scores, axis=0)
    std_eval_scores = np.nanstd(all_eval_scores, axis=0)

    df_eval = pd.DataFrame({
        'steps': all_eval_steps[0],
        'avg_reward': avg_eval_scores,
        'std_reward': std_eval_scores
    })
    os.makedirs('./results', exist_ok=True)
    df_eval.to_csv('./results/sac_q1_score.csv', index=False)

    df_train = pd.DataFrame({
        'steps': all_eval_steps[0],
        'avg_reward': avg_reward,
        'std_reward': std_reward
    })
    df_train.to_csv('./results/sac_q1_results.csv', index=False)

    print("\n SAC Results saved to ./results/")
    print("\n Summary:")
    print(df_train[['avg_reward']].agg(['mean', 'max']))