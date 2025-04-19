import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

# Hyperparameters
lr_critic = 5e-4
gamma = 0.99
lam = 0.95
K_epochs = 4
minibatch_size = 128
kl_threshold = 0.2
hidden_dim = 128
max_steps = int(1e6)
NUM_RUNS = 5

# Environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# Value Network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# GAE computation
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    values = np.append(values, 0.0)
    gae = 0
    returns = np.zeros_like(rewards, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        returns[t] = gae + values[t]
    return torch.FloatTensor(returns)

# Flat parameters
def flat_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

def set_flat_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(param.shape))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.shape))
        prev_ind += flat_size

def get_flat_grad(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return torch.cat([grad.view(-1) for grad in grads])

def hessian_vector_product(kl, model, v):
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True, retain_graph=True)
    flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
    grad_v = torch.dot(flat_grad_kl, v)
    hvp = torch.autograd.grad(grad_v, model.parameters(), retain_graph=True)
    return torch.cat([g.contiguous().view(-1) for g in hvp]) + 0.1 * v  # 添加damping


def conjugate_gradients(Av_func, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Av_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def line_search(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.3):
    fval = f(x)
    for stepfrac in [0.5 ** i for i in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params(model, x_new)
        new_fval = f(x_new)
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / (expected_improve + 1e-8)
        if ratio > accept_ratio and actual_improve > 0:
            return True, x_new
    return False, x

def run_trpo(seed=0):
    actor = PolicyNetwork(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards, eval_scores, eval_steps = [], [], []
    total_steps = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed)
        episode_data, episode_reward = [], []
        done = False

        while not done:
            action, log_prob = actor.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_tensor = torch.FloatTensor(state)
            value = critic(state_tensor).item()
            episode_data.append((state, reward, value, log_prob.detach(), action, done))
            episode_reward.append(reward)
            state = next_state
            total_steps += 1

            if total_steps >= 1250 and total_steps % 250 == 0:
                eval_reward = 0
                eval_state, _ = env.reset(seed=seed)
                while True:
                    eval_tensor = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        probs = actor(eval_tensor)
                    action_eval = torch.argmax(probs, dim=-1).item()
                    eval_state, reward, terminated, truncated, _ = env.step(action_eval)
                    eval_reward += reward
                    if terminated or truncated:
                        break
                eval_scores.append(eval_reward)
                eval_steps.append(total_steps)
                print(f"[Eval @ Step {total_steps}] Reward: {eval_reward}")
                episode_rewards.append(sum(episode_reward))

        states, rewards, values, log_probs_old, actions, dones = zip(*episode_data)
        states = torch.FloatTensor(np.array(states))
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = np.array(rewards)
        values_np = np.array(values)
        dones = np.array(dones)

        returns = compute_gae(rewards, values_np, dones, gamma, lam)
        values_tensor = torch.FloatTensor(values_np)
        advantages = returns - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_probs_old_tensor = torch.stack(log_probs_old).detach()

        dataset_size = len(states)
        for _ in range(K_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_log_probs_old = log_probs_old_tensor[mb_idx]

                mb_probs = actor(mb_states)
                mb_dist = torch.distributions.Categorical(mb_probs)
                mb_log_probs = mb_dist.log_prob(mb_actions)

                ratio = torch.exp(mb_log_probs - mb_log_probs_old)
                surrogate = (ratio * mb_advantages).mean()

                with torch.no_grad():
                    old_probs = mb_probs.detach()
                    old_dist = torch.distributions.Categorical(probs=old_probs)
                kl = torch.distributions.kl.kl_divergence(old_dist, mb_dist).mean()

                loss_pi = -surrogate
                g = get_flat_grad(loss_pi, actor).detach()

                def Hx(v):
                    return hessian_vector_product(kl, actor, v)

                step_dir = conjugate_gradients(Hx, g)
                shs = 0.5 * (step_dir * Hx(step_dir)).sum(0, keepdim=True)
                step_size = torch.sqrt(2 * kl_threshold / (shs + 1e-8))
                fullstep = step_size * step_dir
                old_params = flat_params(actor).detach()

                def get_loss_kl(params):
                    set_flat_params(actor, params)
                    probs = actor(mb_states)
                    dist = torch.distributions.Categorical(probs)
                    log_probs = dist.log_prob(mb_actions)
                    ratio = torch.exp(log_probs - mb_log_probs_old)
                    surrogate = -(ratio * mb_advantages).mean()
                    kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
                    return surrogate + kl

                success, new_params = line_search(actor, get_loss_kl, old_params, fullstep, g.dot(step_dir))
                set_flat_params(actor, new_params)

                mb_value_preds = critic(mb_states).squeeze()
                mb_value_loss = F.mse_loss(mb_value_preds, mb_returns)
                optimizer_critic.zero_grad()
                mb_value_loss.backward()
                optimizer_critic.step()

    return episode_rewards, eval_scores, eval_steps

if __name__ == "__main__":
    all_scores, all_eval_scores, all_eval_steps, all_steps = [], [], [], []

    for run in range(NUM_RUNS):
        scores, eval_scores, eval_steps = run_trpo(seed=run)
        all_scores.append(scores)
        all_eval_scores.append(eval_scores)
        all_eval_steps.append(eval_steps)
        all_steps.append(len(scores))

    max_len = max(len(run) for run in all_scores)
    all_scores = [run + [np.nan] * (max_len - len(run)) for run in all_scores]
    avg_reward = np.nanmean(all_scores, axis=0)
    std_reward = np.nanstd(all_scores, axis=0)

    avg_eval_scores = np.nanmean([run + [np.nan] * (max_len - len(run)) for run in all_eval_scores], axis=0)
    std_eval_scores = np.nanstd([run + [np.nan] * (max_len - len(run)) for run in all_eval_scores], axis=0)

    df_eval = pd.DataFrame({
        'steps': all_eval_steps[0],
        'avg_reward': avg_eval_scores,
        'std_reward': std_eval_scores
    })
    os.makedirs('./results', exist_ok=True)
    df_eval.to_csv('./results/trpo_eval_score.csv', index=False)

    df = pd.DataFrame({
        'steps': all_eval_steps[0],
        'avg_reward': avg_reward,
        'std_reward': std_reward
    })
    df.to_csv('./results/trpo_train_results.csv', index=False)

    print("\n TRPO Results saved to ./results/")
    print("\n Summary:")
    print(df[['avg_reward']].agg(['mean', 'max']))