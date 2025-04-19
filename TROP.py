import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
from torch.distributions import Categorical
from torch.autograd import Variable
from scipy.optimize import minimize

# Hyperparameters
lr_critic = 1e-3
gamma = 0.99
lam = 0.95
delta = 0.01  # KL divergence threshold
max_steps = int(1e6)
NUM_RUNS = 5
hidden_dim = 128
batch_size = 2048
train_critic_iters = 80
train_policy_iters = 80
cg_iters = 10
backtrack_iters = 10
backtrack_coeff = 0.8

# Environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

torch.backends.cudnn.benchmark = True


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
        dist = Categorical(probs)
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


def conjugate_gradient(Avp, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x


def line_search(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = f().item()
    for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f().item()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def hessian_vector_product(model, vector, states, old_dist):
    model.zero_grad()
    probs = model(states)
    dist = Categorical(probs)
    kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
    kl_grad = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    kl_grad_flat = torch.cat([grad.view(-1) for grad in kl_grad])
    kl_grad_vector_product = (kl_grad_flat * vector).sum()
    grad2 = torch.autograd.grad(kl_grad_vector_product, model.parameters())
    grad2_flat = torch.cat([grad.contiguous().view(-1) for grad in grad2])
    return grad2_flat + 0.1 * vector


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

        # Collect trajectories
        while len(episode_data) < batch_size:
            action, log_prob = actor.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_tensor = torch.FloatTensor(state)
            value = critic(state_tensor).item()
            episode_data.append((state, reward, value, log_prob.detach(), action, done))
            episode_reward.append(reward)
            state = next_state
            total_steps += 1

            # 完全保留您要求的评估频率和方式
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

            if done:
                state, _ = env.reset(seed=seed)
                episode_rewards.append(sum(episode_reward))
                episode_reward = []

        # 以下是TRPO的核心算法部分，替换了原来的PPO更新
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

        # Update critic
        for _ in range(train_critic_iters):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), 64):
                end = start + 64
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_returns = returns[mb_idx]

                value_preds = critic(mb_states).squeeze()
                value_loss = F.mse_loss(value_preds, mb_returns)

                optimizer_critic.zero_grad()
                value_loss.backward()
                optimizer_critic.step()

        # TRPO policy update
        probs = actor(states)
        old_dist = Categorical(probs.detach())

        def get_loss():
            probs = actor(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - log_probs_old_tensor)
            return -(ratio * advantages).mean()

        def get_kl():
            probs = actor(states)
            dist = Categorical(probs)
            return torch.distributions.kl.kl_divergence(old_dist, dist).mean()

        loss = get_loss()
        grads = torch.autograd.grad(loss, actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        def Fvp(v):
            return hessian_vector_product(actor, v, states, old_dist)

        stepdir = conjugate_gradient(Fvp, -loss_grad, nsteps=cg_iters)
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / delta)
        fullstep = stepdir / lm[0]

        expected_improve = (-loss_grad * stepdir).sum(0, keepdim=True)

        prev_params = get_flat_params_from(actor)
        success, new_params = line_search(actor, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(actor, new_params)

        if not success:
            print("Line search failed!")
            set_flat_params_to(actor, prev_params)

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

    print("\nTRPO Results saved to ./results/")
    print("\nSummary:")
    print(df[['avg_reward']].agg(['mean', 'max']))
