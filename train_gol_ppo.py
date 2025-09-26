import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.vector

from main import GameOfLifePufferEnv


class MLPPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = int(np.prod(env.single_observation_space.shape))
        space = env.single_action_space
        if hasattr(space, "n"):
            act_size = int(space.n)
        elif hasattr(space, "nvec"):
            # MultiDiscrete: we want one logit per binary decision
            act_size = int(len(space.nvec))
        else:
            act_size = 18
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, act_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        hidden = self.net(x)
        return self.actor(hidden), self.critic(hidden)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy, value


def train(
    total_timesteps=400000,
    num_envs=8,
    num_steps=256,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=8,
    update_epochs=8,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device=None,
    eval_render=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Keep training and eval observation shapes identical
    grid_width = 32
    grid_height = 32

    envs = pufferlib.vector.make(
        GameOfLifePufferEnv,
        backend=pufferlib.vector.Multiprocessing,
        num_envs=num_envs,
        env_kwargs={
            "width": grid_width,
            "height": grid_height,
            "rule_name": "Conway",
            "density": 0.3,
            "render_mode": "ansi",
            "max_steps": None,
            "rl_mode": "learn_rule",
            "reward_mode": "stability_entropy",
        },
    )

    policy = MLPPolicy(envs.driver_env).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    obs = torch.zeros(
        (num_steps, num_envs) + envs.single_observation_space.shape,
        dtype=torch.uint8,
        device=device,
    )
    # For MultiDiscrete(18), we store int actions with shape [steps, envs, 18]
    actions = torch.zeros((num_steps, num_envs, 18), dtype=torch.long, device=device)
    logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.uint8, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()
    # Track episodic returns/lengths per env for live metrics
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    completed = []

    for iteration in range(1, num_iterations + 1):
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            obs_float = next_obs.float()
            with torch.no_grad():
                # Sample 18 independent Bernoulli logits by splitting the actor output
                logits, value = policy.forward(obs_float.view(num_envs, -1))
                # Expand to 18 binary decisions via a linear head
                # Reuse logits as if mapping to 18 outputs
                bin_logits = logits[:, :18]
                bern = torch.distributions.Bernoulli(logits=bin_logits)
                action = bern.sample().long()
                logprob = bern.log_prob(action.float()).sum(dim=1)
                entropy = bern.entropy().sum(dim=1)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(
                action.detach().cpu().numpy()
            )
            done_np = np.logical_or(terminated_np, truncated_np).astype(np.float32)

            next_obs = torch.tensor(next_obs_np, dtype=torch.uint8, device=device)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            # Update per-env episodic stats
            ep_returns += reward_np
            ep_lengths += 1
            for j, d in enumerate(done_np):
                if d:
                    completed.append(ep_returns[j])
                    ep_returns[j] = 0.0
                    ep_lengths[j] = 0

        with torch.no_grad():
            next_value = policy.get_action_and_value(
                next_obs.float().view(num_envs, -1)
            )[3].reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape).float()
        b_actions = actions.reshape(-1, 18)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                # Recompute logprobs under new policy for the sampled binary actions
                logits, new_value = policy.forward(
                    b_obs[mb_inds].view(len(mb_inds), -1)
                )
                bin_logits = logits[:, :18]
                bern = torch.distributions.Bernoulli(logits=bin_logits)
                new_logprob = bern.log_prob(b_actions[mb_inds].float()).sum(dim=1)
                entropy = bern.entropy().sum(dim=1)
                ratio = (new_logprob - b_logprobs[mb_inds]).exp()
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - 0.1, 1 + 0.1)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (new_value.flatten() - b_returns[mb_inds]).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        iter_score = float(rewards.mean().item())
        if completed:
            mean_ret = float(np.mean(completed))
            print(
                f"Iter {iteration}/{num_iterations} | Steps {global_step} | SPS {sps} | IterScore {iter_score:.3f} | EpRet {mean_ret:.3f} (n={len(completed)})"
            )
            completed.clear()
        else:
            print(
                f"Iter {iteration}/{num_iterations} | Steps {global_step} | SPS {sps} | IterScore {iter_score:.3f}"
            )

    # Save checkpoint
    import os

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "gol_policy.pt")
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved policy to {ckpt_path}")

    envs.driver_env.close()

    # Optional short evaluation with live rendering
    if eval_render:
        eval_env = pufferlib.vector.make(
            GameOfLifePufferEnv,
            backend=pufferlib.vector.Serial,
            num_envs=1,
            env_kwargs={
                "width": grid_width,
                "height": grid_height,
                "rule_name": "Conway",
                "density": 0.3,
                "render_mode": "human",
                "max_steps": None,
                "rl_mode": "learn_rule",
                "reward_mode": "stability_entropy",
            },
        )
        # Rebuild same policy for eval and load weights
        eval_policy = MLPPolicy(eval_env.driver_env).to(device)
        eval_policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        eval_policy.eval()

        obs, _ = eval_env.reset()
        obs_t = torch.tensor(obs, dtype=torch.uint8, device=device)
        steps = 0
        try:
            while steps < 1000:
                with torch.no_grad():
                    logits, _ = eval_policy.forward(obs_t.float().view(1, -1))
                    bin_logits = logits[:, :18]
                    act = (
                        torch.distributions.Bernoulli(logits=bin_logits).sample().long()
                    )
                obs_np, _, term, trunc, _ = eval_env.step(act.cpu().numpy())
                eval_env.driver_env.render()
                if term[0] or trunc[0]:
                    obs_np, _ = eval_env.reset()
                obs_t = torch.tensor(obs_np, dtype=torch.uint8, device=device)
                steps += 1
        except KeyboardInterrupt:
            pass
        finally:
            eval_env.driver_env.close()


if __name__ == "__main__":
    train()
