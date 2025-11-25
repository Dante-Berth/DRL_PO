#!/usr/bin/env python3
# ppo_algo_rl_style.py
# PPO training wrapped in an AlgoRL-style class using Config dataclasses
# Adapted from CleanRL's PPO continuous implementation, restructured to match
# the SAC example (Config dataclasses + class inheriting AlgoRL).

import os
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

from src.utils.algo import AlgoRL
import src.envs  # ensure env registration happens
from src.utils.nn import PPOAgent


# -----------------------
# Config dataclasses
# -----------------------
@dataclass
class OUEnvConfig:
    sigma: float = 0.3
    theta: float = 0.1
    T: int = 5000
    random_state: Optional[int] = None
    lambd: float = 0.3
    psi: float = 1
    cost: str = "trade_0"
    max_pos: int = 10
    squared_risk: bool = True
    penalty: str = "none"
    alpha: float = 10
    beta: float = 10
    clip: bool = True
    noise: bool = False
    noise_std: float = 10.0
    noise_seed: Optional[int] = None
    scale_reward: float = 10.0


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None


@dataclass
class TrainConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    env_id: str = "OUTradingEnv-v0"
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    num_envs: int = 1
    capture_video: bool = False


@dataclass
class Config:
    train: TrainConfig = TrainConfig()
    algo: PPOConfig = PPOConfig()
    env: OUEnvConfig = OUEnvConfig()


# -----------------------
# Helpers & network
# -----------------------


def make_env(env_id: str, env_kwargs: dict):
    def thunk():
        # pass env kwargs to env constructor via gym.make(..., **kwargs)
        env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# -----------------------
# PPO algorithm as AlgoRL subclass
# -----------------------
class PPO(AlgoRL):
    def __init__(self, config: Config):
        super().__init__(config)
        # AlgoRL is expected to set up: self.train, self.algo, self.envs, self.device, self.writer
        # compute derived sizes
        self.num_envs = self.train.num_envs
        self.num_steps = self.algo.num_steps
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.algo.num_minibatches)

        # placeholders for components that will be created in init_algo_rl
        self.agent = None
        self.optimizer = None

        # storage tensors — initialized in init_algo_rl (after envs available)
        self.obs = None
        self.actions = None
        self.logprobs = None
        self.rewards = None
        self.dones = None
        self.values = None

    def init_algo_rl(self):
        device = self.device
        envs = self.envs

        # actor-critic
        self.agent = PPOAgent(envs).to(device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.algo.learning_rate, eps=1e-5
        )

        # storage
        obs_shape = envs.single_observation_space.shape
        act_shape = envs.single_action_space.shape

        self.obs = torch.zeros((self.num_steps, self.num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + act_shape).to(
            device
        )
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)

    def train(self):
        device = self.device
        args = self.algo
        envs = self.envs
        num_envs = self.train.num_envs
        episode_counts = np.zeros(num_envs, dtype=int)
        total_cumulative_returns = np.zeros((num_envs), dtype=float)
        cumulative_returns = np.zeros((num_envs), dtype=float)
        global_step = 0

        # reset envs
        next_obs, _ = envs.reset(seed=self.train.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        num_updates = args.total_timesteps // self.batch_size

        for update in range(1, num_updates + 1):
            # anneal lr
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # action selection
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.logprobs[step] = logprob

                # step env
                next_obs, rewards, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(rewards).to(device).view(-1)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(device),
                    torch.Tensor(next_done).to(device),
                )

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            self.writer.add_scalar(
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            self.writer.add_scalar(
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            # bootstrap value
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = (
                        self.rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # update policy
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # approx KL (for early stopping or diagnostics)
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), args.max_grad_norm
                    )
                    self.optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )

        envs.close()


if __name__ == "__main__":
    config = tyro.cli(Config)
    ppo = PPO(config)
    ppo.train()
