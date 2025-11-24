#!/usr/bin/env python3
# train_sac.py
# SAC training using Config dataclasses: config.train, config.sac, config.env

import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils.buffers import ReplayBuffer
import src.envs  # ensure env registration happens if you placed register() in src.envs.__init__
from src.utils.nn import SoftQNetwork, ActorContinous
from src.utils.algo import AlgoRL


# -----------------------
# Config dataclasses
# -----------------------
@dataclass
class OUEnvConfig:
    sigma: float = 0.5
    theta: float = 1.0
    T: int = 1000
    random_state: Optional[int] = None
    lambd: float = 0.5
    psi: float = 0.5
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
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    autotune: bool = True
    buffer_size: int = int(1e6)
    batch_size: int = 256
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    total_timesteps: int = int(1e6)
    learning_starts: int = 5_000  # training starts after X steps


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
    algo: SACConfig = SACConfig()
    env: OUEnvConfig = OUEnvConfig()


# -----------------------
# Environment factory
# -----------------------
def make_env(env_id: str, env_kwargs: dict):
    def thunk():
        # pass env kwargs to env constructor via gym.make(..., **kwargs)
        env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class SAC(AlgoRL):
    def __init__(self, config):
        super().__init__(config)
        self.rb = ReplayBuffer(
            self.env.buffer_size,
            self.envs.observation_space,
            self.envs.action_space,
            self.device,
            n_envs=self.train.num_envs,
            handle_timeout_termination=False,
        )

    def init_algo_rl(self):
        device = self.device
        self.actor = ActorContinous(self.envs).to(device)
        self.qf1 = SoftQNetwork(self.envs).to(device)
        self.qf2 = SoftQNetwork(self.envs).to(device)
        self.qf1_target = SoftQNetwork(self.envs).to(device)
        self.qf2_target = SoftQNetwork(self.envs).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.env.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.env.policy_lr
        )

        # Automatic entropy tuning
        if self.env.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.envs.single_action_space.shape).to(device)
            ).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = log_alpha.exp().item()
            self.a_optimizer = optim.Adam([log_alpha], lr=self.env.q_lr)
        else:
            self.alpha = self.env.alpha
            log_alpha = None
            self.a_optimizer = None
            self.target_entropy = None
        return None

    @staticmethod
    def make_env(env_id: str, env_kwargs: dict):
        def thunk():
            # pass env kwargs to env constructor via gym.make(..., **kwargs)
            env = gym.make(env_id, **env_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    def train(self):
        num_envs = self.train.num_envs
        episode_counts = np.zeros(num_envs, dtype=int)
        total_cumulative_returns = np.zeros((num_envs))
        cumulative_returns = np.zeros((num_envs))
        import tqdm

        device = self.device
        obs, _ = self.envs.reset(seed=self.train.seed)
        for global_step in tqdm(range(self.algo.total_timesteps)):
            # action selection
            if global_step < self.algo.learning_starts:
                actions = np.array(
                    [self.envs.single_action_space.sample() for _ in range(num_envs)]
                )
            else:
                with torch.no_grad():
                    actions_t, _, _ = self.actor.get_action(
                        torch.Tensor(obs).to(device)
                    )
                actions = actions_t.detach().cpu().numpy()

            # step
            next_obs, rewards, terminations, truncations, infos = self.envs.step(
                actions
            )
            cumulative_returns += rewards
            dones = terminations | truncations  # gymnasium style
            done_mask = dones.astype(bool)

            # store episodic returns for finished envs
            total_cumulative_returns[done_mask] += cumulative_returns[done_mask]

            # increment per-env episode counts
            episode_counts[done_mask] += 1

            # reset cumulative returns for envs that finished
            cumulative_returns[done_mask] = 0.0

            # compute mean episodic return over all finished episodes so far
            finished_episode_count = episode_counts.sum()
            if finished_episode_count > 0:
                mean_return = total_cumulative_returns.sum() / finished_episode_count
            else:
                mean_return = 0
            # handle final observation when truncation happened
            real_next_obs = next_obs.copy()

            # add to replay buffer
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # update observation
            obs = next_obs

            # training step
            if global_step > self.algo.learning_starts:
                data = self.rb.sample(self.algo.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations.to(device)
                    )
                    qf1_next_target = self.qf1_target(
                        data.next_observations.to(device), next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data.next_observations.to(device), next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten().to(device) + (
                        1 - data.dones.flatten().to(device)
                    ) * self.algo.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(
                    data.observations.to(device), data.actions.to(device)
                ).view(-1)
                qf2_a_values = self.qf2(
                    data.observations.to(device), data.actions.to(device)
                ).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize Q
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                # delayed policy updates
                if global_step % self.algo.policy_frequency == 0:
                    # perform policy update(s)
                    for _ in range(self.algo.policy_frequency):
                        pi, log_pi, _ = self.actor.get_action(
                            data.observations.to(device)
                        )
                        qf1_pi = self.qf1(data.observations.to(device), pi)
                        qf2_pi = self.qf2(data.observations.to(device), pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # automatic alpha tuning
                        if self.algo.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(
                                    data.observations.to(device)
                                )
                            alpha_loss = (
                                -self.log_alpha.exp() * (log_pi + self.target_entropy)
                            ).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # soft update target networks
                if global_step % self.env.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.env.tau * param.data
                            + (1 - self.env.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.env.tau * param.data
                            + (1 - self.env.tau) * target_param.data
                        )

                # logging
                if global_step % 1000 == 0 and finished_episode_count > 0:
                    self.writer.add_scalar(
                        "charts/mean_episodic_return", mean_return, global_step
                    )
                    self.writer.add_scalar(
                        "charts/steps", global_step * num_envs, global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    self.writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
        self.envs.close()


if __name__ == "__main__":
    config = tyro.cli(Config)
    sac = SAC(config)
