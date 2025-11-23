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
    total_timesteps: int = 1_000_000
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


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    config = tyro.cli(Config)

    # Build run name
    run_name = (
        f"{config.train.env_id}__{config.train.exp_name}__"
        f"{config.train.seed}__{int(time.time())}"
    )

    # Optional: wandb init
    if config.train.track:
        import wandb

        # flatten config for wandb
        wandb_config = {
            "train": asdict(config.train),
            "sac": asdict(config.sac),
            "env": asdict(config.env),
        }

        wandb.init(
            project=config.train.wandb_project_name,
            entity=config.train.wandb_entity,
            sync_tensorboard=True,
            config=wandb_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    # show hyperparams in TB
    all_params = {
        **{f"train.{k}": v for k, v in asdict(config.train).items()},
        **{f"sac.{k}": v for k, v in asdict(config.sac).items()},
        **{f"env.{k}": v for k, v in asdict(config.env).items()},
    }
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in all_params.items()])),
    )

    # -----------------------
    # Seeding
    # -----------------------
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.backends.cudnn.deterministic = config.train.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.train.cuda else "cpu"
    )

    # -----------------------
    # Environment setup
    # -----------------------
    # pass env kwargs as plain dict so gym.make(env_id, **kwargs) works with env signature
    env_kwargs = asdict(config.env)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(config.train.env_id, env_kwargs)
            for _ in range(config.train.num_envs)
        ]
    )

    # -----------------------
    # Networks & optimizers
    # -----------------------
    actor = ActorContinous(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=config.sac.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config.sac.policy_lr)

    # Automatic entropy tuning
    if config.sac.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=config.sac.q_lr)
    else:
        alpha = config.sac.alpha
        log_alpha = None
        a_optimizer = None
        target_entropy = None

    envs.single_observation_space.dtype = np.float32

    rb = ReplayBuffer(
        config.sac.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=config.train.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # -----------------------
    # Start the training loop
    # -----------------------
    obs, _ = envs.reset(seed=config.train.seed)
    for global_step in range(config.sac.total_timesteps):
        # action selection
        if global_step < config.sac.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions_t, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions_t.detach().cpu().numpy()

        # step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # logging episodic returns if present
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    break

        # handle final observation when truncation happened
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]

        # add to replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # update observation
        obs = next_obs

        # training step
        if global_step > config.sac.learning_starts:
            data = rb.sample(config.sac.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations.to(device)
                )
                qf1_next_target = qf1_target(
                    data.next_observations.to(device), next_state_actions
                )
                qf2_next_target = qf2_target(
                    data.next_observations.to(device), next_state_actions
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten().to(device) + (
                    1 - data.dones.flatten().to(device)
                ) * config.sac.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(
                data.observations.to(device), data.actions.to(device)
            ).view(-1)
            qf2_a_values = qf2(
                data.observations.to(device), data.actions.to(device)
            ).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize Q
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # delayed policy updates
            if global_step % config.sac.policy_frequency == 0:
                # perform policy update(s)
                for _ in range(config.sac.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations.to(device))
                    qf1_pi = qf1(data.observations.to(device), pi)
                    qf2_pi = qf2(data.observations.to(device), pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # automatic alpha tuning
                    if config.sac.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(
                                data.observations.to(device)
                            )
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # soft update target networks
            if global_step % config.sac.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        config.sac.tau * param.data
                        + (1 - config.sac.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        config.sac.tau * param.data
                        + (1 - config.sac.tau) * target_param.data
                    )

            # logging
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    envs.close()
    writer.close()
