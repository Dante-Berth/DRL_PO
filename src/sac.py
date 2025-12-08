# sac code
import os
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils.buffers import ReplayBuffer
import src.envs  # needed for gym.make
from src.utils.neural_nets import SoftQNetwork, ActorContinous
from src.utils.algo import AlgoRL


# -----------------------
# Config dataclasses
# -----------------------
@dataclass
class OUEnvConfig:
    sigma: float = 0.1
    theta: float = 0.1
    T: int = 5000
    random_state: Optional[int] = None
    lambd: float = 0.3
    psi: float = 4
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
    batch_size: int = 512
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
    hyp_tune: bool = False


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    algo: SACConfig = field(default_factory=SACConfig)
    env: OUEnvConfig = field(default_factory=OUEnvConfig)


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

        # ReplayBuffer expects single_observation_space & single_action_space
        self.rb = ReplayBuffer(
            self.algo.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            n_envs=self.train.num_envs,
            handle_timeout_termination=False,
        )

    def init_algo_rl(self):
        device = self.device
        # actors / q-nets
        self.actor = ActorContinous(self.envs).to(device)
        self.qf1 = SoftQNetwork(self.envs).to(device)
        self.qf2 = SoftQNetwork(self.envs).to(device)
        self.qf1_target = SoftQNetwork(self.envs).to(device)
        self.qf2_target = SoftQNetwork(self.envs).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # optimizers use algo hyperparameters
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=self.algo.q_lr,
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.algo.policy_lr
        )

        # Automatic entropy tuning
        if self.algo.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.envs.single_action_space.shape).to(device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.algo.q_lr)
        else:
            self.log_alpha = None
            self.alpha = self.algo.alpha
            self.a_optimizer = None
            self.target_entropy = None

    # you may remove SAC.make_env if you want to use the base factory

    def run_training(self):
        num_envs = self.train.num_envs
        device = self.device
        obs, _ = self.envs.reset(seed=self.train.seed)
        from tqdm import tqdm
        import time

        start_time = time.time()
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        self.writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_pnl", info["cumulative_pnl"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
                        break

            # handle final observation when truncation happened
            real_next_obs = next_obs.copy()

            # add to replay buffer
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, {})

            # update observation
            obs = next_obs

            # training step
            if global_step >= self.algo.learning_starts:
                data = self.rb.sample(self.algo.batch_size)
                # move tensors to device (assuming data returns tensors or numpy arrays)
                obs_batch = data.observations.to(device)
                actions_batch = data.actions.to(device)
                next_obs_batch = data.next_observations.to(device)
                rewards_batch = data.rewards.to(device).flatten()
                dones_batch = data.dones.to(device).flatten()

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        next_obs_batch
                    )
                    qf1_next_target = self.qf1_target(
                        next_obs_batch, next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        next_obs_batch, next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
                    next_q_value = rewards_batch + (
                        1 - dones_batch
                    ) * self.algo.gamma * min_qf_next_target.view(-1)

                qf1_a_values = self.qf1(obs_batch, actions_batch).view(-1)
                qf2_a_values = self.qf2(obs_batch, actions_batch).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize Q
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                # delayed policy updates
                if global_step % self.algo.policy_frequency == 0:
                    pi, log_pi, _ = self.actor.get_action(obs_batch)
                    qf1_pi = self.qf1(obs_batch, pi)
                    qf2_pi = self.qf2(obs_batch, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # automatic alpha tuning
                    if self.algo.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(obs_batch)
                        alpha_loss = (
                            -self.log_alpha.exp() * (log_pi + self.target_entropy)
                        ).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = float(self.log_alpha.exp().item())

                # soft update target networks
                if global_step % self.algo.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.algo.tau * param.data
                            + (1 - self.algo.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.algo.tau * param.data
                            + (1 - self.algo.tau) * target_param.data
                        )

                # logging
                if global_step % 1000 == 0:
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
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

        # close envs & writer moved to caller or destructor
        self.envs.close()


if __name__ == "__main__":
    config = tyro.cli(Config)
    sac = SAC(config)

    if config.train.track and getattr(config.train, "hyp_tune", False):
        sweep_cfg = {
            "method": "grid",
            "parameters": {
                "algo.alpha": {"values": [0.1, 0.2, 0.3]},
                "algo.policy_lr": {"values": [3e-4, 1e-3, 1e-2]},
                "algo.q_lr": {"values": [3e-4, 1e-3, 1e-2]},
                "env.sigma": {"values": [0.2, 0.3]},
                "algo.batch_size": {"values": [64, 128, 256, 512]},
                "algo.policy_frequency": {"values": [1, 2, 3]},
                "algo.target_network_frequency": {"values": [1, 2, 3]},
            },
        }
        sac.tune(sweep_cfg, count=config.train.SWEEP_COUNT)
    else:
        sac.run_training()
