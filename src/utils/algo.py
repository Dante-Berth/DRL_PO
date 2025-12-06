from dataclasses import asdict
import time
import copy
import gymnasium as gym
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb


class AlgoRL:
    def __init__(self, config):
        """
        Base class for RL algorithms.
        config: dataclass with .train, .env, .algo attributes
        """
        self.train = config.train
        self.env = config.env
        self.algo = config.algo

        self.run_name = (
            f"{self.train.env_id}__{self.train.exp_name}__"
            f"{self.train.seed}__{int(time.time())}"
        )

        # WandB logging
        if self.train.track:
            wandb_config = {
                "train": asdict(self.train),
                "algo": asdict(self.algo),
                "env": asdict(self.env),
            }
            wandb.init(
                project=self.train.wandb_project_name,
                entity=self.train.wandb_entity,
                sync_tensorboard=True,
                config=wandb_config,
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        # TensorBoard
        self.writer = SummaryWriter(f"runs/{self.run_name}")

        # show hyperparams in TB
        all_params = {
            **{f"train.{k}": v for k, v in asdict(self.train).items()},
            **{f"algo.{k}": v for k, v in asdict(self.algo).items()},
            **{f"env.{k}": v for k, v in asdict(self.env).items()},
        }
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in all_params.items()])),
        )

        # set seeds
        self.seeding()

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.train.cuda else "cpu"
        )

        # environment(s)
        env_kwargs = asdict(self.env)
        self.envs = gym.vector.SyncVectorEnv(
            [
                self.make_env(self.train.env_id, env_kwargs)
                for _ in range(self.train.num_envs)
            ]
        )

        # ensure dtype consistency
        try:
            self.envs.single_observation_space.dtype = np.float32
        except Exception:
            pass

        # algorithm-specific initialization
        self.init_algo_rl()

    def seeding(self):
        random.seed(self.train.seed)
        np.random.seed(self.train.seed)
        torch.manual_seed(self.train.seed)
        torch.backends.cudnn.deterministic = self.train.torch_deterministic

    def init_algo_rl(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def run_training(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    @staticmethod
    def make_env(env_id: str, env_kwargs: dict):
        """Default factory for vectorized envs"""

        def thunk():
            env = gym.make(env_id, **env_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    # -----------------------------
    # WandB Sweep Integration
    # -----------------------------
    def tune(self, sweep_config: dict, count: int = 1):
        """
        Run a hyperparameter sweep using WandB
        sweep_config: dict defining WandB sweep (method, parameters, etc.)
        count: number of runs for the sweep
        """

        def _run_sweep():
            # Initialize AlgoRL instance for each sweep run
            # Merge config from wandb.agent
            run_config = copy.deepcopy(asdict(self.train))
            run_config.update(wandb.config)
            # update config for this run
            self.train = self.train.__class__(**run_config)

            # Re-initialize environments, device, seeding
            self.seeding()
            env_kwargs = asdict(self.env)
            self.envs = gym.vector.SyncVectorEnv(
                [
                    self.make_env(self.train.env_id, env_kwargs)
                    for _ in range(self.train.num_envs)
                ]
            )
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.train.cuda else "cpu"
            )

            # Run training loop
            self.run_training()

        wandb.login()
        sweep_id = wandb.sweep(
            sweep_config,
            project=self.train.wandb_project_name,
            entity=self.train.wandb_entity,
        )

        wandb.agent(sweep_id, function=_run_sweep, count=count)
