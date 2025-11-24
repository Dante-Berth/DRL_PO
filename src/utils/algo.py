from dataclasses import asdict
import time
import gymnasium as gym
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class AlgoRL:
    def __init__(self, config):
        self.train = config.train
        self.env = config.env
        self.algo = config.algo

        run_name = (
            f"{self.train.env_id}__{self.train.exp_name}__"
            f"{self.train.seed}__{int(time.time())}"
        )

        if self.train.track:
            import wandb

            # flatten config for wandb
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
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        self.writer = SummaryWriter(f"runs/{run_name}")

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
        self.seeding()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.train.cuda else "cpu"
        )

        env_kwargs = asdict(self.env)
        # use the factory defined below (can be overridden by subclass)
        self.envs = gym.vector.SyncVectorEnv(
            [
                self.make_env(self.train.env_id, env_kwargs)
                for _ in range(self.train.num_envs)
            ]
        )

        # ensure single-space dtypes are float32
        try:
            self.envs.single_observation_space.dtype = np.float32
        except Exception:
            pass

        # initialize algorithm-specific structures
        self.init_algo_rl()

    def seeding(self):
        random.seed(self.train.seed)
        np.random.seed(self.train.seed)
        torch.manual_seed(self.train.seed)
        torch.backends.cudnn.deterministic = self.train.torch_deterministic

    def init_algo_rl(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    @staticmethod
    def make_env(env_id: str, env_kwargs: dict):
        """Default factory — subclasses may override if needed"""

        def thunk():
            env = gym.make(env_id, **env_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk
