from dataclasses import dataclass, asdict
import time
import gymnasium as gym
import torch
import tyro
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class AlgoRl:
    def __init__(self, config):
        self.train = config.train
        self.env = config.env
        self.algo = config.algo
        run_name = (
            f"{config.train.env_id}__{config.train.exp_name}__"
            f"{config.train.seed}__{int(time.time())}"
        )
        if config.train.track:
            import wandb

            # flatten config for wandb
            wandb_config = {
                "train": asdict(self.train),
                "algo": asdict(self.env),
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
        self.envs = gym.vector.SyncVectorEnv(
            [
                self.make_env(self.train.env_id, env_kwargs)
                for _ in range(self.train.num_envs)
            ]
        )
        self.InitAlgoRL()

    def seeding(self):
        random.seed(self.train.seed)
        np.random.seed(self.train.seed)
        torch.manual_seed(self.train.seed)
        torch.backends.cudnn.deterministic = self.train.torch_deterministic

    def InitAlgoRL():
        raise "Implementation"

    @staticmethod
    def make_env(*args, **kwargs):
        raise " Implementation"
