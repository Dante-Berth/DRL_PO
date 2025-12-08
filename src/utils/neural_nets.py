import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
from tensordict import from_modules
from copy import deepcopy

# official code from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        # combine_state_for_ensemble causes graph breaks
        self.params = from_modules(*modules, as_module=True)
        with self.params[0].data.to("meta").to_module(modules[0]):
            self.module = deepcopy(modules[0])
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return torch.vmap(self._call, (0, None), randomness="different")(
            self.params, *args, **kwargs
        )

    def __repr__(self):
        return f"Vectorized {len(self)}x " + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        x = F.pad(x, self.padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act)
        if act
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    for k in cfg.obs_shape.keys():
        if k == "state":
            out[k] = mlp(
                cfg.obs_shape[k][0] + cfg.task_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        elif k == "rgb":
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(
                f"Encoder for observation type {k} not implemented."
            )
    return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
    """
    Converts a checkpoint from our old API to the new torch.compile compatible API.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ["weight", "bias", "ln.weight", "ln.bias"]
    new_state_dict = dict()

    # rename keys
    for key, val in list(source_state_dict.items()):
        if key.startswith("_Qs."):
            num = key[len("_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_Qs.params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val
            new_total_key = "_detach_Qs_params." + new_key
            new_state_dict[new_total_key] = val
        elif key.startswith("_target_Qs."):
            num = key[len("_target_Qs.params.") :]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_target_Qs_params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val

    # add batch_size and device from target_state_dict to new_state_dict
    for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
        for key in ("__batch_size", "__device"):
            new_key = prefix + "params." + key
            new_state_dict[new_key] = target_state_dict[new_key]

    # check that every key in new_state_dict is in target_state_dict
    for key in new_state_dict.keys():
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    # check that all Qs keys in target_state_dict are in new_state_dict
    for key in target_state_dict.keys():
        if "Qs" in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    # check that source_state_dict contains no Qs keys
    for key in source_state_dict.keys():
        assert "Qs" not in key, f"key {key} contains 'Qs'"

    # copy log_std_min and log_std_max from target_state_dict to new_state_dict
    new_state_dict["log_std_min"] = target_state_dict["log_std_min"]
    new_state_dict["log_std_dif"] = target_state_dict["log_std_dif"]
    if "_action_masks" in target_state_dict:
        new_state_dict["_action_masks"] = target_state_dict["_action_masks"]

    # copy new_state_dict to source_state_dict
    source_state_dict.update(new_state_dict)

    return source_state_dict
# end official code from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py

# -----------------------
# Networks
# -----------------------
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape[1:]).prod())
        act_dim = int(np.prod(env.action_space.shape[1:]))
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorContinous(nn.Module):
    def __init__(self, env, log_std_max=2, log_std_min=-5):
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape[1:]).prod())
        act_dim = int(np.prod(env.action_space.shape[1:]))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high[0] - env.action_space.low[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high[0] + env.action_space.low[0]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape[1:]).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape[1:]).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape[1:])), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape[1:]))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


if __name__ == "__main__":
    import src.envs
    import gymnasium as gym

    env = gym.make("OUTradingEnv-v0", psi=2.5)
    obs, info = env.reset()
    actor = ActorContinous(env)
    critic = SoftQNetwork(env)
    print("✅​ src/utils/nn.py")
