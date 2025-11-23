# jax_ou_env.py
import jax
from jax import jit, lax, random, vmap
import jax.numpy as jnp
from flax import struct
from jax import lax
from typing import Optional
from types import SimpleNamespace

# Enable 64-bit if you prefer (optional)
# jax.config.update("jax_enable_x64", True)


def build_ou_process(key: jax.Array, T: int, theta: float, sigma: float):
    """
    Discrete OU: x_t = x_{t-1} - theta * x_{t-1} + sigma * eps_t
    returns array shape (T,)
    """
    (eps_key,) = jax.random.split(key, 1)
    eps = jax.random.normal(eps_key, shape=(T,), dtype=jnp.float32)

    def body(carry, eps_t):
        x = carry
        x_next = x - theta * x + sigma * eps_t
        return x_next, x_next

    init = jnp.array(0.0, dtype=jnp.float32)
    _, xs = jax.lax.scan(body, init, eps)
    # normalize similarly to original: X /= sigma * sqrt(1/(2*theta))
    denom = sigma * jnp.sqrt(1.0 / (2.0 * theta))
    xs = xs / denom
    return xs


@struct.dataclass
class EnvState:
    key: jax.Array  # PRNGKey
    signal: jnp.ndarray  # shape (T,)
    it: jnp.int32  # time index (scalar)
    pi: jnp.float32  # current position
    p: jnp.float32  # next signal value (signal[it+1])
    state: jnp.ndarray  # (2,) or any shape representing state e.g. (p, pi)
    done: jnp.bool_  # scalar
    scale_reward: jnp.float32
    # bookkeeping fields for returns/noise if desired
    noise_key: Optional[jax.Array] = None


class OUEnv:
    """
    JAX version of your NumPy Environment for OU signal trading.
    - Use env.reset(key) -> EnvState
    - Use env.step(state, action) -> new EnvState, reward
    """

    def __init__(
        self,
        sigma: float = 0.5,
        theta: float = 1.0,
        T: int = 10000,
        random_state: Optional[int] = None,
        lambd: float = 0.5,
        psi: float = 0.5,
        cost: str = "trade_0",
        max_pos: float = 10.0,
        squared_risk: jnp.bool_ = jnp.bool_(True),
        penalty: str = "none",
        alpha: float = 10.0,
        beta: float = 10.0,
        clip: jnp.bool_ = jnp.bool_(True),
        noise: jnp.bool_ = jnp.bool_(False),
        noise_std: float = 1.0,
        noise_seed=None,
        scale_reward: float = 10.0,
    ):
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.T = int(T)
        self.lambd = float(lambd)
        self.psi = float(psi)
        assert cost in ("trade_0", "trade_l1", "trade_l2")
        self.cost = cost
        self.max_pos = float(max_pos)
        self.squared_risk = jnp.bool_(squared_risk)
        assert penalty in ("none", "constant", "tanh", "exp")
        self.penalty = penalty
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.clip = jnp.bool_(clip)
        self.noise = jnp.bool_(noise)
        self.noise_std = float(noise_std)
        self.noise_seed = noise_seed
        self.scale_reward = float(scale_reward)
        # Map string to integer index
        self.cost_idx = {"none": 0, "trade_0": 1, "trade_l1": 2, "trade_l2": 3}[
            self.cost
        ]
        self.penalty_idx = {"none": 0, "constant": 1, "tanh": 2, "exp": 3}[self.penalty]
        # optional fixed RNG seed for environment generation (not used by default)
        self._fixed_key = (
            None if random_state is None else jax.random.PRNGKey(random_state)
        )
        # if fixed_key provided, we use it to build deterministic initial signal
        # otherwise each reset should receive a key from the caller.
        self.action_space = SimpleNamespace(
            low=-self.max_pos, high=self.max_pos, shape=(1,), dtype=jnp.float32
        )
        self.observation_space = SimpleNamespace(
            low=-jnp.inf, high=jnp.inf, shape=(2,), dtype=jnp.float32
        )

    def _make_signal(self, key: jax.Array):
        # Use fixed_key if provided, otherwise split from provided key
        if self._fixed_key is None:
            key, sub = jax.random.split(key)
            signal = build_ou_process(sub, self.T, self.theta, self.sigma)
            return key, signal
        else:
            # deterministic signal based on constructor seed
            signal = build_ou_process(self._fixed_key, self.T, self.theta, self.sigma)
            return key, signal

    def reset(self, key: jax.Array) -> EnvState:
        """
        Reset environment. Caller owns PRNG key; this function splits it and returns
        an EnvState containing a new 'key' for future sampling.
        """
        # split
        key, subkey = jax.random.split(key)
        key_for_noise, key_for_signal = jax.random.split(subkey)

        key, signal = self._make_signal(key_for_signal)

        # initial iteration = 0 (we access signal[it+1] so T must be >=2)
        it = jnp.int32(0)
        pi = jnp.array(0.0, dtype=jnp.float32)
        p = signal[it + 1]
        state = jnp.array([p, pi], dtype=jnp.float32)
        done = jnp.bool_(False)
        noise_key = key_for_noise if self.noise else None

        return EnvState(
            key=key,
            signal=signal,
            it=it,
            pi=pi,
            p=p,
            state=state,
            done=done,
            scale_reward=jnp.array(self.scale_reward, dtype=jnp.float32),
            noise_key=noise_key,
        )

    def _cost_fn(self, pi_next, pen, risk_term, p_next, noise_t, pi, scale_reward):
        def trade_0(_):
            return (p_next * pi_next - risk_term - pen) / scale_reward

        def trade_l1(_):
            trade_cost = self.psi * jnp.abs(pi_next - pi)
            return (
                (p_next + noise_t) * pi_next - risk_term - trade_cost - pen
            ) / scale_reward

        def trade_l2(_):
            trade_cost = self.psi * (pi_next - pi) ** 2
            return (
                (p_next + noise_t) * pi_next - risk_term - trade_cost - pen
            ) / scale_reward

        def default(_):
            return 0.0

        return lax.switch(
            self.cost_idx, [trade_0, trade_l1, trade_l2, default], operand=None
        )

    def _pen_fn(self, pi_abs):
        def penalty_none(_):
            return 0.0

        def penalty_const(_):
            return jnp.where(pi_abs > self.max_pos, self.alpha, 0.0)

        def penalty_tanh(_):
            return self.beta * (
                jnp.tanh(self.alpha * (pi_abs - 5 * self.max_pos / 4)) + 1.0
            )

        def penalty_exp(_):
            return self.beta * jnp.exp(self.alpha * (pi_abs - self.max_pos))

        return lax.switch(
            self.penalty_idx,
            [penalty_none, penalty_const, penalty_tanh, penalty_exp],
            operand=None,
        )

    def step(self, env_state: EnvState, action: jnp.ndarray):
        """Fully JAX-pure, scan-optimizable version."""

        # --- Handle done via JAX (NO Python branch) ---
        # Reset must be functional & JAX-friendly
        reset_state = self.reset(env_state.key)
        s = lax.cond(
            env_state.done, lambda _: reset_state, lambda _: env_state, operand=None
        )

        # local bindings
        key = s.key
        it = s.it
        pi = s.pi
        signal = s.signal
        scale_reward = s.scale_reward

        # --- Clip position (static boolean) ---
        pi_next_unclipped = pi + action
        pi_next = lax.select(
            self.clip,
            jnp.clip(pi_next_unclipped, -self.max_pos, self.max_pos),
            pi_next_unclipped,
        )

        # --- Penalty function rewritten fully JAX ---
        pi_abs = jnp.abs(pi_next_unclipped)

        pen = self._pen_fn(pi_abs)

        # --- Noise sampling without Python branching ---
        key, noise_k = jax.random.split(key)
        noise_t = jax.random.normal(noise_k, ()) * self.noise_std
        noise_t = lax.select(self.noise, noise_t, 0.0)

        # --- Reward cost function (fully JAX) ---
        p_next = signal[it + 1]
        risk_term = self.lambd * (pi_next**2) * self.squared_risk

        reward = self._cost_fn(
            pi_next, pen, risk_term, p_next, noise_t, pi, scale_reward
        )

        # --- Next state ---
        it_next = it + 1
        done_next = it_next == (signal.shape[0] - 2)

        new_state = jnp.array([p_next, pi_next], dtype=jnp.float32)

        new_env_state = s.replace(
            key=key,
            it=it_next,
            pi=pi_next,
            p=p_next,
            state=new_state,
            done=done_next,
        )
        return new_env_state, reward

    def apply(self, state_tuple, thresh=1.0, lambd_override=None, psi_override=None):
        """
        Determine the trade according to analytic/myopic solution (mirrors original apply).
        state_tuple = (p, pi)
        returns trade = pi_next - pi (consistent with step expecting delta)
        """
        p, pi = state_tuple
        lambd = self.lambd if lambd_override is None else lambd_override
        psi = self.psi if psi_override is None else psi_override

        if not self.squared_risk:
            # band policy
            def case_none():
                return jnp.where(
                    jnp.abs(p) < thresh,
                    0.0,
                    jnp.where(p >= thresh, self.max_pos - pi, -self.max_pos - pi),
                )

            return case_none()
        else:
            if self.cost == "trade_0":
                return p / (2 * lambd) - pi
            elif self.cost == "trade_l2":
                return (p + 2 * psi * pi) / (2 * (lambd + psi)) - pi
            elif self.cost == "trade_l1":
                # piecewise linear band
                cond1 = p < (-psi + 2 * lambd * pi)
                cond2 = (p >= (-psi + 2 * lambd * pi)) & (p <= (psi + 2 * lambd * pi))
                out1 = (p + psi) / (2 * lambd) - pi
                out2 = 0.0
                out3 = (p - psi) / (2 * lambd) - pi
                return jnp.where(cond1, out1, jnp.where(cond2, out2, out3))
            else:
                return 0.0

    # Optionally implement test and test_apply similarly, but keep them pure-functional and JITable
    def test_apply(
        self,
        key: jax.Array,
        total_episodes: int = 10,
        thresh: float = 1.0,
        lambd=None,
        psi=None,
    ):
        """
        Run deterministic 'apply' policy over several episodes using provided random keys.
        Returns average cumulative reward.
        This is a simple implementation showing how to run multiple episodes.
        """

        def one_episode(k):
            # reset with key k
            env_state = self.reset(k)

            def body_fun(carry, _):
                s = carry
                trade = self.apply(
                    (s.p, s.pi), thresh=thresh, lambd_override=lambd, psi_override=psi
                )
                s, r = self.step(s, trade)
                return s, r

            # run until done using lax.while_loop or scan with max steps
            max_steps = self.T - 2

            def cond_fn(val):
                s, _ = val
                return ~s.done

            def body_while(val):
                s, _ = val
                s, r = self.step(
                    s,
                    self.apply(
                        (s.p, s.pi),
                        thresh=thresh,
                        lambd_override=lambd,
                        psi_override=psi,
                    ),
                )
                return (s, r)

            # simple loop: run for max_steps and accumulate rewards
            def scan_step(carry, _):
                s, rewards = carry
                trade = self.apply(
                    (s.p, s.pi), thresh=thresh, lambd_override=lambd, psi_override=psi
                )
                s, r = self.step(s, trade)
                rewards = rewards + r
                return (s, rewards), None

            # run fixed-length episode and then mask by done
            (s_final, total_reward), _ = lax.scan(
                scan_step, (env_state, 0.0), None, length=max_steps
            )
            return total_reward

        keys = jax.random.split(key, total_episodes)
        rewards = jax.vmap(one_episode)(keys)
        return jnp.mean(rewards), rewards


if __name__ == "__main__":
    import time
    from tqdm import tqdm

    T = 5000
    PSI = 1
    SIGMA = 0.1
    LAMBD = 0.3
    THETA = 0.1
    MAX_STEPS = int(1e6)
    VERBOSE = False
    begin_time = time.time()
    env = OUEnv(
        T=T, cost="trade_l2", noise=False, sigma=SIGMA, lambd=LAMBD, theta=THETA
    )

    def scan_step(carry, _):
        key, state = carry
        key, subk = jax.random.split(key)

        action = jax.random.uniform(
            subk,
            shape=env.action_space.shape,
            minval=env.action_space.low,
            maxval=env.action_space.high,
        )[0]

        new_state, reward = env.step(state, action)

        return (key, new_state), reward

    @jax.jit
    def rollout(key):
        state = env.reset(key)
        (_, final_state), rewards = lax.scan(
            scan_step, (key, state), None, length=MAX_STEPS
        )
        return final_state, rewards

    rollout = jax.jit(rollout)
    begin_time = time.time()
    final_state, rewards = rollout(jax.random.PRNGKey(0))
    print(f"Time taken for {MAX_STEPS} steps: {time.time() - begin_time} seconds")
