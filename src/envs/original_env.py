import numpy as np
from types import SimpleNamespace
import gymnasium as gym
from gymnasium import spaces

# Code from https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization
# Added gymnasium


def build_ou_process(T=100000, theta=0.1, sigma=0.1, random_state=None):
    """
    Description
    ---------------
    Build a discrete OU process signal of length T starting at p_0=0:
    ```
        p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t;
    ```
    (epsilon_t)_t are standard normal random variables


    Parameters:
    ---------------
    T : Int, length of the signal.
    theta : Float>0, parameter of the OU process.
    sigma : Float>0, parameter of the OU process.
    random_state : None or Int, if Int, generate the same sequence of noise each time.

    Returns
    ---------------
    np.array of shape (T,), the OU signal generated.
    """
    X = np.empty(T)
    t = 0
    x = 0.0
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        normals = rng.normal(0, 1, T)

    else:
        normals = np.random.normal(0, 1, T)

    for t in range(T):
        x += -x * theta + sigma * normals[t]
        X[t] = x
    X /= sigma * np.sqrt(1.0 / 2.0 / theta)
    return X


def get_returns(signal, random_state=None):
    """
    Description
    ---------------
    Compute the returns r_t = p_t + eta_t, where p_t is the signal and eta_t is a Gaussian
    white noise.

    Parameters
    ---------------
    signal : 1D np.array, the signal computed as a sample path of an OU process.
    random_state : Int or None:
        - if None, do not use a random state (useful to simulate different paths each time
          running the simulation).
        - if Int, use a random state (useful to compare different experimental results).

    Returns
    ---------------
    1D np.array containing the returns
    """

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return signal + rng.normal(size=signal.size)

    else:
        return signal + np.random.normal(size=signal.size)


class Environment:
    """
    The environment consists of the following:
        - state  : (current_position, next_signal)
        - action : next_position.
        - reward : it depends on our model.
    The signal is built as an OU process.
    """

    def __init__(
        self,
        sigma=0.5,
        theta=1.0,
        T=1000,
        random_state=None,
        lambd=0.5,
        psi=0.5,
        cost="trade_0",
        max_pos=10,
        squared_risk=True,
        penalty="none",
        alpha=10,
        beta=10,
        clip=True,
        noise=False,
        noise_std=10,
        noise_seed=None,
        scale_reward=10,
    ):
        """
        Description
        ---------------
        Constructor of class Environment.

        Parameters & Attributes
        ---------------
        sigma          : Float, parameter of price predictor signal
                         p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t; (epsilon_t)_t
                         are standard normal random variables
        theta          : Float, parameter of price predictor signal
                         p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t; (epsilon_t)_t
                         are standard normal random variables
        T              : Float, time horizon.
        random_state   : Int or None:
                         - if None, do not use a random state (useful to simulate
                           different paths each time running the simulation).
                         - if Int, use a random state (useful to compare different
                           experimental results).
        lambd          : Float, penalty term of the position in the reward function.
        psi            : Float, penalty term of the trade magnitude in the reward
                         function.
        cost           : String in ['trade_0', 'trade_l1', 'trade_l2']
                          - 'trade_0'  : no trading cost.
                          - 'trade_l1' : squared trading cost.
                          - 'trade_l2' : linear trading cost.
        max_pos        : Float > 0, maximum allowed position.
        squared_risk   : Boolean, whether to use the squared risk term or not.
        penalty        : String in ['none', 'constant', 'tanh', 'exp'], the type of
                         penalty to add to penalize positions beyond maxpos.
                         (It is advised to use a tanh penalty in the maxpos setting).
        alpha          : Int, a parameter of the penalty function.
        beta           : Int, a parameter of the penalty function.
        clip           : Boolean, whether to clip positions beyond maxpos or not.
        noise          : Boolean, whether to consider noisy returns or returns equal to
                         predictor values.
        noise_std      : Float, standard deviation of the noise added to the returns.
        noise_seed     : Int, see used to produce the additive noise of the returns.
        scale_reward   : Float>0, parameter that scales the rewards.
        signal         : 1D np.array of shape (T,) containing the sampled OU process.
        it             : Int, the current time iteration.
        pi             : Float, the current position.
        p              : Float, the next value of the signal.
        state          : 2-tuple, the current state: (p, pi).
        done           : Boolean, whether the episode is over or not. Initialized to
                         False.
        state_size     : Int, state size.
        action_size    : Int, action size.

        Returns
        ---------------
        """

        self.sigma = sigma
        self.theta = theta
        self.T = T
        self.lambd = lambd
        self.psi = psi
        self.cost = cost
        self.max_pos = max_pos
        self.squared_risk = squared_risk
        self.random_state = random_state
        self.signal = build_ou_process(T, sigma, theta, random_state)
        self.it = 0  # First iteration is 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.done = False
        self.state_size = len(self.state)
        self.action_size = 1
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.clip = clip
        self.scale_reward = scale_reward
        self.noise = noise
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.action_space = SimpleNamespace(
            low=-self.max_pos, high=self.max_pos, shape=(1,), dtype=np.float32
        )
        self.observation_space = SimpleNamespace(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

    def reset(self, random_state=None, noise_seed=None):
        """
        Description
        ---------------
        Reset the environment to run a new episode.

        Parameters
        ---------------
        random_state : Int or None:
            - if None, do not use a random state (useful to simulate different paths each
              time running the simulation).
            - if Int, use a random state (useful to compare different experimental
              results).
        noise_seed   : Same as random_state but for the noisy returns instead of the
                       predictor signal.

        Returns
        ---------------
        """

        self.signal = build_ou_process(self.T, self.sigma, self.theta, random_state)
        self.it = 0  # First iteration is 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.cumulative_pnl = 0
        self.done = False
        if self.noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, self.noise_std, self.T)

            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, self.noise_std, self.T)

    def env_step(self, action):
        """
        Description
        ---------------
        Aplly action to the environment to modify the state of the agent and get the
        corresponding reward.

        Parameters
        ---------------
        action : Float, the action to perform (next trade to make).

        Returns
        ---------------
        Float, the reward we get by applying action to the current state.
        """

        pi_next_unclipped = self.pi + action
        if self.clip:
            # Clip the next position between -max_pos and max_pos
            pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)

        else:
            pi_next = self.pi + action

        if self.penalty == "none":
            pen = 0

        elif self.penalty == "constant":
            pen = self.alpha * max(
                0,
                (self.max_pos - pi_next) / abs(self.max_pos - pi_next),
                (-self.max_pos - pi_next) / abs(-self.max_pos - pi_next),
            )

        elif self.penalty == "tanh":
            pen = self.beta * (
                np.tanh(self.alpha * (abs(pi_next_unclipped) - 5 * self.max_pos / 4))
                + 1
            )

        elif self.penalty == "exp":
            pen = self.beta * np.exp(self.alpha * (abs(pi_next) - self.max_pos))

        if self.cost == "trade_0":
            reward = (
                self.p * pi_next - self.lambd * pi_next**2 * self.squared_risk - pen
            ) / self.scale_reward

        elif self.cost == "trade_l1":
            if self.noise:
                reward = (
                    (self.p + self.noise_array[self.it]) * pi_next
                    - self.lambd * pi_next**2 * self.squared_risk
                    - self.psi * abs(pi_next - self.pi)
                    - pen
                ) / self.scale_reward

            else:
                reward = (
                    self.p * pi_next
                    - self.lambd * pi_next**2 * self.squared_risk
                    - self.psi * abs(pi_next - self.pi)
                    - pen
                ) / self.scale_reward

        elif self.cost == "trade_l2":
            if self.noise:
                reward = (
                    (self.p + self.noise_array[self.it]) * pi_next
                    - self.lambd * pi_next**2 * self.squared_risk
                    - self.psi * (pi_next - self.pi) ** 2
                    - pen
                ) / self.scale_reward

            else:
                reward = (
                    self.p * pi_next
                    - self.lambd * pi_next**2 * self.squared_risk
                    - self.psi * (pi_next - self.pi) ** 2
                    - pen
                ) / self.scale_reward

        self.pi = pi_next
        self.it += 1
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.done = self.it == (len(self.signal) - 2)  # terminated
        self.cumulative_pnl = reward + (self.lambd * self.pi**2) * self.squared_risk
        return reward

    def get_state(self):
        """
        Description
        ---------------
        Get the current state of the environment.

        Parameters
        ---------------

        Returns
        ---------------
        2-tuple representing the current state.
        """

        return self.state

    def test(
        self, agent, model, total_episodes=10, random_states=None, noise_seeds=None
    ):
        """
        Description
        ---------------
        Test a model on a number of simulated episodes and get the average cumulative
        reward.

        Parameters
        ---------------
        agent          : Agent object, the agent that loads the model.
        model          : Actor object, the actor network.
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
            - if None, do not use random state when generating episodes
              (useful to get an idea about the performance of a single model).
            - if List, generate episodes with the values in random_states (useful when
              comparing different models).

        noise_seeds    : None or List of length total_episodes:
                         - if None, do not use a random state when generating the additive
                           noise of the returns
                         - if List, generate noise with seeds in noise_seeds.

        Returns
        ---------------
        2-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        agent.actor_local = model
        if random_states is not None:
            assert total_episodes == len(random_states), (
                "random_states should be a list of length total_episodes!"
            )

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                _, _, action = agent.get_action(state)
                pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)
                episode_positions.append(pi_next)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi**2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = total_pnl
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #      'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )

    def apply(self, state, thresh=1, lambd=None, psi=None):
        """
        Description
        ---------------
        Apply solution with a certain band and slope outside the band, otherwise apply the
        myopic solution.

        Parameters
        ---------------
        state      : 2-tuple, the current state.
        thresh     : Float>0, price threshold to make a trade.
        lambd      : Float, slope of the solution in the non-banded region.
        psi        : Float, band width of the solution.

        Returns
        ---------------
        Float, the trade to make in state according to this function.
        """

        p, pi = state
        if lambd is None:
            lambd = self.lambd

        if psi is None:
            psi = self.psi

        if not self.squared_risk:
            if abs(p) < thresh:
                return 0
            elif p >= thresh:
                return self.max_pos - pi
            elif p <= -thresh:
                return -self.max_pos - pi

        else:
            if self.cost == "trade_0":
                return p / (2 * lambd) - pi

            elif self.cost == "trade_l2":
                return (p + 2 * psi * pi) / (2 * (lambd + psi)) - pi

            elif self.cost == "trade_l1":
                if p < -psi + 2 * lambd * pi:
                    return (p + psi) / (2 * lambd) - pi
                elif -psi + 2 * lambd * pi <= p <= psi + 2 * lambd * pi:
                    return 0
                elif p > psi + 2 * lambd * pi:
                    return (p - psi) / (2 * lambd) - pi

    def test_apply(
        self,
        total_episodes=10,
        random_states=None,
        thresh=1,
        lambd=None,
        psi=None,
        noise_seeds=None,
        max_point=6.0,
        n_points=1000,
    ):
        """
        Description
        ---------------
        Test a function with certain slope and band width for each reward model (with and
        without trading cost, and depending on the penalty when trading cost is used).
        When psi and lambd are not provided, use the myopic solution.

        Parameters
        ---------------
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
                         - if None, do not use random state when generating episodes
                           (useful to get an idea about the performance of a single
                           model).
                         - if List, generate episodes with the values in random_states
                           (useful when comparing different models).
        lambd          : Float, slope of the solution in the non-banded region.
        psi            : Float, band width of the solution.
        max_point      : Float, the maximum point in the grid [0, max_point]
        n_points       : Int, the number of points in the grid.

        Returns
        ---------------
        5-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
                  - Dict, cumulative sum of the reward at each time step per episode.
                  - Dict, pnl per episode.
                  - Dict, positions per episode.
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        if random_states is not None:
            assert total_episodes == len(random_states), (
                "random_states should be a list of length total_episodes!"
            )

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = self.apply(state, thresh=thresh, lambd=lambd, psi=psi)
                reward = self.env_step(action)
                pnl = reward + (self.lambd * self.pi**2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                episode_positions.append(state[1])
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = episode_pnls
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #       'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )

    def signals_results(self, pi, psi, thresh, lambd):
        import torch

        range_values = np.arange(-4, 4, 0.01)
        signal_zeros = torch.tensor(
            np.vstack((range_values, np.zeros(len(range_values)))).T,
            dtype=torch.float,
        )
        signal_ones_pos = torch.tensor(
            np.vstack((range_values, 0.5 * np.ones(len(range_values)))).T,
            dtype=torch.float,
        )
        signal_ones_neg = torch.tensor(
            np.vstack((range_values, -0.5 * np.ones(len(range_values)))).T,
            dtype=torch.float,
        )
        if psi is None:
            psi = self.psi
        if lambd is None:
            lambd = self.lambd

        if self.squared_risk:
            result1 = optimal_f_vec(
                signal_ones_neg[:, 0].numpy(),
                -pi,
                lambd=lambd,
                psi=psi,
                cost=self.cost,
            )
            result2 = optimal_f_vec(
                signal_zeros[:, 0].numpy(), 0, lambd=lambd, psi=psi, cost=env.cost
            )
            result3 = optimal_f_vec(
                signal_ones_pos[:, 0].numpy(),
                pi,
                lambd=lambd,
                psi=psi,
                cost=env.cost,
            )

        else:
            result1 = optimal_max_pos_vec(
                signal_ones_neg[:, 0].numpy(), -pi, thresh, env.max_pos
            )
            result2 = optimal_max_pos_vec(
                signal_zeros[:, 0].numpy(), 0, thresh, env.max_pos
            )
            result3 = optimal_max_pos_vec(
                signal_ones_pos[:, 0].numpy(), pi, thresh, env.max_pos
            )
        return signal_zeros, signal_ones_pos, signal_ones_neg, result1, result2, result3


class GymOUTradingEnv(Environment, gym.Env):
    def __init__(self, render_mode=None, **kwargs):
        Environment.__init__(self, **kwargs)

        # Replace SimpleNamespace with proper Gym spaces
        self.action_space = spaces.Box(
            low=-self.max_pos, high=self.max_pos, shape=(1,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

    def reset(self, random_state=None, noise_seed=None, *, seed=None, options=None):
        super().reset(random_state=random_state, noise_seed=noise_seed)
        obs = np.array(self.state, dtype=np.float32)
        info = {"cumulative_pnl": self.cumulative_pnl}
        return obs, info

    def step(self, action):
        reward = super().env_step(float(action))  # ensure scalar

        obs = np.array(self.state, dtype=np.float32)
        terminated = self.done  # episode ends naturally
        truncated = False  # no time truncation unless you add one
        info = {"cumulative_pnl": self.cumulative_pnl}

        return obs, reward, terminated, truncated, info


# Code from https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/master/agent.py#L44C1-L112C2


def optimal_f(p, pi, lambd=0.5, psi=0.3, cost="trade_l2"):
    """
    Description
    --------------
    Function with the shape of the optimal solution for cost models with 0, l2 and l1
    trading costs.

    Parameters
    --------------
    p     : Float, the next signal value.
    pi    : Float, the current position.
    lambd : Float > 0, Parameter of the cost model.
    psi   : Float > 0, Parameter of our model defining the trading cost.
    cost  : String in ['none', 'trade_l1', 'trade_l2'], cost model.

    Returns
    --------------
    Float, The function evaluation (which is the next trade).
    """

    if cost == "trade_0":
        return p / (2 * lambd) - pi

    elif cost == "trade_l2":
        return p / (2 * (lambd + psi)) + psi * pi / (lambd + psi) - pi

    elif cost == "trade_l1":
        if p <= -psi + 2 * lambd * pi:
            return (p + psi) / (2 * lambd) - pi

        elif -psi + 2 * lambd * pi < p < psi + 2 * lambd * pi:
            return 0

        elif p >= psi + 2 * lambd * pi:
            return (p - psi) / (2 * lambd) - pi


def optimal_max_pos(p, pi, thresh, max_pos):
    """
    Description
    --------------
    Function with the shape of the optimal solution for MaxPos cost model with l1 trading
    cost.

    Parameters
    --------------
    p       : Float, the next signal value.
    pi      : Float, the current position.
    thresh  : Float > 0, threshold of the solution in the infinite horizon case.
    max_pos : Float > 0, maximum allowed position.

    Returns
    --------------
    Float, The function evaluation (which is the next trade).
    """

    if abs(p) < thresh:
        return 0
    elif p >= thresh:
        return max_pos - pi
    elif p <= -thresh:
        return -max_pos - pi


# Vectorizing.
optimal_f_vec = np.vectorize(optimal_f, excluded=set(["pi", "lambd", "psi", "cost"]))
optimal_max_pos_vec = np.vectorize(
    optimal_max_pos, excluded=set(["pi", "thresh", "max_pos"])
)

if __name__ == "__main__":
    import random
    from tqdm import tqdm
    import time
    import src.envs  # this executes the register() in __init__
    import gymnasium as gym
    import multiprocessing as mp

    T = 5000
    PSI = 1
    SIGMA = 0.1
    LAMBD = 0.3
    THETA = 0.1
    MAX_STEPS = int(1e5)
    VERBOSE = False
    begin_time = time.time()
    env = Environment(
        T=T, cost="trade_l2", noise=False, sigma=SIGMA, lambd=LAMBD, theta=THETA
    )
    seed = random.randint(1, 100)
    state = env.reset(random_state=seed)
    total_reward = 0.0
    for step_i in tqdm(range(MAX_STEPS)):
        action = np.random.uniform(
            low=env.action_space.low,
            high=env.action_space.high,
            size=env.action_space.shape,
        )[0]

        # Step (env.step will manage its own internal RNG evolution)
        reward = env.env_step(action)
        total_reward += reward
        step_i += 1
        if VERBOSE:
            print(
                f"Step {step_i:02d} | action={float(action):+.3f} | "
                f"reward={float(reward):+.4f}"
            )

        if env.done:
            seed = random.randint(1, 100)
            if VERBOSE:
                print(f"\nTotal cumulative reward: {float(total_reward):.4f}")
            state = env.reset(random_state=seed)
            total_reward = 0.0
    print(f"Time taken for {MAX_STEPS} steps: {time.time() - begin_time} seconds")

    SIGMA = 0.1
    THETA = 0.1
    T = 5000
    LAMBD = 0.3
    PSI = 4
    env = gym.make(
        "OUTradingEnv-v0",
        sigma=SIGMA,
        theta=THETA,
        T=T,
        lambd=LAMBD,
        psi=PSI,
        scale_reward=1,
        cost="trade_l1",
    )

    random_state = 256
    n_episodes = 10
    rng = np.random.RandomState(random_state)
    random_states = rng.randint(0, int(1e6), size=n_episodes)

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    lambds = np.linspace(0.2, 0.6, 10)
    psis = np.linspace(0.8, 1.2, 10)
    grid = [
        (i, j, lambd, psi)
        for i, lambd in enumerate(lambds)
        for j, psi in enumerate(psis)
    ]

    def worker(args):
        i, j, lambd, psi = args
        score, score_episode, _, _, _ = env.unwrapped.test_apply(
            total_episodes=n_episodes,
            random_states=random_states,
            lambd=lambd,
            psi=psi,
        )
        # Convert score_episode dict → vector
        return i, j, score, np.array(list(score_episode.values()))

    # Run in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(worker, grid), total=len(grid)))

    # Allocate arrays
    scores = np.empty((len(lambds), len(psis)))
    scores_episodes = np.empty((len(lambds), len(psis), n_episodes))

    # Fill them
    for i, j, score, score_ep_vec in results:
        scores[i, j] = score
        scores_episodes[i, j, :] = score_ep_vec
        # print('lambd=%.1f , psi=%.1f -> score=%.3f \n' % (lambd, psi, score))

    # +
    i_max = np.argmax(scores) // scores.shape[0]
    j_max = np.argmax(scores[i_max, :])

    lambd_max, psi_max = lambds[i_max], psis[j_max]
    print("lambd_max=%.2f , psi_max=%.2f" % (lambd_max, psi_max))
    print("✅​ src/envs/original_env.py")
