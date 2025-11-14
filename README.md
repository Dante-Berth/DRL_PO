# Does Deep Reinforcement Learning algorithm solve Portfolion Optimisation ?
The [paper](https://arxiv.org/pdf/2003.06497) has shown that using Deep Reinforcement Learning (DRL) algorithms in particular, the DDPO method can nearly find the optimal strategy for well-known portfolio optimisation problems.

In this work, we aim to apply different DRL algorithms, such as [Soft Actor-Critic (SAC)](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) and [Proximal Policy Optimisation (PPO)](https://arxiv.org/pdf/1707.06347), to investigate whether certain types of DRL algorithms are better suited for portfolio optimisation tasks.

# Tasks
- [ ] Create a parent env
- [ ] From the parent env, code the three child envs
- [ ] Add wrapper to vectorize the three child envs
- [ ] Code in Jax SAC and PPO
- [ ] Choose their best hyperparameters
- [ ] Compare them
