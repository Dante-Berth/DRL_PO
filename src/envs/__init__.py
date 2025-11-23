from gymnasium.envs.registration import register

register(
    id="OUTradingEnv-v0",
    entry_point="src.envs.original_env:GymOUTradingEnv",
)
