from gym.envs.registration import register

register(
    id="monopoly",
    entry_point="gym_example.envs:MonopolyEnv"
)

register(
    id="monopoly2",
    entry_point="gym_example.envs:MonopolyEnv2"
)