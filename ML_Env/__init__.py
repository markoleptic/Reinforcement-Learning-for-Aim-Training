from gymnasium.envs.registration import register

register(
    id='ML_Env/ML_RL_Env-v0',
    entry_point='ML_Env.envs:ML_RL_Env',
)