from gym.envs.registration import register

register(id='tl1-v0',
         entry_point='tl1.envs:tl1_env',
         )
