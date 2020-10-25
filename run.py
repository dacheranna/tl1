import time

import gym
from pygame.examples.vgrade import timer
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

import tl1

env = gym.make("tl1-v0")
obs = env.reset()
env.render()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1)
    print("reward: ", reward)
    if done:
        obs = env.reset()
        break
