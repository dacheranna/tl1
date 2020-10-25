import time

import numpy as np
import gym
from pygame.examples.vgrade import timer
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import clear_output

import warnings
warnings.simplefilter('ignore')

#%matplotlib inline
import seaborn as sns
sns.set()

import tl1

np.random.seed(0)

env = gym.make("tl1-v0")

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

Q = np.zeros((12, 4))
alpha = 0.1 # learning rate
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.0001

train_episodes = 3000
test_episodes = 100
max_steps = 100

training_rewards = []
epsilons = []

for episode in range(train_episodes):
    state = env.reset()
    total_training_rewards = 0

    for step in range(100):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.random.choice(np.argmax(Q[state, :], axis=1))

        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
        total_training_rewards += reward
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)

print("Training score over time: " + str(sum(training_rewards) / train_episodes))

obs = env.reset()
env.render()
while True:
    action = np.random.choice(np.argmax(Q[obs, :], axis=1))
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    print("reward: ", reward)
    if done:
        obs = env.reset()
        break
