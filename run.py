import time

import gym


import tl1


def random_agent(episodes=100):
    env = gym.make("tl1-v0")
    env.reset()
    env.render()
    for e in range(episodes):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        print(reward)
        #time.sleep(1)
        if done:
            break


if __name__ == "__main__":
    random_agent()
