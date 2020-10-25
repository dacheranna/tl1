import sys
from contextlib import closing

from gym import spaces, logger

import gym
import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class tl1_env(gym.Env):
    startPosition = [2, 0]

    done = False

    lastaction = None

    mymap = (
        ("O", "X", "O", "G"),
        ("O", "O", "O", "O"),
        ("S", "O", "X", "O")
    )

    def __init__(self, mm=mymap, startpos=None):
        if startpos is None:
            startpos = self.startPosition
        self.position = startpos
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([2, 3]), dtype=np.int)
        desc = mm
        self.desc = desc = np.asarray(desc, dtype='c')

    def makestep(self, a):
        col = self.position[1]
        row = self.position[0]
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, 2)
        elif a == RIGHT:
            col = min(col + 1, 3)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    def step(self, action):
        reward = 0
        newpos = self.makestep(action)
        self.position = newpos
        newletter = self.desc[self.position[0], self.position[1]]
        self.lastaction = action
        if newletter == b'X':
            reward = -5
        elif newletter == b'G':
            reward = 10
            self.done = True
        else:
            reward = 0
        return newpos, reward, self.done, {}

    def reset(self, pos=None):
        if pos is None:
            pos = self.startPosition
        self.done = False
        self.position = pos
        self.lastaction = None
        return self.position

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.position[0], self.position[1]
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "cyan", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
