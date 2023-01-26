# Gym imports
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Helpers
import numpy as np
import random


class ShowerEnvClass(Env):
    def __init__(self):
        # here the action :
        # more hot 
        # more cold
        # nothing
        self.action_space =  Discrete(3)

        # it's an array of 1 element which is between 0 to 100 degrees
        self.observation_space = Box(low=0, high=100, shape=(1,))

        # the temperature changes randomly 
        self.state = 38 + random.randint(-3, 3)

        # 60 second length shower
        self.shower_length = 60

    def step(self, action):
        # action = (0, 1, 2) - 1 = (-1, 0, 1)
        self.state += action -1

        # decrease shower time
        self.shower_length -= 1

        # calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
        
        if self.shower_length <= 0:
            done = True
        else:
            done = False
        
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # We reset the values
        self.shower_length = 60
        self.state = np.array([39+random.randint(-3, 3)]).astype(float)
        return self.state
        