import os
import gym

from random import *

from stable_baselines3 import A2C

# To vectorize and train on multiple environments at the time 
from stable_baselines3.common.vec_env import VecFrameStack


from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.env_util import make_atari_env


environment_name = "Breakout-v0"
env = make_atari_env(environment_name, n_envs=10, seed=0)
env = VecFrameStack(env, n_stack=10)
save_model = os.path.join('files', 'Saved_Models', 'A2C')


# log_path = os.path.join('files', 'Logs')
# #Cnn policy = good for images output training 
# model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=1000000)
# model.save(save_model)

# episodes = 20

# Random method
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()


# log_path = os.path.join('files',  'Logs')


# save_path = os.path.join('files', 'Saved_Models', 'A2C_Model_Breakout')

# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=10000)

# model.save(save_path)
