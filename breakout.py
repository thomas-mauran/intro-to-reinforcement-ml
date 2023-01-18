import os
import gym

from random import *

from stable_baselines3 import A2C

# To vectorize and train on multiple environments at the time 
from stable_baselines3.common.vec_env import VecFrameStack


from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env


environment_name = "Breakout-v0"
env = gym.make(environment_name)

episodes = 20


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


log_path = os.path.join('Training', 'breakout', 'Logs')

env = DummyVecEnv([lambda: env])

save_path = os.path.join('Training', 'breakout', 'Saved_Models', 'A2C_Model_Breakout')

# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=10000)

# model.save(save_path)


A2C_Path = os.path.join("Training",  'breakout', 'Saved_Models', 'A2C_Model_Breakout')

model = A2C.load(A2C_Path)



for episode in range (1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print ("Episode: {} Score: {}".format(episode, score))
env.close()