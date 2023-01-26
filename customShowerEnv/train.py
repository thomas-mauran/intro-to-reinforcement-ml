from ShowerEnv import ShowerEnvClass

import os

from stable_baselines3 import PPO
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env = ShowerEnvClass()

episodes = 5

# Random method
# for episode in range (1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print ("Episode: {} Score: {}".format(episode, score))
# env.close()

log_path = os.path.join('files', 'Logs')
env = DummyVecEnv([lambda: env])

save_path = os.path.join('files', 'Saved_Models')

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

PPO_Path = os.path.join('files', 'Saved_Models', 'PPO')
model.save(PPO_Path)

