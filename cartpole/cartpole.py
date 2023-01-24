import os
import gym


from stable_baselines3 import PPO
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

environment_name = "CartPole-v0"

# env = gym.make(environment_name)
 
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

env = gym.make(environment_name)

# Wrap the env in the vec environment 
env = DummyVecEnv([lambda: env])
 

# Training callback = way to stop the trzining if the reward is maxed before the end for example
save_path = os.path.join('files', 'Saved_Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

# Every 10.000 steps it will save the best modle and check if we haven't passed the best reward callback
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback,best_model_save_path=save_path, eval_freq=10000, verbose=1)

# Verbose 1 = we want logs
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000, callback=eval_callback)

PPO_Path = os.path.join("files", 'Saved_Models', 'PPO_Model_Cartpole')
# model.save(PPO_Path)

# To load tyhe model
# model = PPO.load(PPO_Path, env=env)

# Here we can have a sneak peak of the current 
# evaluate_policy(model, env, n_eval_episodes=2, render=True)

# NWe are now using the trained model to the env 
# for episode in range (1, episodes + 1):
#     obs = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         score += reward
#     print ("Episode: {} Score: {}".format(episode, score))
# env.close()

