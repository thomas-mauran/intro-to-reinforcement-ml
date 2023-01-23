import os
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import time
A2C_Path = os.path.join("Training",  'breakout', 'Saved_Models', 'A2C')

env = make_atari_env('Breakout-v0', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=10)
# print(env.observation_space)
model = A2C.load(A2C_Path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     time.sleep(0.1)
#     env.render()

# env.close()

# episodes = 20

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