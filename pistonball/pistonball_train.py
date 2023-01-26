from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO

from pettingzoo.butterfly import pistonball_v6
from pettingzoo.test import render_test
from pettingzoo.utils.conversions import aec_to_parallel; 

import supersuit as ss

env = pistonball_v6.env(n_pistons=20, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)

# here ss allows us to pass the output as a grayscale array instead of a 3 channel color one speeding the compute time by 3 
env = ss.color_reduction_v0(env, mode='B')

# We reduce the size to avoid processing useless pixels 
env = ss.resize_v1(env, x_size=84, y_size=84)

# We stack 3 frames at the time to speed up compute time, less frame = less things to analyse 
env = ss.frame_stack_v2(env, 3)

# We convert the api to do parameter sharing of the policy
env = aec_to_parallel(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)

# We want to run multiple version of the env in parrallel
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
model.learn(total_timesteps=2000000)
model.save('policy')



# for agent in env.agent_iter():
#     obs, reward, done, info = env.last()
#     action = policy(obs, agent)
#     env.step(action)
