from ShowerEnv import ShowerEnvClass


from stable_baselines3 import PPO
import os 
env = ShowerEnvClass()

PPO_Path = os.path.join('files', 'Saved_Models', 'PPO.zip')

model = PPO.load(PPO_Path, env=env)

episodes = 10

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