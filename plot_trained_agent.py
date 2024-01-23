import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from src.env.env import EvacuationEnv
from src.env import constants
from src import params
env = EvacuationEnv(number_of_pedestrians=constants.NUM_PEDESTRIANS, draw=True)

model = PPO(
            "MultiInputPolicy", 
            env, verbose=1
        )

model.load('saved_data/models/fuckyou')

import wandb

wandb.init()

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    #action = np.array(( + np.cos(i)/5 ,  + np.sin(i)/5 , 1 - 2*i/1000, 1-2*i/1000, -1+2*i/1000))[None, :]
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

env.save_animation()