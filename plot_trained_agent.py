import gymnasium as gym

from stable_baselines3 import PPO

from src.env.env import EvacuationEnv

env = EvacuationEnv(number_of_pedestrians=100, draw=True)

model = PPO(
            "MultiInputPolicy", 
            env, verbose=1
        )

model.load("saved_data\models\Our_potential_corrected_n-60_lr-0.0003_gamma-0.99_s-ped_ss-0.01_vr-0.1_27-Oct-19-20-49.zip")

import wandb

wandb.init()

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

env.save_animation()