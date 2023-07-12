# # %%
# from env import EvacuationEnv
# from agents import RandomAgent, RotatingAgent
# import numpy as np

# print('starting the experiment')

# env = EvacuationEnv(number_of_pedestrians=100, experiment_name='experiment_test', draw=True)
# # agent = RandomAgent(env.action_space)
# agent = RotatingAgent(env.action_space, 0.05)

# obs, _ = env.reset()
# for i in range(3):
#     action = agent.act(obs)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     if reward != 0:
#         print('reward = ', reward)

# # env.save_animation()
# # env.render()

# print('code completed succesfully')
# %%

from src.env import EvacuationEnv
from src.agents import RLAgent
import numpy as np

from src.utils import *
from src.params import *

experiment_prefix = 'test'

experiment_name = get_experiment_name(experiment_prefix)

env = EvacuationEnv(
    number_of_pedestrians = NUMBER_OF_PEDESTRIANS,
    experiment_name=experiment_name,
    verbose=VERBOSE_GYM,
    draw=DRAW,
    enable_gravity_embedding=ENABLE_GRAVITY_EMBEDDING
    )

# %%
from src.model.net import ActorCritic

action_space = env.action_space
obs_space = env.observation_space

act_size = get_act_size(action_space)
obs_size = get_obs_size(obs_space)

network = ActorCritic(
    obs_size=obs_size,
    act_size=act_size,
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    embedding=None
)
# %%
from src.agents import RLAgent

agent = RLAgent(
    action_space=action_space,
    network=network,
    mode=MODE,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    load_pretrain=False
)
# %%
from src.trainer import Trainer

trainer = Trainer(
    env=env,
    agent=agent,
    experiment_name=experiment_name,
    verbose=VERBOSE
    )
# %%
trainer.learn(number_of_episodes=300)
# %%
obs = env.observation_space.sample()

# %%
obs.values()
# %%
obs, _ = env.reset()
# %%
obs
# %%
