import numpy as np
from src.env.env import EvacuationEnv

e = EvacuationEnv(number_of_pedestrians=100)

e.reset()
e.step([1, 0])

for i in range(300):
    e.step([np.sin(i*0.1), np.cos(i*0.1)])
e.save_animation()
e.render()