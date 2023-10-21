from typing import Dict, List
import numpy as np


class Agent:
    start_position : np.ndarray                 # [2], r0_x and r0_y (same each reset)
    start_direction : np.ndarray                # [2], v0_x and v0_x (same each reset)
    position : np.ndarray                       # [2], r_x and r_y
    direction : np.ndarray                      # [2], v_x and v_y
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (position)
    enslaving_degree: float                     # 0 < enslaving_degree <= 1

    def __init__(self, enslaving_degree):
        self.start_position = np.zeros(2, dtype=np.float32)
        self.start_direction = np.zeros(2, dtype=np.float32)
        self.enslaving_degree = enslaving_degree
        
    def reset(self):
        self.position = self.start_position.copy()
        self.direction = self.start_position.copy()
        self.memory = {'position' : []}

    def save(self):
        self.memory['position'].append(self.position.copy())