from typing import Dict, List
import numpy as np


class Agent:
    start_position : np.ndarray                 # [2], r0_x and r0_y (same each reset)
    start_direction : np.ndarray                # [2], v0_x and v0_x (same each reset)
    start_doors : np.ndarray 
    position : np.ndarray                       # [2], r_x and r_y
    dooors : np.ndarray
    direction : np.ndarray                      # [2], v_x and v_y
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (position)
    enslaving_degree: float                     # 0 < enslaving_degree <= 1

    def __init__(self, enslaving_degree):
        self.start_position = np.array([0.0,-0.9]) #€ randomize or set at the entrance
        self.start_direction = np.zeros(2, dtype=np.float32)
        self.start_doors = np.zeros(3, dtype=np.float32)
        self.enslaving_degree = enslaving_degree
        
    def reset(self):
        self.position = self.start_position.copy() #€ if randomize, i need to randomize it here too
        self.direction = self.start_position.copy()
        self.doors = self.start_doors.copy()
        self.memory = {'position' : [],'doors' : []}

    def save(self):
        self.memory['position'].append(self.position.copy())
        self.memory['doors'].append(self.doors.copy())