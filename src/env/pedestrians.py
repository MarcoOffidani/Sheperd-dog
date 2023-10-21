from functools import reduce
from typing import Dict, List
import numpy as np

from src.env.utils import Status, update_statuses


class Pedestrians:
    num : int                                   # number of pedestrians
    
    positions : np.ndarray                      # [num x 2], r_x and r_y
    directions : np.ndarray                     # [num x 2], v_x and v_y
    accelerations: np.ndarray                   # [num x 2], a_x and a_y
    statuses : np.ndarray                       # [num], status : Status 
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (positions and statuses)

    def __init__(self, num : int):
        self.num = num

    def reset(self, agent_position, exit_position):
        self.positions  = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.directions = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.accelerations = np.zeros_like(self.positions)
        self.normirate_directions()
        self.statuses = np.array([Status.VISCEK for _ in range(self.num)])
        self.statuses = update_statuses(
            self.statuses,
            self.positions,
            agent_position,
            exit_position
        )
        self.memory = {'positions' : [], 'statuses' : []}
    
    def normirate_directions(self) -> None:
        x = self.directions
        self.directions = (x.T / np.linalg.norm(x, axis=1)).T

    def save(self):
        self.memory['positions'].append(self.positions.copy())
        self.memory['statuses'].append(self.statuses.copy())

    @property
    def status_stats(self):
        return {
            'escaped'  : sum(self.statuses == Status.ESCAPED),
            'exiting'  : sum(self.statuses == Status.EXITING),
            'following': sum(self.statuses == Status.FOLLOWER),
            'viscek'   : sum(self.statuses == Status.VISCEK),
        }