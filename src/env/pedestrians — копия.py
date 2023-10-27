from functools import reduce
from typing import Dict, List
import numpy as np
from scipy.spatial import distance_matrix
from src.env.utils import Status, update_statuses
from src.env import constants


class Pedestrians:
    num : int                                   # number of pedestrians
    
    positions : np.ndarray                      # [num x 2], r_x and r_y
    directions : np.ndarray                     # [num x 2], v_x and v_y
    accelerations: np.ndarray                   # [num x 2], a_x and a_y
    statuses : np.ndarray                       # [num], status : Status 
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (positions and statuses)
    Ai: float  # Interaction force constant Ai
    Bi: float  # Interaction force constant Bi
    k: float  # Interaction force constant k
    kappa: float  # Interaction force constant kappa

    def __init__(self, num, v0i = 1, Ai = 2e3, Bi = 0.08, k = 1.2e5, kappa = 2.4e5, mass = 80, radius = constants.HUMAN_RADIUS):

        self.num = num
        self.v0i = v0i #m/s
        self.Ai = Ai
        self.Bi = Bi
        self.k = k
        self.kappa = kappa
        self.mass = mass
        self.radius = radius
        

    def reset(self, agent_position, exit_position):
        self.positions  = np.random.uniform(-np.minimum(constants.WIDTH, constants.HEIGHT), np.minimum(constants.WIDTH, constants.HEIGHT), size=(self.num, 2))
        # self.positions  = np.array([[-3., 3.], [3., 3.]])
        self.directions = np.random.uniform(-5.0, 5.0, size=(self.num, 2))
        # self.directions = np.array([[0.2, 0], [-0.2, 0]])        
        self.accelerations = np.zeros_like(self.positions)
        # self.normirate_directions()
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
        # print(self.directions)
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