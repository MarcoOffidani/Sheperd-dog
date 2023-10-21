from functools import reduce
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix
from src.env import constants

from enum import Enum, auto

def update_statuses(statuses, pedestrian_positions, agent_position, exit_position):
    """Measure statuses of all pedestrians based on their position"""
    new_statuses = statuses.copy()

    following = is_distance_low(
        pedestrian_positions, agent_position, SwitchDistances.to_leader)
    new_statuses[following] = Status.FOLLOWER
    
    exiting = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_exit)
    new_statuses[exiting] = Status.EXITING
    
    escaped = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_escape)
    new_statuses[escaped] = Status.ESCAPED
    
    viscek = np.logical_not(reduce(np.logical_or, (exiting, following, escaped)))
    new_statuses[viscek] = Status.VISCEK
    
    return new_statuses

def is_distance_low(
    pedestrians_positions: npt.NDArray, 
    destination: npt.NDArray, 
    radius: float
    ) -> npt.NDArray:
    """Get boolean matrix showing pedestrians,
    which are closer to destination than raduis 

    Args:
        pedestrians_positions (npt.NDArray): coordinates of pedestrians 
        (dim: [n x 2])
        
        destination (npt.NDArray): coordinates of destination
        (dim: [2])
        
        radius (float): max distance to destination

    Returns:
        npt.NDArray: boolean matrix
    """
    
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return np.where(distances < radius, True, False).squeeze()


class UserEnum(Enum):
    @classmethod
    def all(cls):
        return list(map(lambda c: c, cls))


class Status(UserEnum):
    VISCEK = auto()
    "Pedestrian under Viscek rules."

    FALLEN = auto()

    FRIGHTENED = auto()

    FOLLOWER = auto()
    "Follower of the leader particle (agent)."

    EXITING = auto()
    "Pedestrian in exit zone."

    ESCAPED = auto()
    "Evacuated pedestrian."


class SwitchDistances:
    to_leader: float     = constants.SWITCH_DISTANCE_TO_LEADER
    to_exit: float       = constants.SWITCH_DISTANCE_TO_EXIT
    to_escape: float     = constants.SWITCH_DISTANCE_TO_ESCAPE
    to_pedestrian: float = constants.SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN
    to_fall: float       = constants.SWITCH_DISTANCE_TO_FALL


class Exit:
    position : np.ndarray
    def __init__(self):
        self.position = np.array([0, -1], dtype=np.float32)


class Time:
    def __init__(self, 
        max_timesteps : int = constants.MAX_TIMESTEPS,
        n_episodes : int = constants.N_EPISODES,
        n_timesteps : int = constants.N_TIMESTEPS
        ) -> None:
        self.now = 0
        self.max_timesteps = max_timesteps
        self.n_episodes = n_episodes
        self.overall_timesteps = n_timesteps

    def reset(self):
        self.now = 0
        self.n_episodes += 1

    def step(self):
        self.now += 1
        self.overall_timesteps += 1
        return self.truncated()
        
    def truncated(self):
        return self.now >= self.max_timesteps 

