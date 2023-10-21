from src.env.utils import Status

from scipy.spatial import distance_matrix
import numpy as np

def sum_distance(pedestrians_positions, destination):
    """Mean distance between pedestrians and destination"""
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return distances.sum() / pedestrians_positions.shape[0]

class Reward:
    def __init__(self, 
        is_new_exiting_reward: bool,
        is_new_followers_reward: bool,
        is_termination_agent_wall_collision: bool
        ) -> None:
        
        self.is_new_exiting_reward = is_new_exiting_reward
        self.is_new_followers_reward = is_new_followers_reward
        self.is_termination_agent_wall_collision = is_termination_agent_wall_collision

    def estimate_intrinsic_reward(self, pedestrians_positions, exit_position):
        """This is intrinsic reward, which is given to the agent at each step"""
        return (0 - sum_distance(pedestrians_positions, exit_position))

    def estimate_status_reward(self, old_statuses, new_statuses, timesteps, num_pedestrians):
        """This reward is based on how pedestrians update their status
        
        VISCEK or FOLLOWER --> EXITING  :  (15 +  5 * time_factor)
        VISCEK             --> FOLLOWER :  (10 + 10 * time_factor)
        """
        reward = 0
        time_factor = 1 - timesteps / (200 * num_pedestrians)

        # Reward for new exiting
        if self.is_new_exiting_reward:
            prev = np.logical_or(old_statuses == Status.VISCEK, 
                                 old_statuses == Status.FOLLOWER)
            curr = new_statuses == Status.EXITING
            n = sum(np.logical_and(prev, curr))
            reward += (15 + 10 * time_factor) * n

        # Reward for new followers
        if self.is_new_followers_reward:
            prev = old_statuses == Status.VISCEK
            curr = new_statuses == Status.FOLLOWER
            n = sum(np.logical_and(prev, curr))
            reward += (10 + 5 * time_factor) * n

        return reward