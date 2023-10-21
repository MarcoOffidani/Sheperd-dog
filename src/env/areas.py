from functools import reduce
from typing import Tuple

import numpy as np
from src.env import constants

from scipy.spatial import distance_matrix
from src.env.agents import Agent
from src.env.pedestrians import Pedestrians

from src.env.rewards import Reward
from src.env.utils import Exit, Status, SwitchDistances, update_statuses


class Area:
    def __init__(self, 
        reward: Reward,
        width = constants.WIDTH, 
        height = constants.HEIGHT,
        step_size = constants.STEP_SIZE,
        noise_coef = constants.NOISE_COEF,
        ):
        self.reward = reward
        self.width = width
        self.height = height
        self.step_size = step_size
        self.noise_coef = noise_coef
        self.exit = Exit()
    
    def reset(self):
        pass

    @staticmethod
    def estimate_mean_direction_among_neighbours(
            intersection,           # [f+v, f+v+e]  boolean matrix
            efv_directions,        # [f+v+e, 2]    vectors of directions of pedestrians
            # n_intersections         # [f+v]         amount of neighbouring pedestrians
        ):
        """Viscek model"""
    
        n_intersections = np.maximum(1, intersection.sum(axis=1))

        # Estimate the contibution if each neighbouring particle 
        fv_directions_x = (intersection * efv_directions[:, 0]).sum(axis=1) / n_intersections
        fv_directions_y = (intersection * efv_directions[:, 1]).sum(axis=1) / n_intersections
        fv_theta = np.arctan2(fv_directions_x, fv_directions_y)
                                
        # Create randomization noise to obtained directions
        noise = np.random.normal(loc=0., scale=constants.NOISE_COEF, size=len(n_intersections))
        
        # New direction = estimated_direction + noise
        fv_theta = fv_theta + noise
        
        return np.vstack((np.cos(fv_theta), np.sin(fv_theta)))

    def escaped_directions_update(self, pedestrians: Pedestrians, escaped) -> None:
        pedestrians.directions[escaped] = 0
        pedestrians.positions[escaped] = self.exit.position

    def exiting_directions_update(self, pedestrians: Pedestrians, exiting) -> None:
        if any(exiting):
            vec2exit = self.exit.position - pedestrians.positions[exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T
            pedestrians.directions[exiting] = vec2exit

    def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent, now : int) -> Tuple[Pedestrians, bool, float, float]:

        # Check evacuated pedestrians & record new directions and positions of escaped pedestrians
        escaped = pedestrians.statuses == Status.ESCAPED
        self.escaped_directions_update(pedestrians, escaped)
        
        # Check exiting pedestrians & record new directions for exiting pedestrians
        exiting = pedestrians.statuses == Status.EXITING
        self.exiting_directions_update(pedestrians, exiting)

        # Check following pedestrians & record new directions for following pedestrians
        following = pedestrians.statuses == Status.FOLLOWER

        # Check viscek pedestrians
        viscek = pedestrians.statuses == Status.VISCEK
        
        # Use all moving particles (efv -- exiting, following, viscek) to estimate the movement of viscek particles
        efv = reduce(np.logical_or, (exiting, following, viscek))
        efv_directions = pedestrians.directions[efv]
        efv_directions = (efv_directions.T / np.linalg.norm(efv_directions, axis=1)).T  # TODO: try using keepdims for simplisity
        
        # Find neighbours between following and viscek (fv) particles and all other moving particles
        fv = reduce(np.logical_or, (following, viscek))
        dm = distance_matrix(pedestrians.positions[fv],
                             pedestrians.positions[efv], 2)
        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0) 

        fv_directions = self.estimate_mean_direction_among_neighbours(
            intersection, efv_directions
        )            

        # Record new directions of following and viscek pedestrians
        pedestrians.directions[fv] = fv_directions.T * self.step_size
        
        # Add enslaving factor of leader's direction to following particles
        f_directions = pedestrians.directions[following]
        l_directions = agent.direction
        f_directions = agent.enslaving_degree * l_directions + (1. - agent.enslaving_degree) * f_directions
        pedestrians.directions[following] = f_directions
        
        # Record new positions of exiting, following and viscek pedestrians
        pedestrians.positions[efv] += pedestrians.directions[efv] 

        # Handling of wall collisions
        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        pedestrians.directions *= np.where(miss!=0, -1, 1)

        # Estimate pedestrians statues, reward & update statuses
        old_statuses = pedestrians.statuses.copy()
        new_pedestrians_statuses = update_statuses(
            statuses=pedestrians.statuses,
            pedestrian_positions=pedestrians.positions,
            agent_position=agent.position,
            exit_position=self.exit.position
        )
        reward_pedestrians = self.reward.estimate_status_reward(
            old_statuses=old_statuses,
            new_statuses=new_pedestrians_statuses,
            timesteps=now,
            num_pedestrians=pedestrians.num
        )
        intrinsic_reward = self.reward.estimate_intrinsic_reward(
            pedestrians_positions=pedestrians.positions,
            exit_position=self.exit.position
        )
        pedestrians.statuses = new_pedestrians_statuses
        
        # Termination due to all pedestrians escaped
        if sum(pedestrians.statuses == Status.ESCAPED) == pedestrians.num: # TODO: np.all(pedestrians.statuses == Status.ESCAPED) ?
            termination = True
        else: 
            termination = False

        return pedestrians, termination, reward_pedestrians, intrinsic_reward

    def agent_step(self, action : list, agent : Agent) -> Tuple[Agent, bool, float]:
        """
        Perform agent step:
            1. Read & preprocess action
            2. Check wall collision
            3. Return (updated agent, termination, reward)
        """
        action = np.array(action)
        action /= np.linalg.norm(action) + constants.EPS # np.clip(action, -1, 1, out=action)

        agent.direction = self.step_size * action
        
        if not self._if_wall_collision(agent):
            agent.position += agent.direction
            return agent, False, 0.
        else:
            return agent, self.reward.is_termination_agent_wall_collision, -5.

    def _if_wall_collision(self, agent : Agent):
        pt = agent.position + agent.direction

        left  = pt[0] < -self.width
        right = pt[0] > self.width
        down  = pt[1] < -self.height  
        up    = pt[1] > self.height
        
        if left or right or down or up:
            return True
        return False

