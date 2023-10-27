from functools import reduce
from typing import Tuple

import numpy as np
from src.env import constants

from scipy.spatial import distance_matrix
from src.env.agents import Agent
from src.env.pedestrians import Pedestrians

from src.env.rewards import Reward
from src.env.utils import Exit, Status, SwitchDistances, update_statuses

#print(1)
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

    # def compute_acceleration_and_forces(self, pedestrians : Pedestrians, fv_directions):
    
    #     pedestrians_num = fv_directions.shape[1]
    #     acceleration = np.zeros((pedestrians_num, 2))
    #     forces = np.zeros((pedestrians_num, 2))
        
    #     # dm = distance_matrix(...)

    #     # n = fv_directions[..., None] - efv_directions[None, ...]

    #     for i in range(pedestrians_num):
    #         #scipy.spacial.distance_matrix
    #         #Force calculation
    #         for j in range(pedestrians_num):
                
    #             dij = np.linalg.norm(pedestrians.positions[i] - pedestrians.positions[j])
    #             nij = (pedestrians.positions[i] - pedestrians.positions[j]) / (dij + 1e-6)

    #             psych_force = pedestrians.Ai * np.exp((2 * pedestrians.radius - dij) / pedestrians.Bi) * nij

    #             if dij < 2 * pedestrians.radius:

    #                 body_force = pedestrians.k * (2 * pedestrians.radius - dij) * nij
    #                 tij = np.array([-nij[1], nij[0]])
    #                 tang_v = np.dot(pedestrians.directions[j] - pedestrians.directions[i], tij)
                    
    #                 sliding_force = pedestrians.kappa * (2 * pedestrians.radius - dij) * tang_v * tij
    #                 forces[i] += (psych_force + body_force + sliding_force)
    #             else:
    #                 forces[i] += psych_force

    #         acceleration[i] = (pedestrians.v0i * fv_directions.T[i] - pedestrians.directions[i]) / pedestrians.mass + forces[i]
    #     return acceleration, forces
    
    def compute_acceleration_and_forces(self, pedestrians : Pedestrians, old_directions, efv):
        #Viscek acceleration
        # print(old_directions.shape[0])
        pedestrians_num = old_directions.shape[0]
        acceleration = np.zeros((pedestrians_num, 2))
        forces = np.zeros((pedestrians_num, 2))
        # delta_v = np.zeros((pedestrians_num, 2))
        # positions = pedestrians.positions[efv]
        desired_directions = pedestrians.directions[efv]
        for i in range(pedestrians_num):
            #scipy.spacial.distance_matrix
            #Force calculation
            # desired_acceleration = (average_velocity - pedestrians.directions[i]) / self.step_size
            # print(f'average_velocity: {average_velocity}')
            desired_acceleration = 0
            for j in range(pedestrians_num):
                if j != i:
                    # print(f'Interaction i:{i}, j:{j}:\n')
                    dij = np.linalg.norm(pedestrians.positions[i] - pedestrians.positions[j])
                    nij = (pedestrians.positions[i] - pedestrians.positions[j]) / (dij + 1e-6)
                    # if (i == 0 and j == 1) or (i == 1 and j == 0):
                    #     print(f'i: {i}, j: {j}, nij = {nij}')
                    
                    psych_force = pedestrians.Ai * np.exp((2 * pedestrians.radius - dij) / pedestrians.Bi) * nij
                    # print(f'Iteration i:{i}, j:{j}\n2 * pedestrians.radius - dij {2 * pedestrians.radius - dij}, np.exp((2 * pedestrians.radius - dij) / pedestrians.Bi) {np.exp((2 * pedestrians.radius - dij) / pedestrians.Bi)}\n nij: {nij}')

                    # print('psych_force', psych_force)
                    # if np.abs(psych_force[0]) >= 5.0:
                    #     print(f'i :{i}, j:{j}, dij: {dij}')
                    psych_force = np.clip(psych_force, -2, 2)
                    
                    # print(psych_force)
                    # if np.abs(psych_force[0]) >= 5.0:
                    #     print(f'i :{i}, j:{j}, dij: {dij}')
                    # print(f'2 * pedestrians.radius - dij {2 * pedestrians.radius - dij}, np.exp((2 * pedestrians.radius - dij) / pedestrians.Bi) {np.exp((2 * pedestrians.radius - dij) / pedestrians.Bi)}')
                    if dij < 2 * pedestrians.radius:

                        body_force = pedestrians.k * (2 * pedestrians.radius - dij) * nij
                        tij = np.array([-nij[1], nij[0]])
                        tang_v = np.dot(old_directions[j] - old_directions[i], tij)
                        
                        sliding_force = pedestrians.kappa * (2 * pedestrians.radius - dij) * tang_v * tij

                        body_force = np.clip(body_force, -5, 5)
                        sliding_force = np.clip(sliding_force, -3, 3)

                        forces[i] += (psych_force + body_force + sliding_force)
                        # forces[i] += psych_force * 2
                        # print(1)
                        # forces[i] += psych_force + body_force
                        # print(f'body_force {body_force}')
                        # print(f'Interaction i:{i}, j:{j}:\n \
                                    # dij  {dij}, nij {nij}, tij {tij}\n\
                                    # tang_v  {tang_v}, 2 * pedestrians.radius {2 * pedestrians.radius}')
                        # print(f'Interaction i:{i}, j:{j}:\npsych_force  {psych_force}, body_force {body_force}, sliding_force {sliding_force}')
                    else:
                        forces[i] += psych_force
                        # forces[i] += 0
                else:
                    continue
            # forces[i] = 0
            # acceleration[i] = desired_acceleration + forces[i] / pedestrians.mass
            # acceleration[i] = desired_acceleration + forces[i] / pedestrians.mass
            acceleration[i] = (pedestrians.directions[i] - old_directions[i]) / 0.6 + forces[i] / pedestrians.mass
        # print(f'fv_directions.shape: {fv_directions.shape}, acceleration.shape: {acceleration.shape}, forces.shape {forces.shape}')
        return acceleration

    def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent, now : int) -> Tuple[Pedestrians, bool, float, float]:

        # print(np.any(pedestrians.statuses == Status.FALLEN))

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
        # efv_directions = (efv_directions.T / np.linalg.norm(efv_directions, axis=1)).T  # TODO: try using keepdims for simplisity #unit vector
        
        # Find neighbours between following and viscek (fv) particles and all other moving particles
        fv = reduce(np.logical_or, (following, viscek))
        dm = distance_matrix(pedestrians.positions[fv],
                             pedestrians.positions[efv], 2)
        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0) 

        #Find desired speed
        fv_directions = self.estimate_mean_direction_among_neighbours(
            intersection, efv_directions
        ) 

        fv_directions = fv_directions / np.linalg.norm(fv_directions, axis=1, keepdims= True)

        directions_old = pedestrians.directions[efv].copy()
        # pedestrians.position[efv] += pedestrians.directions[efv] * self.step_size

        #Desired speed
        average_velocity = np.mean(fv_directions, axis = 1)

        pedestrians.directions[fv] = average_velocity

        #Update desired speed for the following pedestrians
        # Add enslaving factor of leader's direction to following particles
        f_directions = pedestrians.directions[following] 
        f_positions = pedestrians.positions[following]
        l_directions = (agent.position.reshape(1, -1) - f_positions) / self.step_size
        # l_directions /=  np.linalg.norm(l_directions, axis=1, keepdims=True) / self.step_size
        f_directions = agent.enslaving_degree * l_directions + (1. - agent.enslaving_degree) * f_directions
        pedestrians.directions[following] = f_directions

        acceleration = self.compute_acceleration_and_forces(pedestrians, directions_old, efv)
        # forces += pedestrians.directions[fv] / self.step_size * pedestrians.mass 
        # pedestrians.positions[efv] += pedestrians.directions[efv] * self.step_size # + a * self.step_size ** 2 / 2
        
        
        
        #TODO Forces
        pedestrians.directions[efv] = directions_old + acceleration * self.step_size
        pedestrians.positions[efv] += directions_old * self.step_size + 1.5 * acceleration * self.step_size ** 2
        # Record new directions of following and viscek pedestrians

        # pedestrians.directions[fv] = fv_directions.T * self.step_size + acceleration * self.step_size ** 2 / 2 
        

        

        
        # to_fall = np.where(dm < SwitchDistances.to_fall, 1, 0).any(axis=0)
        # to_fall = np.flatnonzero(efv)[to_fall]
        # pedestrians.directions[to_fall] *= 0
        # pedestrians.statuses[to_fall] = Status.FALLEN
        # # print(np.any(pedestrians.statuses == Status.FALLEN))
        
        # Record new positions of exiting, following and viscek pedestrians
        # pedestrians.positions[efv] += pedestrians.directions[efv] + forces

        # Handling of wall collisions
        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        # print(miss.shape)
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
        # print('Position:', pedestrians.positions, "Velocity:", pedestrians.directions)
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
