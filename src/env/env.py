# ## %%
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

from src.env.agents import Agent
from src.env.pedestrians import Pedestrians
from src.env.rewards import Reward
from src.env.utils import Exit, Status, SwitchDistances, Time
from src.env.areas import Area


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from src.env import constants
from src.params import WALK_DIAGRAM_LOGGING_FREQUENCY

import wandb

log = logging.getLogger(__name__)

def setup_logging(verbose: bool, experiment_name: str) -> None:
    logs_folder = constants.SAVE_PATH_LOGS
    if not os.path.exists(logs_folder): os.makedirs(logs_folder)

    logs_filename = os.path.join(logs_folder, f"logs_{experiment_name}.log")

    logging.basicConfig(
        filename=logs_filename, filemode="w",
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )


def count_new_statuses(old_statuses, new_statuses):
    """Get number of pedestrians, who have updated their status"""
    count = {}
    for status in Status.all():
        count[status] = sum(
            np.logical_and(new_statuses == status, old_statuses != status)
        )
    return count


def grad_potential_pedestrians(
        agent: Agent, pedestrians: Pedestrians, alpha: float = constants.ALPHA
    ) -> np.ndarray:
    R = agent.position[np.newaxis, :] - pedestrians.positions
    R = R[pedestrians.statuses == Status.VISCEK]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + constants.EPS
        grad = - alpha / norm ** (alpha + 2) * R
        grad = grad.sum(axis = 0)
    else:
        grad = np.zeros(2)
        
    return grad

def grad_time_derivative_pedestrians(
        agent: Agent, pedestrians: Pedestrians, alpha: float = constants.ALPHA
    ) -> np.ndarray:

    R = agent.position[np.newaxis, :] - pedestrians.positions
    R = R[pedestrians.statuses == Status.VISCEK]

    V = agent.direction[np.newaxis, :] - pedestrians.positions
    V = V[pedestrians.statuses == Status.VISCEK]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + constants.EPS
        grad = - alpha / norm ** (alpha + 4) * (V * norm**2 - (alpha + 2) * np.sum(V * R, axis=1, keepdims=True) * R)
        grad = grad.sum(axis=0)
    else:
        grad = np.zeros(2)
        
    return grad


def grad_potential_exit(
        agent: Agent, pedestrians: Pedestrians, exit: Exit, alpha: float = constants.ALPHA
    ) -> np.ndarray:
    R = agent.position - exit.position
    norm = np.linalg.norm(R) + constants.EPS
    grad = - alpha / norm ** (alpha + 2) * R
    grad *= sum(pedestrians.statuses == Status.FOLLOWER)
    return grad


def grad_time_derivative_exit(
        agent: Agent, pedestrians: Pedestrians, exit: Exit, alpha: float = constants.ALPHA
    ) -> np.ndarray:

    R = agent.position - exit.position

    V = agent.direction
    
    N = sum(pedestrians.statuses == Status.FOLLOWER)

    if N != 0:
        norm = np.linalg.norm(R) + constants.EPS
        grad = - alpha / norm ** (alpha + 4) * (V * norm**2 - (alpha + 2) * np.dot(V, R) * R)
        grad *= N
    else:
        grad = np.zeros(2)
        
    return grad


class EvacuationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    Evacuation Game Enviroment for Gym
    Continious Action and Observation Space.
    """
    def __init__(self, 
        experiment_name='test',
        number_of_pedestrians=constants.NUM_PEDESTRIANS,
        
        # leader params
        enslaving_degree=constants.ENSLAVING_DEGREE,
        
        # area params
        width=constants.WIDTH,
        height=constants.HEIGHT,
        step_size=constants.STEP_SIZE,
        noise_coef=constants.NOISE_COEF,
        
        # reward params
        is_termination_agent_wall_collision=constants.TERMINATION_AGENT_WALL_COLLISION,
        is_new_exiting_reward=constants.IS_NEW_EXITING_REWARD,
        is_new_followers_reward=constants.IS_NEW_FOLLOWERS_REWARD,
        intrinsic_reward_coef=constants.INTRINSIC_REWARD_COEF,
        
        # time params
        max_timesteps=constants.MAX_TIMESTEPS,
        n_episodes=constants.N_EPISODES,
        n_timesteps=constants.N_TIMESTEPS,
        
        # gravity embedding params
        enabled_gravity_embedding=constants.ENABLED_GRAVITY_EMBEDDING,
        enabled_gravity_and_speed_embedding = constants.ENABLED_GRAVITY_AND_SPEED_EMBEDDING,
        alpha=constants.ALPHA,
        
        # logging params
        verbose=False,
        render_mode=None,
        draw=False
        
        ) -> None:
        super(EvacuationEnv, self).__init__()
        
        # setup env
        self.pedestrians = Pedestrians(num=number_of_pedestrians)
        
        reward = Reward(
            is_new_exiting_reward=is_new_exiting_reward,
            is_new_followers_reward=is_new_followers_reward,
            is_termination_agent_wall_collision=is_termination_agent_wall_collision)        
        
        self.area = Area(
            reward=reward, 
            width=width, height=height, 
            step_size=step_size, noise_coef=noise_coef)
        
        self.time = Time(
            max_timesteps=max_timesteps, 
            n_episodes=n_episodes, n_timesteps=n_timesteps)
        
        # setup agent
        self.agent = Agent(enslaving_degree=enslaving_degree)
        
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0        
        self.enabled_gravity_embedding = enabled_gravity_embedding
        self.enabled_gravity_and_speed_embedding = enabled_gravity_and_speed_embedding
        self.alpha = alpha

        self.action_space = spaces.Box(low=-1., high=1., shape=(5,), dtype=np.float32) #€ add one extra dimension for each door
        self.observation_space = self._get_observation_space()
        
        # logging
        self.render_mode = render_mode
        self.experiment_name = experiment_name
        setup_logging(verbose, experiment_name)

        # drawing
        self.draw = draw
        self.save_next_episode_anim = False
        log.info(f'Env {self.experiment_name} is initialized.')        
        
    def _get_observation_space(self):        
        observation_space = {
            'agent_position' : spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        }
        
        if self.enabled_gravity_and_speed_embedding:
            observation_space['grad_potential_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_potential_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)  
            observation_space['grad_time_derivative_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_time_derivative_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)   
        elif self.enabled_gravity_embedding:
            observation_space['grad_potential_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_potential_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        else:
            observation_space['pedestrians_positions'] = \
                spaces.Box(low=-1, high=1, shape=(self.pedestrians.num, 2), dtype=np.float32)
            observation_space['exit_position'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)

        return spaces.Dict(observation_space)

    def _get_observation(self):
        observation = {}
        observation['agent_position'] = self.agent.position

        if self.enabled_gravity_and_speed_embedding:
            observation['grad_potential_pedestrians'] = grad_potential_pedestrians(
                agent=self.agent, 
                pedestrians=self.pedestrians, 
                alpha=self.alpha
            )
            observation['grad_potential_exit'] = grad_potential_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
            observation['grad_time_derivative_exit'] = grad_time_derivative_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
            observation['grad_time_derivative_pedestrians'] = grad_time_derivative_pedestrians(
                agent=self.agent,
                pedestrians=self.pedestrians,
                alpha=self.alpha
            )
        elif self.enabled_gravity_embedding:
            observation['grad_potential_pedestrians'] = grad_potential_pedestrians(
                agent=self.agent, 
                pedestrians=self.pedestrians, 
                alpha=self.alpha
            )
            observation['grad_potential_exit'] = grad_potential_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
        else:
            observation['pedestrians_positions'] = self.pedestrians.positions
            observation['exit_position'] = self.area.exit.position

        return observation

    def reset(self, seed=None):
        if self.save_next_episode_anim or (self.time.n_episodes + 1) % WALK_DIAGRAM_LOGGING_FREQUENCY == 0:
            self.draw = True
            self.save_next_episode_anim = True

        if self.time.n_episodes > 0:
            logging_dict = {
                "episode_intrinsic_reward" : self.episode_intrinsic_reward,
                "episode_status_reward" : self.episode_status_reward,
                "episode_reward" : self.episode_reward,
                "episode_length" : self.time.now,
                "escaped_pedestrians" : sum(self.pedestrians.statuses == Status.ESCAPED),
                "exiting_pedestrians" : sum(self.pedestrians.statuses == Status.EXITING),
                "following_pedestrians" : sum(self.pedestrians.statuses == Status.FOLLOWER),
                "viscek_pedestrians" : sum(self.pedestrians.statuses == Status.VISCEK),
                "overall_timesteps" : self.time.overall_timesteps
            }
            log.info('\t'.join([f'{key}={value}' for key, value in logging_dict.items()]))
            wandb.log(logging_dict)
        
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0 
        self.time.reset()
        self.area.reset()
        self.agent.reset()
        self.pedestrians.reset(agent_position=self.agent.position,
                               exit_position=self.area.exit.position)
        self.pedestrians.save()
        log.info('Env is reseted.')
        return self._get_observation(), {}

    def step(self, action: list):
        # Increment time
        truncated = self.time.step()

        # Agent step
        self.agent, terminated_agent, reward_agent = self.area.agent_step(action, self.agent)
        #self.area.doors = self.agent.doors #€
        # Pedestrians step
        self.pedestrians, terminated_pedestrians, reward_pedestrians, intrinsic_reward = \
            self.area.pedestrians_step(self.pedestrians, self.agent, self.time.now)
        
        # Save positions for rendering an animation
        if self.draw:
            self.pedestrians.save()
            self.agent.save()

        # Collect rewards
        reward = reward_agent + reward_pedestrians + self.intrinsic_reward_coef * intrinsic_reward
        
        # Record observation
        observation = self._get_observation()
        
        # Draw animation
        if (terminated_agent or terminated_pedestrians or truncated) and self.draw:
            self.save_animation()
        
        # Log reward
        self.episode_reward += reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_status_reward += reward_agent + reward_pedestrians
        return observation, reward, terminated_agent or terminated_pedestrians, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.position[0], self.agent.position[1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        ax.add_patch(following_zone)
        
        # Draw pedestrians
        for status in Status.all():
            selected_pedestrians = self.pedestrians.statuses == status
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(self.pedestrians.positions[selected_pedestrians, 0], 
                    self.pedestrians.positions[selected_pedestrians, 1],
                lw=0, marker='.', color=color)
            # for i in range(self.pedestrians.directions.shape[0]):
            #     ax.plot(self.pedestrians.positions[i],
            #     self.pedestrians.positions[i] + self.pedestrians.directions[i])

        # Draw agent
        ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')

        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--', color='grey')
        ####
        print('this is render')
        #print(scaled_doors)
        plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 + constants.VERTICAL_WALL_HALF_WIDTH, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2], linestyle='-', color='red')
        # Plot the bottom side of the vertical door

        plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 - constants.VERTICAL_WALL_HALF_WIDTH, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH + scaled_doors[2], linestyle='-', color='red')
		
        # Plot the left side of left door
        #shit= (0.95 + 1)/2 * constants.WALL_HOLE_HALF_WIDTH   #€ rename this



        plt.hlines([0],  opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH, opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[0]  , linestyle='-', color='red')
        # Plot the right side of left door

        plt.hlines([0],  opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[0], opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH   , linestyle='-', color='red')
        ####
        plt.title(f"{self.experiment_name}. Timesteps: {self.time.now}")

        plt.tight_layout()
        if not os.path.exists(constants.SAVE_PATH_PNG): os.makedirs(constants.SAVE_PATH_PNG)
        filename = os.path.join(constants.SAVE_PATH_PNG, f'{self.experiment_name}_{self.time.now}.png')
        plt.savefig(filename)
        plt.show()
        log.info(f"Env is rendered and pnd image is saved to {filename}")

    def save_animation(self):
        
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.title(f"{self.experiment_name}\nn_episodes = {self.time.n_episodes}")
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--',  color='grey')
        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])

        #plot Middle Wall HERE€
        #plt.hlines([0], -self.area.width, -constants.WALL_HOLE_HALF_WIDTH, linestyle='--', color='grey')
        #plt.hlines([0], constants.WALL_HOLE_HALF_WIDTH, self.area.width, linestyle='--', color='grey')
        # Define the positions of the two openings
        # Define the positions of the openings
        opening_positions = [-0.5, 0.5]


        # Define the width of the wall (assuming self.area.width is available)
        wall_width = self.area.width


        # Plot the left segment of the wall
        plt.hlines([0], -wall_width, opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH, linestyle='-', color='grey')
        # Plot the left side of left door
        #shit= (0.95 + 1)/2 * constants.WALL_HOLE_HALF_WIDTH   #€ rename this
        scaled_doors= (self.agent.memory['doors'][0][0], self.agent.memory['doors'][0][1], self.agent.memory['doors'][0][2]) 
        
        print('this is save animation')
        print(scaled_doors)
        plt.hlines([0],  opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH, opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[0]  , linestyle='-', color='red')
        # Plot the right side of left door

        plt.hlines([0],  opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[0], opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH   , linestyle='-', color='red')
        # Plot the middle segment of the wall (between the two openings)
        plt.hlines([0], opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH, opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH, linestyle='-', color='grey')
        # Plot the left side of right door
 
        plt.hlines([0],  opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH, opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[1]  , linestyle='-', color='red')
        # Plot the right side of right door

        plt.hlines([0],  opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[1], opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH   , linestyle='-', color='red')
        # Plot the right segment of the wall
        plt.hlines([0], opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH, wall_width, linestyle='-', color='grey')
        # Plot the vertical wall with an opening in the middle
        # Plot the first vertical wall segment (top)
        plt.vlines([constants.VERTICAL_WALL_POSITION], 1.0, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH, linestyle='-', color='grey')
        # Plot the top side of vertical door
 
        plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 + constants.VERTICAL_WALL_HALF_WIDTH, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2], linestyle='-', color='red')
        # Plot the bottom side of the vertical door

        plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 - constants.VERTICAL_WALL_HALF_WIDTH, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH + scaled_doors[2], linestyle='-', color='red')
        # Plot the second vertical wall segment (bottom)
        plt.vlines([constants.VERTICAL_WALL_POSITION], 0.0, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH, linestyle='-', color='grey')





        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.memory['position'][0][0], self.agent.memory['position'][0][1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        following_zone_plots = ax.add_patch(following_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')
        
        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        pedestrian_position_plots = {}
        for status in Status.all():
            selected_pedestrians = self.pedestrians.memory['statuses'][0] == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
            pedestrian_position_plots[status] = \
                ax.plot(self.pedestrians.memory['positions'][0][selected_pedestrians, 0], 
                self.pedestrians.memory['positions'][0][selected_pedestrians, 1],
                lw=0, marker='.', color=color)[0]

        # Draw agent
        agent_position_plot = ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')[0]
        ##draw lines by ChatGDP (tm)
        # Create empty lists to store the door lines
        door_lines = []

        # Plot the top side of vertical door
        top_door_line = plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 + constants.VERTICAL_WALL_HALF_WIDTH, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2], linestyle='-', color='red')
        door_lines.append(top_door_line)

        # Plot the bottom side of the vertical door
        bottom_door_line = plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 - constants.VERTICAL_WALL_HALF_WIDTH, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH + scaled_doors[2], linestyle='-', color='red')
        door_lines.append(bottom_door_line)

        # Plot the left side of left door
        left_door_line = plt.hlines([0], opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH, opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[0], linestyle='-', color='red')
        door_lines.append(left_door_line)

        # Plot the right side of left door
        right_door_line = plt.hlines([0], opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[0], opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH, linestyle='-', color='red')
        door_lines.append(right_door_line)

        # Plot the left side of right door
        left_door_line2 = plt.hlines([0], opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH, opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[1], linestyle='-', color='red')
        door_lines.append(left_door_line2)

        # Plot the right side of right door
        right_door_line2 = plt.hlines([0], opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[1], opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH, linestyle='-', color='red')
        door_lines.append(right_door_line2)
        # Initialize the white lines for the complementary areas
        white_door_lines = []

        # Vertical white door line (for the complementary area of the vertical door)
        vertical_white_line = plt.vlines([constants.VERTICAL_WALL_POSITION], 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2], 0.5 - constants.VERTICAL_WALL_HALF_WIDTH + scaled_doors[2], linestyle='-', color='white')
        white_door_lines.append(vertical_white_line)

        # Horizontal white door lines (for the complementary areas of the horizontal doors)
        for position in opening_positions:
            horizontal_white_line = plt.hlines([0], position - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[0], position + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[0], linestyle='-', color='white')
            white_door_lines.append(horizontal_white_line)
        def update(i):
            agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            following_zone_plots.set_center(agent_coordinates)
            scaled_doors= (self.agent.memory['doors'][i][0], self.agent.memory['doors'][i][1], self.agent.memory['doors'][i][2]) 
            for status in Status.all():
                selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])

            #  agent_position_plot.set_xdata(agent_coordinates[0])
            #  agent_position_plot.set_ydata(agent_coordinates[1])
            agent_position_plot.set_data(agent_coordinates)
            #  Update agent doors
            for idx, door_line in enumerate(door_lines):
                door_line.set_data(scaled_doors)
        def update(i):
            agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            following_zone_plots.set_center(agent_coordinates)
            scaled_doors = (self.agent.memory['doors'][i][0], self.agent.memory['doors'][i][1], self.agent.memory['doors'][i][2]) 
            for status in Status.all():
                selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])

            agent_position_plot.set_data(agent_coordinates[0], agent_coordinates[1])

            # Update vertical door lines
            door_lines[0].set_segments([[(constants.VERTICAL_WALL_POSITION, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH), 
                                         (constants.VERTICAL_WALL_POSITION, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2])]])
            door_lines[1].set_segments([[(constants.VERTICAL_WALL_POSITION, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH), 
                                         (constants.VERTICAL_WALL_POSITION, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH + scaled_doors[2])]])

            # Update horizontal door lines (left and right)
            door_lines[2].set_segments([[(opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH, 0), 
                                         (opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[0], 0)]])
            door_lines[3].set_segments([[(opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[0], 0), 
                                         (opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH, 0)]])
            door_lines[4].set_segments([[(opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH, 0), 
                                         (opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[1], 0)]])
            door_lines[5].set_segments([[(opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[1], 0), 
                                         (opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH, 0)]])
            # Update the white lines for the complementary areas
            white_door_lines[0].set_segments([[(constants.VERTICAL_WALL_POSITION, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2]), 
                                               (constants.VERTICAL_WALL_POSITION, 0.5 - constants.VERTICAL_WALL_HALF_WIDTH + scaled_doors[2])]])

            white_door_lines[1].set_segments([[(opening_positions[0] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[0], 0), 
                                               (opening_positions[0] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[0], 0)]])
            white_door_lines[2].set_segments([[(opening_positions[1] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[1], 0), 
                                               (opening_positions[1] + constants.WALL_HOLE_HALF_WIDTH - scaled_doors[1], 0)]])
         # def update(i):
             # agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
             # following_zone_plots.set_center(agent_coordinates)
             # scaled_doors = (self.agent.memory['doors'][i][0], self.agent.memory['doors'][i][1], self.agent.memory['doors'][i][2])

            # #  Update pedestrian positions (assuming pedestrian_position_plots is a dictionary)
             # for status in Status.all():
                 # selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                 # pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                 # pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])

            # #  Update agent position
             # agent_position_plot.set_data(agent_coordinates)

            # #  Update agent doors
             # for idx, door_line in enumerate(door_lines):
                 # if idx == 0 or idx == 1:
                     # door_line.set_ydata([0.5 + constants.VERTICAL_WALL_HALF_WIDTH, 0.5 + constants.VERTICAL_WALL_HALF_WIDTH - scaled_doors[2]])
                 # else:
                     # door_line.set_xdata([opening_positions[idx - 2] - constants.WALL_HOLE_HALF_WIDTH, opening_positions[idx - 2] - constants.WALL_HOLE_HALF_WIDTH + scaled_doors[idx - 2]])
        # def update(i):
            # agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            # following_zone_plots.set_center(agent_coordinates)
            # scaled_doors = (self.agent.memory['doors'][i][0], self.agent.memory['doors'][i][1], self.agent.memory['doors'][i][2])

            # # Update pedestrian positions (assuming pedestrian_position_plots is a dictionary)
            # for status in Status.all():
                # selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                # pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                # pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])

            # # Update agent position
            # agent_position_plot.set_data(agent_coordinates)

            # # Update horizontal door lines
            # for idx, horizontal_door_line_collection in enumerate(horizontal_door_line_collections):
                # # Calculate the new positions for left and right door lines
                # left_door_x = opening_positions[idx] - constants.WALL_HOLE_HALF_WIDTH
                # right_door_x = left_door_x + scaled_doors[idx]

                # # Update the data for the horizontal door LineCollection
                # horizontal_door_line_collection.set_segments([[(left_door_x, 0), (right_door_x, 0)]])

            # # Update vertical door lines
            # for idx, vertical_door_line_collection in enumerate(vertical_door_line_collections):
                # # Calculate the new positions for top and bottom door lines
                # top_door_y = 0.5 + constants.VERTICAL_WALL_HALF_WIDTH
                # bottom_door_y = top_door_y - scaled_doors[idx]

                # # Update the data for the vertical door LineCollection
                # vertical_door_line_collection.set_segments([[(constants.VERTICAL_WALL_POSITION, top_door_y),
                                                             # (constants.VERTICAL_WALL_POSITION, bottom_door_y)]])        


 
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.time.now, interval=20)

        if not os.path.exists(constants.SAVE_PATH_GIFF): os.makedirs(constants.SAVE_PATH_GIFF)
        filename = os.path.join(constants.SAVE_PATH_GIFF, f'{self.experiment_name}_ep-{self.time.n_episodes}.gif')
        ani.save(filename=filename, writer='pillow')
        log.info(f"Env is rendered and gif animation is saved to {filename}")

        if self.save_next_episode_anim:
            self.save_next_episode_anim = False
            self.draw = False

    def close(self):
        pass
# # %%
# e = EvacuationEnv(number_of_pedestrians=100)

# e.reset()
# e.step([1, 0])

# for i in range(300):
#     e.step([np.sin(i*0.1), np.cos(i*0.1)])
# e.save_animation()
# e.render()
# # %%
