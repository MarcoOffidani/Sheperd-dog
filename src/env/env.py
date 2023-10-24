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

        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
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
            raise Exception
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
            raise Exception
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

        def update(i):

            agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            following_zone_plots.set_center(agent_coordinates)

            for status in Status.all():
                selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])
                 
            # agent_position_plot.set_xdata(agent_coordinates[0])
            # agent_position_plot.set_ydata(agent_coordinates[1])
            agent_position_plot.set_data(agent_coordinates)

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
