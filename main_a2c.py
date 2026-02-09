import os
import numpy as np
from gymnasium.wrappers import FrameStack, FlattenObservation
from stable_baselines3 import PPO

from src.env import EvacuationEnv, RelativePosition, constants
from src import params
from src.utils import get_experiment_name, parse_args


from src.env import EvacuationEnv, RelativePosition, constants
from src import params
from src.utils import get_experiment_name, parse_args



#from agents import RandomAgent, RotatingAgent
import numpy as np
import os
from stable_baselines3 import PPO
print('starting the experiment')
def setup_env(args, experiment_name):
    env = EvacuationEnv(
        experiment_name=experiment_name,
        number_of_pedestrians=args.number_of_pedestrians,
        enslaving_degree=args.enslaving_degree, 
        width=args.width,
        height=args.height,
        step_size=args.step_size,
        noise_coef=args.noise_coef,
        is_termination_agent_wall_collision=args.is_termination_agent_wall_collision,
        is_new_exiting_reward=args.is_new_exiting_reward,
        is_new_followers_reward=args.is_new_followers_reward,
        intrinsic_reward_coef=args.intrinsic_reward_coef,
        max_timesteps=args.max_timesteps,
        n_episodes=args.n_episodes,
        n_timesteps=args.n_timesteps,
        enabled_gravity_embedding=args.enabled_gravity_embedding,
        #enabled_gravity_and_speed_embedding=args.enabled_gravity_and_speed_embedding,
        alpha=args.alpha,
        verbose=args.verbose,
        render_mode=None,
        draw=args.draw
    )
    
    # enable relative positions of pedestrians and exit
    if args.use_relative_positions:
        env = RelativePosition(env)

    # Dict[str, Box] observation to Box observation
    env = FlattenObservation(env)
    
    # add stacked (previos observations)
    if args.num_obs_stacks != 1:
        env = FrameStack(env, num_stack=args.num_obs_stacks)
        ## TODO may be we should filter some stacks here
        env = FlattenObservation(env)

    return env
def setup_model(args, env): 
    
    if args.origin == 'ppo':
        model = PPO(
            "MlpPolicy", 
            env, verbose=1, 
            tensorboard_log=params.SAVE_PATH_TBLOGS,
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma
        )
    elif args.origin == 'sac':
        model = PPO(
            "MlpPolicy", 
            env, verbose=1, 
            tensorboard_log=params.SAVE_PATH_TBLOGS,
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma
        )
    else:
        raise NotImplementedError
    return model

if __name__ == "__main__":
    args = parse_args()
    experiment_name = get_experiment_name(args)
    env = setup_env(args, experiment_name)



    model = setup_model(args, env)
    model = model.load(os.path.join(params.SAVE_PATH_MODELS, f"test_n-100_lr-0.0003_gamma-0.99_s-gra_a-2_ss-0.01_vr-0.05_11-May-03-23-38.zip"))
    obs, _ = env.reset()
    for i in range(2000):
        action = model.predict(obs)
        print(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        if reward != 0:
            print(' = ', reward)

    env.save_animation()
    env.render()

print('code completed succesfully')
