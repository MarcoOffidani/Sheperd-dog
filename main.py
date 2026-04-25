import os
import numpy as np
from gymnasium.wrappers import FrameStack, FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
#print("something")
from src.env import EvacuationEnv, RelativePosition, constants
#print("something2")
from src import params
#print("something3")
from src.utils import get_experiment_name, parse_args
#print("something4")

def build_run_metadata(args, experiment_name):
    return {
        "experiment_name": experiment_name,
        "origin": args.origin,
        "learn_timesteps": args.learn_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "device": args.device,
        "number_of_pedestrians": args.number_of_pedestrians,
        "width": args.width,
        "height": args.height,
        "step_size": args.step_size,
        "noise_coef": args.noise_coef,
        "num_obs_stacks": args.num_obs_stacks,
        "use_relative_positions": args.use_relative_positions,
        "deactivate_walls": args.deactivate_walls,
        "enslaving_degree": args.enslaving_degree,
        "is_new_exiting_reward": args.is_new_exiting_reward,
        "is_new_followers_reward": args.is_new_followers_reward,
        "intrinsic_reward_coef": args.intrinsic_reward_coef,
        "is_termination_agent_wall_collision": args.is_termination_agent_wall_collision,
        "init_reward_each_step": args.init_reward_each_step,
        "max_timesteps": args.max_timesteps,
        "n_episodes": args.n_episodes,
        "n_timesteps": args.n_timesteps,
        "enabled_gravity_embedding": args.enabled_gravity_embedding,
        "enabled_gravity_embedding_speed": args.enabled_gravity_embedding_speed,
        "alpha": args.alpha,
        "checkpoint_frequency_timesteps": args.checkpoint_frequency_timesteps,
        "load_model_path": args.load_model_path,
    }


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
        enabled_gravity_and_speed_embedding=args.enabled_gravity_embedding_speed,
        alpha=args.alpha,
        run_metadata=build_run_metadata(args, experiment_name),
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
    if args.load_model_path:
        return PPO.load(args.load_model_path, env=env, device=args.device)
    
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

def setup_wandb(args, experiment_name):
    config_args = vars(args)
    # config_env = {key : value for key, value in constants.__dict__.items() if key[0] != '_'}
    # config_model = {key : value for key, value in params.__dict__.items() if key[0] != '_'}
    # save_config = dict(config_args, **config_env, **config_model)
    from src.env.env import SwitchDistances as sd
    config_switch_distances = {k : vars(sd)[k] for k in sd.__annotations__.keys()}
    save_config = dict(config_args, **config_switch_distances)

    wandb.init(
        project="evacuation",
        name=args.exp_name,
        notes=experiment_name,
        config=save_config
    )

if __name__ == "__main__":
    #print("first line")
    args = parse_args()

    experiment_name = get_experiment_name(args)

    #setup_wandb(args, experiment_name)
    env = setup_env(args, experiment_name)

    model = setup_model(args, env)
    #model.load('11may.zip') #remove .for training
    #model.learning_rate = 0.0
    checkpoint_dir = os.path.join(params.SAVE_PATH_MODELS, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_frequency_timesteps,
        save_path=checkpoint_dir,
        name_prefix=experiment_name,
    )
    try:
        model.learn(
            args.learn_timesteps,
            tb_log_name=experiment_name,
            callback=checkpoint_callback,
            reset_num_timesteps=not bool(args.load_model_path),
        )
    finally:
        env.close()
 

    model.save(os.path.join(params.SAVE_PATH_MODELS, f"{experiment_name}.zip"))
