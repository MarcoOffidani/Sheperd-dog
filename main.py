import os
import csv
import re
import random
import numpy as np
from gymnasium.wrappers import FrameStack, FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from src.optional_wandb import wandb
#print("something")
from src.env import EvacuationEnv, RelativePosition, constants
#print("something2")
from src import params
#print("something3")
from src.utils import get_experiment_name, parse_args
#print("something4")

RUN_ID_PATTERN = re.compile(r"\d{18}_[0-9a-f]{6}")


def infer_parent_run_id(load_model_path):
    if not load_model_path:
        return None

    match = RUN_ID_PATTERN.search(load_model_path)
    if match:
        return match.group(0)

    runs_csv = os.path.join(params.SAVE_PATH_OUTPUT, 'runs.csv')
    if not os.path.exists(runs_csv):
        return None

    parent_run_id = None
    best_match_length = 0
    normalized_model_path = load_model_path.replace('\\', '/').replace(os.sep, '/')
    with open(runs_csv, newline='') as f:
        for row in csv.DictReader(f):
            run_id = row.get('run_id')
            experiment_name = row.get('experiment_name')
            if not run_id or not experiment_name:
                continue
            if experiment_name in normalized_model_path and len(experiment_name) >= best_match_length:
                parent_run_id = run_id
                best_match_length = len(experiment_name)

    return parent_run_id


def build_run_metadata(args, experiment_name):
    return {
        "experiment_name": experiment_name,
        "parent_run_id": infer_parent_run_id(args.load_model_path),
        "origin": args.origin,
        "learn_timesteps": args.learn_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "device": args.device,
        "seed": args.seed,
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


def seed_everything(seed, using_cuda=False):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed, using_cuda=using_cuda)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def configure_output_paths(args, experiment_name):
    output_dir = args.output_dir or os.path.join("saved_data", "runs", experiment_name)
    params.SAVE_PATH_OUTPUT = output_dir
    params.SAVE_PATH_MODELS = os.path.join(output_dir, "models")
    params.SAVE_PATH_TBLOGS = os.path.join(output_dir, "tb-logs")
    params.SAVE_PATH_CHECKPOINTS = args.checkpoint_dir or os.path.join(output_dir, "checkpoints")
    constants.SAVE_PATH_LOGS = args.log_dir or os.path.join(output_dir, "logs")
    constants.SAVE_PATH_PNG = os.path.join(output_dir, "png")
    constants.SAVE_PATH_GIFF = os.path.join(output_dir, "giff")
    for path in (
        params.SAVE_PATH_OUTPUT,
        params.SAVE_PATH_MODELS,
        params.SAVE_PATH_TBLOGS,
        params.SAVE_PATH_CHECKPOINTS,
        constants.SAVE_PATH_LOGS,
        constants.SAVE_PATH_PNG,
        constants.SAVE_PATH_GIFF,
    ):
        os.makedirs(path, exist_ok=True)
    return output_dir


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
        n_episodes=0 if args.eval_only else args.n_episodes,
        n_timesteps=args.n_timesteps,
        enabled_gravity_embedding=args.enabled_gravity_embedding,
        enabled_gravity_and_speed_embedding=args.enabled_gravity_embedding_speed,
        alpha=args.alpha,
        seed=args.seed,
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
            gamma=args.gamma,
            seed=args.seed
        )
    elif args.origin == 'sac':
        model = PPO(
            "MlpPolicy", 
            env, verbose=1, 
            tensorboard_log=params.SAVE_PATH_TBLOGS,
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            seed=args.seed
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

def evaluate_model(model, env, n_episodes):
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

if __name__ == "__main__":
    #print("first line")
    args = parse_args()

    experiment_name = get_experiment_name(args)
    configure_output_paths(args, experiment_name)
    seed_everything(args.seed, using_cuda=args.device == "cuda")

    #setup_wandb(args, experiment_name)
    env = setup_env(args, experiment_name)

    model = setup_model(args, env)
    #model.load('11may.zip') #remove .for training
    #model.learning_rate = 0.0
    if args.eval_only:
        try:
            evaluate_model(model, env, args.n_episodes)
        finally:
            env.close()
        raise SystemExit(0)

    checkpoint_dir = params.SAVE_PATH_CHECKPOINTS
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
