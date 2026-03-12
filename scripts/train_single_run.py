# train_single_run.py
# Training script for a single agent that generalizes across multiple traffic timesteps

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import time
import yaml
from collections import defaultdict
import shutil
import csv


# --- PATH SETUP START ---
def get_project_root():
    """
    Returns the project root directory.
    Assumes this script is located in project_root/scripts/
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)  # Go up one level to project_root


# Setup paths BEFORE importing project modules
PROJECT_ROOT = get_project_root()
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
# --- PATH SETUP END ---

# Import project modules
from src.NetworkHeatmap import NetworkHeatmap
from src.gcn_env import GCNJunctionEnv
from src.gcn_agent import GCNPPOAgent
from src.data_utils import preload_timestep_data, create_env_from_timestep

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_agent_all_timesteps(timestep_data, timesteps_to_evaluate, agent, lambda_reg, congestion_threshold, device,
                                 max_steps):
    """
    Evaluate agent on all timesteps.
    Returns dictionary mapping timestep -> (reward, steps)
    """
    results = {}

    for timestep in timesteps_to_evaluate:
        env = create_env_from_timestep(timestep_data, timestep, lambda_reg, congestion_threshold, device)

        # Evaluate agent on a single timestep
        state, _ = env.reset()
        gcn_state = env.get_gcn_observation()

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action, _, _ = agent.select_action(gcn_state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            gcn_state = env.get_gcn_observation()
            total_reward += reward
            steps += 1

            if info.get("finished_by_action", False):
                break

        results[timestep] = (total_reward, steps)
        env.close()

    return results
    

def save_evaluation_data(evaluation_history, output_dir, prefix=''):
    """Save evaluation data to CSV for further analysis"""
    prefix_str = f'{prefix}_' if prefix else ''
    filename = os.path.join(output_dir, f'{prefix_str}evaluation_data.csv')

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestep', 'Episode', 'Reward', 'Steps'])

        for timestep in sorted(evaluation_history.keys()):
            for episode, reward, steps in evaluation_history[timestep]:
                writer.writerow([timestep, episode, reward, steps])

    print(f"Evaluation data saved to {filename}")


def main():
    # 1. Setup Argparse for dynamic config and city selection
    parser = argparse.ArgumentParser(description="Run script with a specific configuration.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--city', type=str,
                        help='Name of the city (folder in data/). Overrides default_city in config.')
    args = parser.parse_args()

    # 2. Use PROJECT_ROOT for paths
    config_path = os.path.join(PROJECT_ROOT, args.config)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Set random seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Set random seed to {seed}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- DYNAMIC PATH CONSTRUCTION ---
    # Determine city: Command line arg > Config default > "Toronto" fallback
    city = args.city
    data_root = config['paths']['data_root']

    # --- CITY CONFIG OVERRIDE START ---
    # Construct path to city_config.yaml
    city_config_path = os.path.join(PROJECT_ROOT, data_root, city, "city_config.yaml")

    # If the file exists, load it and update the main config
    if os.path.exists(city_config_path):
        print(f"Found city-specific config: {city_config_path}")
        city_cfg = load_config(city_config_path)

        # Override environment settings (train_timesteps, eval_timesteps)
        if 'environment' in city_cfg:
            config['environment'].update(city_cfg['environment'])
            print(f"Overriding environment timesteps from {city}")
    # --- CITY CONFIG OVERRIDE END ---

    # Construct relative paths to be passed to functions
    # e.g., "data/Toronto/osm.net.xml"
    net_file_rel = os.path.join(data_root, city, config['paths']['net_filename'])
    data_file_rel = os.path.join(data_root, city, config['paths']['data_filename'])

    # Output directory includes the city name
    # e.g., "outputs/Toronto/"
    output_dir = os.path.join(PROJECT_ROOT, config['paths']['output_root'], city)

    print(f"City: {city}")
    print(f"Network File: {net_file_rel}")
    print(f"Data File: {data_file_rel}")
    print(f"Output Directory: {output_dir}")

    train_timesteps = config['environment']['train_timesteps']
    eval_timesteps = config['environment']['eval_timesteps']
    lambda_reg = config['environment']['lambda_reg']
    congestion_threshold = config['environment']['congestion_threshold']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Pre-load all required timesteps (training and evaluation)
    start_time = time.time()
    all_timesteps_to_load = sorted(list(set(train_timesteps + eval_timesteps)))

    # Pass PROJECT_ROOT to the preload function so it can find the data
    timestep_data = preload_timestep_data(PROJECT_ROOT, net_file_rel, data_file_rel, all_timesteps_to_load)

    preload_time = time.time() - start_time
    print(f"\nPre-loading completed in {preload_time:.2f}s ({preload_time / 60:.2f} min)")

    # Create initial environment (use first training timestep to initialize agent)
    initial_timestep = train_timesteps[0]
    env = create_env_from_timestep(timestep_data, initial_timestep, lambda_reg, congestion_threshold, device)

    # Create agent
    print(f"\n{'=' * 80}")
    print("INITIALIZING AGENT")
    print(f"{'=' * 80}")
    agent = GCNPPOAgent(env=env, config=config, device=device)
    print(f"Agent initialized")

    # Training parameters
    num_episodes = config['training']['num_episodes']
    max_steps = config['training']['max_steps_per_episode']
    update_freq = config['training']['update_frequency']
    evaluation_interval = config['training']['evaluation_interval']
    log_interval = config['training']['log_interval']

    # Training tracking
    episode_rewards = []
    train_evaluation_history = defaultdict(list)  # {timestep: [(episode, reward, steps), ...]}
    test_evaluation_history = defaultdict(list)  # {timestep: [(episode, reward, steps), ...]}

    print(f"\n{'=' * 80}")
    print("STARTING TRAINING")
    print(f"{'=' * 80}")
    print(f"Total episodes: {num_episodes}")
    print(f"Train timesteps: {train_timesteps}")
    print(f"Evaluation timesteps: {eval_timesteps}")
    print(f"Lambda: {lambda_reg}")
    print(f"Evaluation interval: {evaluation_interval}")
    print(f"{'=' * 80}\n")

    training_start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # Randomly select a training timestep for this episode
        current_timestep = np.random.choice(train_timesteps)

        # Create environment for this timestep
        env = create_env_from_timestep(timestep_data, current_timestep, lambda_reg, congestion_threshold, device)

        # Reset environment
        state, _ = env.reset()
        gcn_state = env.get_gcn_observation()

        episode_reward = 0
        steps = 0
        done = False

        # Training episode
        while not done and steps < max_steps:
            action, log_prob, value = agent.select_action(gcn_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_gcn_state = env.get_gcn_observation()

            mask = 0.0 if done else 1.0
            agent.store_transition(
                gcn_state, action, reward, value, log_prob, mask,
                gcn_state['action_mask']
            )

            gcn_state = next_gcn_state
            episode_reward += reward
            steps += 1

            # Update policy if buffer is full
            if len(agent.states) >= update_freq:
                agent.update_policy()

        # Update policy at end of episode if needed
        if len(agent.states) > 0:
            agent.update_policy()

        # Update learning rate schedule
        agent.update_schedules(episode, num_episodes, config)

        # Store episode reward
        episode_rewards.append(episode_reward)

        # Close environment
        env.close()

        # Logging
        if episode % log_interval == 0:
            elapsed = time.time() - training_start_time
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward (last {log_interval}): {avg_reward:.4f} | "
                  f"Current Timestep: {current_timestep} | "
                  f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        # Evaluation on both training and test sets
        if episode % evaluation_interval == 0:
            print(f"\n{'=' * 60}")
            print(f"EVALUATION AT EPISODE {episode}")
            print(f"{'=' * 60}")

            eval_start_time = time.time()

            # Evaluate on training set - PASS max_steps
            train_eval_results = evaluate_agent_all_timesteps(
                timestep_data, train_timesteps, agent, lambda_reg, congestion_threshold, device,
                max_steps=max_steps
            )

            # Evaluate on test set - PASS max_steps
            test_eval_results = evaluate_agent_all_timesteps(
                timestep_data, eval_timesteps, agent, lambda_reg, congestion_threshold, device,
                max_steps=max_steps
            )

            eval_time = time.time() - eval_start_time

            # Store results for training set
            for timestep, (reward, steps) in train_eval_results.items():
                train_evaluation_history[timestep].append((episode, reward, steps))

            # Store results for test set
            for timestep, (reward, steps) in test_eval_results.items():
                test_evaluation_history[timestep].append((episode, reward, steps))

            # Print evaluation summary
            print(f"\nEvaluation Results (completed in {eval_time:.1f}s):")

            print(f"\n  Training Set:")
            for timestep in train_timesteps:
                reward, steps = train_eval_results[timestep]
                print(f"    Timestep {timestep}: Reward = {reward:.4f}, Steps = {steps}")

            print(f"\n  Test Set:")
            for timestep in eval_timesteps:
                reward, steps = test_eval_results[timestep]
                print(f"    Timestep {timestep}: Reward = {reward:.4f}, Steps = {steps}")

            print(f"{'=' * 60}\n")

    total_training_time = time.time() - training_start_time

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETED!")
    print(f"{'=' * 80}")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time / 60:.2f} min)")

    # Generate outputs based on the evaluation set
    print(f"\n{'=' * 80}")
    print("GENERATING OUTPUTS")
    print(f"{'=' * 80}")

    # Save evaluation data for both sets
    save_evaluation_data(train_evaluation_history, output_dir, prefix='train')
    save_evaluation_data(test_evaluation_history, output_dir, prefix='test')

    # Save final model
    model_name = config['paths'].get('model_filename', 'final_model.pt')
    model_path = os.path.join(output_dir, model_name)
    agent.save_models(model_path)
    print(f"Final model saved to {model_path}")

    print(f"\n{'=' * 80}")
    print("ALL DONE!")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print("Generated files:")
    print("  - train_evaluation_data.csv")
    print("  - test_evaluation_data.csv")
    print(f"  - {model_name}")
    print("  - train_set/convexhull/...")
    print("  - train_set/convexhull_binary/...")
    print("  - evaluation_set/convexhull/...")
    print("  - evaluation_set/convexhull_binary/...")


if __name__ == "__main__":
    main()