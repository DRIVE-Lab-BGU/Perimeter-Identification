import os
import sys
import torch
import cv2
import yaml
import time
import argparse

# --- PATH SETUP START ---
def get_project_root():
    """
    Returns the project root directory.
    Assumes this script is located in project_root/scripts/
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


PROJECT_ROOT = get_project_root()
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
# --- PATH SETUP END ---

# Import project modules
from src.gcn_agent import GCNPPOAgent
from src.data_utils import preload_timestep_data, create_env_from_timestep

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_convexhull_images(timestep_data, timesteps, agent, lambda_reg, congestion_threshold, device,
                               output_dir, subfolder, max_steps):
    """
    Generate and save convex hull images for specified timesteps.
    Saves both colored and binary grayscale versions.
    
    Saves to:
    - outputs/{City}/{subfolder}/convexhull/
    - outputs/{City}/{subfolder}/convexhull_binary/
    """
    print(f"\n{'=' * 80}")
    print(f"GENERATING CONVEX HULL IMAGES ({subfolder.upper()})")
    print(f"{'=' * 80}")

    # Define paths
    base_set_path = os.path.join(output_dir, subfolder)
    path_rgb = os.path.join(base_set_path, "convexhull")
    path_binary = os.path.join(base_set_path, "convexhull_binary")

    os.makedirs(path_rgb, exist_ok=True)
    os.makedirs(path_binary, exist_ok=True)

    print(f"Generating images for {len(timesteps)} timesteps...")

    for timestep in timesteps:
        print(f"\nProcessing timestep {timestep}...")

        # 1. Setup Environment
        env = create_env_from_timestep(timestep_data, timestep, lambda_reg, congestion_threshold, device)
        state, _ = env.reset()
        gcn_state = env.get_gcn_observation()

        done = False
        steps = 0
        total_reward = 0

        # 2. Run greedy policy
        while not done and steps < max_steps:
            action, _, _ = agent.select_action(gcn_state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            gcn_state = env.get_gcn_observation()
            total_reward += reward
            steps += 1

            if info.get("finished_by_action", False):
                break

        # 3. Render and save colored version
        final_image_colored = env.render(mode='rgb_array')
        if final_image_colored is not None:
            save_path = os.path.join(path_rgb, f'convexhull_timestep_{timestep}.png')
            cv2.imwrite(save_path, final_image_colored)
            print(f"  ✓ Colored convex hull saved to {save_path}")

        # 4. Render and save binary grayscale version
        final_image_binary = env.render(mode='binary_grayscale')
        if final_image_binary is not None:
            save_path = os.path.join(path_binary, f'convexhull_binary_timestep_{timestep}.png')
            cv2.imwrite(save_path, final_image_binary)
            print(f"  ✓ Binary grayscale saved to {save_path}")

        env.close()

    print(f"\n✓ All convex hull images generated in {base_set_path}")


def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--city', type=str, help='City name. Overrides config default.')
    parser.add_argument('--model-city', type=str, help='City to load model from (if different from target city)')
    parser.add_argument('--set', type=str, default='eval', choices=['train', 'eval', 'all'],
                        help="Which timesteps to process: 'train', 'eval', or 'all'")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    print(f"✓ Loaded configuration from {config_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # --- DYNAMIC PATH CONSTRUCTION ---
    city = args.city if args.city else config['paths'].get('default_city', 'Toronto')
    data_root = config['paths']['data_root']

    # --- CITY CONFIG OVERRIDE START ---
    # Construct path to city_config.yaml
    city_config_path = os.path.join(PROJECT_ROOT, data_root, city, "city_config.yaml")

    # If the file exists, load it and update the main config
    if os.path.exists(city_config_path):
        print(f"✓ Found city-specific config: {city_config_path}")
        city_cfg = load_config(city_config_path)

        # Override environment settings (train_timesteps, eval_timesteps)
        if 'environment' in city_cfg:
            config['environment'].update(city_cfg['environment'])
            print(f"✓ Overriding environment timesteps from {city}")
    # --- CITY CONFIG OVERRIDE END ---

    # Construct relative paths
    net_file_rel = os.path.join(data_root, city, config['paths']['net_filename'])
    data_file_rel = os.path.join(data_root, city, config['paths']['data_filename'])

    # Determine output directory name
    model_city = args.model_city if args.model_city else city
    if model_city != city:
        output_city_name = f"{city}_checked_on_{model_city}'s_model"
    else:
        output_city_name = city

    # Output directory
    output_dir = os.path.join(PROJECT_ROOT, config['paths']['output_root'], output_city_name)

    print(f"✓ City: {city}")
    print(f"✓ Network File: {net_file_rel}")
    print(f"✓ Data File: {data_file_rel}")
    print(f"✓ Output Directory: {output_dir}")

    # Get timesteps based on what set we're generating
    eval_timesteps = config['environment']['eval_timesteps']
    lambda_reg = config['environment']['lambda_reg']
    congestion_threshold = config['environment']['congestion_threshold']
    max_steps = config['training']['max_steps_per_episode']

    # Determine which timesteps to process and which sets to generate
    timesteps_to_process = []
    sets_to_generate = []

    if args.set in ['eval', 'all']:
        timesteps_to_process.extend(eval_timesteps)
        sets_to_generate.append(('evaluation_set', eval_timesteps))

    if args.set in ['train', 'all']:
        train_timesteps = config['environment'].get('train_timesteps', [])
        if not train_timesteps:
            print("⚠️  Warning: No train_timesteps found in config, skipping train set generation")
        else:
            timesteps_to_process.extend(train_timesteps)
            sets_to_generate.append(('train_set', train_timesteps))

    # Remove duplicates and sort
    timesteps_to_process = sorted(list(set(timesteps_to_process)))

    if not timesteps_to_process:
        print("❌ ERROR: No timesteps to process!")
        return

    print(f"✓ Processing {len(timesteps_to_process)} unique timesteps for set(s): {args.set}")

    # Parameters for image generation
    model_filename = config['paths'].get('model_filename', 'final_model.pt')

    # Determine model path
    model_dir = os.path.join(PROJECT_ROOT, config['paths']['output_root'], model_city)
    model_path = os.path.join(model_dir, model_filename)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("Please train the model first using train_single_run.py")
        return

    print(f"\n✓ Found trained model at {model_path}")

    # Pre-load timestep data
    print(f"\nPre-loading {len(timesteps_to_process)} timesteps...")
    start_time = time.time()
    timestep_data = preload_timestep_data(PROJECT_ROOT, net_file_rel, data_file_rel, timesteps_to_process)
    preload_time = time.time() - start_time
    print(f"\n✓ Pre-loading completed in {preload_time:.2f}s ({preload_time / 60:.2f} min)")

    # Create initial environment (use first available timestep to initialize agent)
    initial_timestep = timesteps_to_process[0]
    env = create_env_from_timestep(timestep_data, initial_timestep, lambda_reg, congestion_threshold, device)

    # Create agent
    print(f"\n{'=' * 80}")
    print("INITIALIZING AGENT")
    print(f"{'=' * 80}")
    agent = GCNPPOAgent(env=env, config=config, device=device)

    # Load trained model
    print(f"Loading model from {model_path}...")
    agent.load_models(model_path)
    print(f"✓ Model loaded successfully")

    # Generate images for all requested sets
    for subfolder, timesteps in sets_to_generate:
        generate_convexhull_images(
            timestep_data=timestep_data,
            timesteps=timesteps,
            agent=agent,
            lambda_reg=lambda_reg,
            congestion_threshold=congestion_threshold,
            device=device,
            output_dir=output_dir,
            subfolder=subfolder,
            max_steps=max_steps
        )

    print(f"\n{'=' * 80}")
    print("DONE!")
    print(f"{'=' * 80}")
    print(f"Convex hull images saved to: {output_dir}")
    
    for subfolder, timesteps in sets_to_generate:
        print(f"  - {subfolder}/convexhull/")
        print(f"  - {subfolder}/convexhull_binary/")
    
    print(f"Generated {len(timesteps_to_process)} convex hull image pairs across {len(sets_to_generate)} set(s)")


if __name__ == "__main__":
    main()