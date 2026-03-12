"""
Shared utility functions for data loading and environment creation.
This module eliminates code duplication across training and evaluation scripts.
"""

import os
import time
import shutil
import cv2
import pickle
import hashlib
from src.NetworkHeatmap import NetworkHeatmap
from src.gcn_env import GCNJunctionEnv


def get_cache_dir(base_dir, net_file):
    """Get the cache directory for a specific city (inside the city's data folder)"""
    # Extract the directory containing the network file (the city folder)
    city_data_dir = os.path.dirname(os.path.join(base_dir, net_file))
    cache_dir = os.path.join(city_data_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_timestep_cache_path(cache_dir, timestep, net_file, data_file):
    """Generate cache filename for a specific timestep"""
    # Create a hash based on the source files to detect if data changes
    source_hash = hashlib.md5(f"{net_file}_{data_file}".encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f'timestep_{timestep}_{source_hash}.pkl')
    return cache_file


def save_timestep_cache(cache_path, timestep_data_entry):
    """
    Save a single timestep's data to cache.
    Excludes the 'net' object as it's not serializable and can be reloaded.
    """
    cache_data = {
        'bbox': timestep_data_entry['bbox'],
        'prepared_junctions': timestep_data_entry['prepared_junctions'],
        'heatmap_without_network': timestep_data_entry['heatmap_without_network'],
        'heatmap_with_network': timestep_data_entry['heatmap_with_network']
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)


def load_timestep_cache(cache_path, net):
    """
    Load a single timestep's data from cache and add the net object.
    """
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    # Add the net object back
    cache_data['net'] = net
    return cache_data
    
    

def preload_timestep_data(base_dir, net_file, data_file, timesteps_to_load, use_cache=True):
    """
    Pre-load all timestep data to avoid regenerating heatmaps.
    Uses caching to save/load previously generated data.
    
    Parameters:
    -----------
    base_dir : str
        Project root directory
    net_file : str
        Relative path to network file (e.g., 'data/Toronto/osm.net.xml')
    data_file : str
        Relative path to data file (e.g., 'data/Toronto/fcd.csv')
    timesteps_to_load : list
        List of timesteps to load
    use_cache : bool, default=True
        Whether to use caching for timestep data
        
    Returns:
    --------
    dict
        Dictionary mapping timestep -> environment data
    """
    print(f"\n{'=' * 80}")
    print("PRE-LOADING TIMESTEP DATA")
    print(f"{'=' * 80}")
    print(f"Requested {len(timesteps_to_load)} timesteps: {timesteps_to_load}")

    # Extract city name from path
    city = os.path.basename(os.path.dirname(os.path.join(base_dir, net_file)))
    
    # Setup cache
    cache_dir = get_cache_dir(base_dir, net_file) if use_cache else None
    
    # Construct full paths
    net_file_path = os.path.join(base_dir, net_file)
    data_file_path = os.path.join(base_dir, data_file)
    
    # Load network once (shared across all timesteps)
    print(f"\nLoading network from {net_file_path}...")
    net = NetworkHeatmap.prepare_data(net_file_path, data_file_path, timesteps_to_load[0])[0]
    
    timestep_data = {}
    timesteps_to_generate = []
    
    # Check which timesteps are cached
    if use_cache:
        print(f"\nChecking cache in {cache_dir}...")
        for timestep in timesteps_to_load:
            cache_path = get_timestep_cache_path(cache_dir, timestep, net_file, data_file)
            
            if os.path.exists(cache_path):
                try:
                    timestep_data[timestep] = load_timestep_cache(cache_path, net)
                    print(f"  Loaded timestep {timestep} from cache")
                except Exception as e:
                    print(f"  Failed to load cache for timestep {timestep}: {e}")
                    timesteps_to_generate.append(timestep)
            else:
                timesteps_to_generate.append(timestep)
    else:
        timesteps_to_generate = timesteps_to_load

    # Generate missing timesteps
    if timesteps_to_generate:
        print(f"\nGenerating {len(timesteps_to_generate)} timesteps: {timesteps_to_generate}")
        
        for timestep in timesteps_to_generate:
            print(f"\nProcessing timestep {timestep}...")
            start_time = time.time()

            # Load network and prepare data
            net, bbox, z, extent = NetworkHeatmap.prepare_data(net_file_path, data_file_path, timestep)

            # Create temporary directory for this timestep
            temp_dir = f"temp_timestep_{timestep}"
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Create heatmaps
                NetworkHeatmap.create_heatmap(
                    net, z, extent, show_network=False,
                    output_filename=os.path.join(temp_dir, "heatmap_without.png")
                )

                NetworkHeatmap.create_heatmap(
                    net, z, extent, show_network=True,
                    output_filename=os.path.join(temp_dir, "heatmap_with.png")
                )

                # Load heatmaps
                heatmap_without_network = cv2.imread(os.path.join(temp_dir, "heatmap_without.png"))
                heatmap_with_network = cv2.imread(os.path.join(temp_dir, "heatmap_with.png"))

                # Prepare junctions
                prepared_junctions = NetworkHeatmap.prepare_junctions(net, bbox, heatmap_without_network)

                # Store all data for this timestep
                timestep_data[timestep] = {
                    'net': net,
                    'bbox': bbox,
                    'prepared_junctions': prepared_junctions,
                    'heatmap_without_network': heatmap_without_network,
                    'heatmap_with_network': heatmap_with_network
                }

                # Save to cache
                if use_cache:
                    cache_path = get_timestep_cache_path(cache_dir, timestep, net_file, data_file)
                    save_timestep_cache(cache_path, timestep_data[timestep])
                    print(f"  Cached timestep {timestep}")

                elapsed = time.time() - start_time
                print(f"  Timestep {timestep} loaded in {elapsed:.2f}s")

            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    print(f"\n All {len(timesteps_to_load)} timesteps ready!")
    print(f"  - Loaded from cache: {len(timesteps_to_load) - len(timesteps_to_generate)}")
    print(f"  - Newly generated: {len(timesteps_to_generate)}")
    
    return timestep_data

def create_env_from_timestep(timestep_data, timestep, lambda_reg, congestion_threshold, device):
    """
    Create an environment from pre-loaded timestep data.
    
    Parameters:
    -----------
    timestep_data : dict
        Pre-loaded timestep data from preload_timestep_data()
    timestep : int
        Timestep to create environment for
    lambda_reg : float
        Lambda regularization parameter
    congestion_threshold : float
        Congestion threshold parameter
    device : torch.device
        Device to use (cuda or cpu)
        
    Returns:
    --------
    GCNJunctionEnv
        Initialized environment
    """
    data = timestep_data[timestep]

    env = GCNJunctionEnv(
        net=data['net'],
        prepared_junctions=data['prepared_junctions'],
        heatmap_without_network=data['heatmap_without_network'],
        heatmap_with_network=data['heatmap_with_network'],
        lambda_reg=lambda_reg,
        congestion_threshold=congestion_threshold,
        device=device
    )

    return env