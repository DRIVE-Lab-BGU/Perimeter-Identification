import os
import sys
import numpy as np
import pandas as pd
import yaml
from collections import defaultdict
import torch
import argparse


# --- PATH SETUP START ---
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


PROJECT_ROOT = get_project_root()
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
# --- PATH SETUP END ---

from src.gcn_agent import GCNPPOAgent

# --- CONSTANTS ---
MIN_EDGE_LENGTH = 15.0  # Filter out short edges
TAU = 30.0  # Density threshold (veh/km)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_edge_density_and_lengths(net, data_file, timestep):
    """
    Calculate vehicle density and extract lengths.
    Filters out edges shorter than MIN_EDGE_LENGTH.
    """
    df = pd.read_csv(data_file, delimiter=";")
    df_filtered = df[df['timestep_time'] == timestep]

    raw_lengths = {}
    for edge in net.getEdges():
        raw_lengths[edge.getID()] = edge.getLength()

    edge_vehicle_count = defaultdict(int)
    for _, row in df_filtered.iterrows():
        lane_id = str(row['vehicle_lane'])
        if '_' in lane_id:
            edge_id = lane_id.rsplit('_', 1)[0]
        else:
            edge_id = lane_id

        if edge_id in raw_lengths:
            edge_vehicle_count[edge_id] += 1

    density_dict = {}
    length_dict = {}

    for edge_id, length in raw_lengths.items():
        if length < MIN_EDGE_LENGTH:
            continue

        length_dict[edge_id] = length

        if length > 0:
            val = (edge_vehicle_count[edge_id] / length) * 1000
            density_dict[edge_id] = round(val, 8)
        else:
            density_dict[edge_id] = 0.0

    return density_dict, length_dict


def export_edge_details_to_csv(net, data_file, timestep, output_dir):
    """
    Exports edge details to CSV (filtered > 15m).
    """
    df = pd.read_csv(data_file, delimiter=";")
    df_filtered = df[df['timestep_time'] == timestep]

    edge_data = []

    for edge in net.getEdges():
        edge_length_m = edge.getLength()
        if edge_length_m < MIN_EDGE_LENGTH:
            continue

        edge_id = edge.getID()
        vehicle_count = 0
        for _, row in df_filtered.iterrows():
            lane_id = str(row['vehicle_lane'])
            current_edge_id = lane_id.rsplit('_', 1)[0] if '_' in lane_id else lane_id
            if current_edge_id == edge_id:
                vehicle_count += 1

        density = round((vehicle_count / edge_length_m) * 1000, 8)

        edge_data.append({
            'edge_id': edge_id,
            'vehicle_count': vehicle_count,
            'edge_length_m': edge_length_m,
            'density_veh_per_km': density
        })

    df_edges = pd.DataFrame(edge_data)
    if not df_edges.empty:
        df_edges = df_edges.sort_values('density_veh_per_km', ascending=False)

    csv_path = os.path.join(output_dir, f'edge_details_timestep_{timestep}.csv')
    df_edges.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return df_edges



def get_agent_partition(env, agent, max_steps):
    """
    Runs the agent ONE last time (deterministically) to define the perimeter
    for the partition calculation.
    inside_edges matches exactly the cyan roads drawn in render_final_perimeter.
    """
    state, _ = env.reset(seed=42)
    gcn_state = env.get_gcn_observation()

    done = False
    steps = 0
    while not done and steps < max_steps:
        action, _, _ = agent.select_action(gcn_state, greedy=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        gcn_state = env.get_gcn_observation()
        steps += 1
        if info.get("finished_by_action", False):
            break

    all_edges = set(e.getID() for e in env.net.getEdges())
    if env.convex_hull is None:
        return set(), all_edges

    # 1. Perimeter Nodes
    perimeter_nodes = set()
    for j_id in env.junction_dict.keys():
        if env.is_junction_important(j_id):
            perimeter_nodes.add(j_id)

    if not perimeter_nodes:
        return set(), all_edges

    # 2. Internal Nodes (BFS from centroid) — same as render_final_perimeter
    hull_coords = env.hull_points[env.convex_hull.vertices]
    centroid_x  = np.mean(hull_coords[:, 0])
    centroid_y  = np.mean(hull_coords[:, 1])

    start_node_id = None
    min_dist = float('inf')
    for j_id, j_data in env.junction_dict.items():
        if j_id in perimeter_nodes:
            continue
        dist = (j_data['pixel_x'] - centroid_x) ** 2 + (j_data['pixel_y'] - centroid_y) ** 2
        if dist < min_dist:
            min_dist = dist
            start_node_id = j_id

    internal_nodes = set()
    if start_node_id:
        queue   = [start_node_id]
        visited = {start_node_id}
        while queue:
            current_id = queue.pop(0)
            if current_id in perimeter_nodes:
                continue
            internal_nodes.add(current_id)
            if env.net.hasNode(current_id):
                node  = env.net.getNode(current_id)
                edges = list(node.getOutgoing()) + list(node.getIncoming())
                for edge in edges:
                    neighbor = edge.getToNode() if edge.getFromNode().getID() == current_id else edge.getFromNode()
                    neighbor_id = neighbor.getID()
                    if neighbor_id in env.junction_dict and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

    # 3. Classify Edges — mirrors render_final_perimeter exactly:
    #    at least one endpoint must be internal (not just perimeter-to-perimeter)
    cluster_nodes = perimeter_nodes.union(internal_nodes)
    inside_edges  = set()

    for edge in env.net.getEdges():
        from_id = edge.getFromNode().getID()
        to_id   = edge.getToNode().getID()

        from_internal = from_id in internal_nodes
        to_internal   = to_id   in internal_nodes
        from_in_cluster = from_id in cluster_nodes
        to_in_cluster   = to_id   in cluster_nodes

        # Same condition as render_final_perimeter's should_draw:
        # at least one side must be internal
        if (from_internal and to_in_cluster) or (to_internal and from_in_cluster):
            inside_edges.add(edge.getID())

    outside_edges = all_edges - inside_edges
    return inside_edges, outside_edges


def calculate_partition_stats(density_dict, inside_ids):
    """
    Calculates basic statistics for the partition:
    Variance of inside edges, and counts for inside/outside.
    """
    all_valid_ids = list(density_dict.keys())
    group_in_ids = [e for e in inside_ids if e in density_dict]
    group_out_ids = [e for e in all_valid_ids if e not in inside_ids]

    k_in = [density_dict[e] for e in group_in_ids]

    N_in = len(k_in)
    N_out = len(group_out_ids)

    var_in = np.var(k_in) if N_in > 0 else 0.0

    return var_in, N_in, N_out


def calculate_actual_congestion_counts(density_dict, tau=TAU):
    """
    Calculate how many roads are actually congested/uncongested based on density threshold.
    Returns counts of roads with density > tau and density <= tau.
    """
    congested_count = sum(1 for density in density_dict.values() if density > tau)
    uncongested_count = sum(1 for density in density_dict.values() if density <= tau)

    return congested_count, uncongested_count


def calculate_wwc(density_dict, length_dict, inside_ids, tau=TAU):
    """
    Calculates WWC (formerly V_v_Last): Length-Weighted Mean Squared Error.
    """
    all_valid_ids = list(density_dict.keys())
    group_in_ids = [e for e in inside_ids if e in density_dict]
    group_out_ids = [e for e in all_valid_ids if e not in inside_ids]

    N_in = len(group_in_ids)
    N_out = len(group_out_ids)

    # --- Inside Cost (Variance) ---
    vals_in = [density_dict[e] for e in group_in_ids]
    var_in = np.var(vals_in) if N_in > 0 else 0.0

    # --- WWC: Outside Cost (Length-Weighted MSE) ---
    weighted_sq_error_sum = 0.0
    total_length_out = 0.0

    for e in group_out_ids:
        w_e = density_dict[e]
        L_e = length_dict[e]
        total_length_out += L_e

        if w_e > tau:
            sq_error = (w_e - tau) ** 2
            weighted_sq_error_sum += L_e * sq_error

    if total_length_out > 0:
        penalty_out_wwc = weighted_sq_error_sum / total_length_out
    else:
        penalty_out_wwc = 0.0

    numerator_wwc = (N_in * var_in) + (N_out * penalty_out_wwc)
    WWC = numerator_wwc / (N_in + N_out) if (N_in + N_out) > 0 else 0.0

    return WWC


def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--city', type=str, help='City name. Overrides config default.')
    parser.add_argument('--model-city', type=str, help='City to load model from (if different from target city)')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # --- DYNAMIC PATH CONSTRUCTION ---
    city = args.city
    data_root = config['paths']['data_root']

    # --- CITY CONFIG OVERRIDE START ---
    city_config_path = os.path.join(PROJECT_ROOT, data_root, city, "city_config.yaml")

    if os.path.exists(city_config_path):
        print(f"✓ Found city-specific config: {city_config_path}")
        city_cfg = load_config(city_config_path)

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

    eval_timesteps = config['environment']['eval_timesteps']
    lambda_reg = config['environment']['lambda_reg']
    congestion_threshold = config['environment']['congestion_threshold']
    max_steps = config['training']['max_steps_per_episode']

    # Create metrics folder inside output directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    model_filename = config['paths'].get('model_filename', 'final_model.pt')

    model_city = args.model_city if args.model_city else city
    model_dir = os.path.join(PROJECT_ROOT, config['paths']['output_root'], model_city)
    model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_path):
        print(f" ERROR: Model not found at {model_path}")
        return

    from src.data_utils import preload_timestep_data, create_env_from_timestep

    print("\n--- Loading Data & Model ---")
    timestep_data = preload_timestep_data(PROJECT_ROOT, net_file_rel, data_file_rel, eval_timesteps)

    initial_env = create_env_from_timestep(timestep_data, eval_timesteps[0], lambda_reg, congestion_threshold, device)
    agent = GCNPPOAgent(env=initial_env, config=config, device=device)
    agent.load_models(model_path)

    results = []

    for timestep in eval_timesteps:
        print(f"\nProcessing Timestep: {timestep}")

        net = timestep_data[timestep]['net']

        data_file_path_full = os.path.join(PROJECT_ROOT, data_file_rel)

        # 1. Export Edge Details to metrics folder
        export_edge_details_to_csv(net, data_file_path_full, timestep, metrics_dir)
        density_dict, length_dict = calculate_edge_density_and_lengths(net, data_file_path_full, timestep)

        # 2. Setup Environment
        env = create_env_from_timestep(timestep_data, timestep, lambda_reg, congestion_threshold, device)

        # 3. Get Final Partition (Visual/Geometric result)
        inside_ids, outside_ids = get_agent_partition(env, agent, max_steps=max_steps)
        env.close()

        # 4. Calculate Metrics
        # Partition-based counts (what the agent decided)
        var_in, n_inside, n_outside = calculate_partition_stats(density_dict, inside_ids)

        # Density-based counts (ground truth)
        n_congested, n_uncongested = calculate_actual_congestion_counts(density_dict, tau=TAU)

        # WWC metric
        wwc = calculate_wwc(density_dict, length_dict, inside_ids, tau=TAU)

        results.append({
            'timestep': timestep,
            'WWC': wwc,
            'Inside_Variance': var_in,
            'N_inside': n_inside,
            'N_outside': n_outside,
            'N_congested': n_congested,
            'N_uncongested': n_uncongested
        })

        print(f"   WWC: {wwc:.2f}")
        print(f"   Partition: {n_inside} inside, {n_outside} outside")
        print(f"   Ground truth: {n_congested} congested (>{TAU} veh/km), {n_uncongested} uncongested")

    df_results = pd.DataFrame(results)

    # Reorder columns
    cols = ['timestep', 'WWC', 'Inside_Variance', 'N_inside', 'N_outside', 'N_congested', 'N_uncongested']
    df_results = df_results[cols]

    csv_path = os.path.join(metrics_dir, 'metrics_final_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved FINAL metrics comparison to {csv_path}")
    print("✓ Validation Complete.")


if __name__ == "__main__":
    main()