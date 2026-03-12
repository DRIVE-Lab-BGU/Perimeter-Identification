import torch
import numpy as np
import sys
import os


# Determine project root and update Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from PI_env import SimplifiedJunctionEnv
from preprocessing import GraphPreprocessor


class GCNJunctionEnv(SimplifiedJunctionEnv):
    """
    Extended environment that integrates GCN features
    """

    def __init__(self, net, prepared_junctions, heatmap_without_network, heatmap_with_network,
                 lambda_reg, congestion_threshold,  device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(net, prepared_junctions, heatmap_without_network, heatmap_with_network, lambda_reg, congestion_threshold)

        # Create graph preprocessor
        self.graph_preprocessor = GraphPreprocessor(
            net, prepared_junctions, heatmap_without_network.shape
        )

        # Device for PyTorch
        self.device = device

        # Store current node features and adjacency matrix
        self.current_node_features = None
        self.adjacency_tensor = self.graph_preprocessor.get_normalized_adjacency_tensor().to(self.device)

    def reset(self, seed=None, options=None):
        """
        Reset environment and update graph features
        """
        # Reset the base environment
        observation, info = super().reset(seed=seed, options=options)

        # Update graph features
        self._update_graph_features()

        return observation, info

    def step(self, action):
        """
        Execute action and update graph features
        """
        # Execute in base environment
        observation, reward, terminated, truncated, info = super().step(action)

        # Update graph features
        self._update_graph_features()

        return observation, reward, terminated, truncated, info

    def _update_graph_features(self):
        """
        Update node features based on current state
        """
        self.current_node_features = self.graph_preprocessor.create_node_features(
            self.active_junctions
        ).to(self.device)

    def get_gcn_observation(self):
        """
        Get the current node features and adjacency matrix for GCN processing
        """
        # Get action mask from parent class
        action_mask = self.get_action_mask()
        
        # Ensure action mask has the correct size (num_junctions + 1)
        expected_size = self.num_junctions + 1
        if len(action_mask) != expected_size:
            # Create a corrected mask
            corrected_mask = np.zeros(expected_size, dtype=np.uint8)
            # Copy the junction actions
            corrected_mask[:self.num_junctions] = action_mask[:self.num_junctions]
            # Always allow finish action
            corrected_mask[self.num_junctions] = 1
            action_mask = corrected_mask
        
        return {
            'node_features': self.current_node_features,
            'adjacency': self.adjacency_tensor,
            'action_mask': torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        }

    def get_valid_actions_tensor(self):
        """
        Get a tensor of valid actions
        """
        valid_actions = self.get_valid_actions()
        return torch.LongTensor(valid_actions).to(self.device)

    def render(self, mode='human'):
        """
        Render the environment
        """
        # Use the rendering from the base class
        if mode == 'human':
            super().render(mode=mode)
        else:
            img = super().render(mode=mode)
            return img
