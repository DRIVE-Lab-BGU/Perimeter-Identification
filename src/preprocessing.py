import numpy as np
import torch
import networkx as nx

class GraphPreprocessor:
    """
    Preprocessor for creating graph structure and features from SUMO network data
    """

    def __init__(self, net, prepared_junctions, heatmap_shape):
        self.net = net
        self.prepared_junctions = prepared_junctions
        self.heatmap_height, self.heatmap_width = heatmap_shape[:2]

        # Create a mapping from junction ID to index
        self.junction_dict = {j['id']: j for j in prepared_junctions}
        self.node_id_to_index = {j['id']: i for i, j in enumerate(prepared_junctions)}
        self.index_to_node_id = {i: j['id'] for i, j in enumerate(prepared_junctions)}
        self.num_junctions = len(prepared_junctions)

        # Create adjacency matrix
        self.adjacency_matrix = self._create_adjacency_matrix()
        self.normalized_adj = self._normalize_adjacency_matrix(self.adjacency_matrix)

    def _create_adjacency_matrix(self):
        """
        Create a sparse adjacency matrix based on the network connectivity
        """
        # Create a NetworkX graph
        G = nx.DiGraph()

        # Add nodes
        for junction in self.prepared_junctions:
            G.add_node(junction['id'])

        # Add edges from the SUMO network
        for edge in self.net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()

            # Only add edges if both nodes are in our junction list
            if from_node in self.node_id_to_index and to_node in self.node_id_to_index:
                G.add_edge(from_node, to_node)
                G.add_edge(to_node, from_node)  # Make it undirected for GCN

        # Convert to adjacency matrix (using node_id_to_index mapping)
        adj_matrix = np.zeros((self.num_junctions, self.num_junctions), dtype=np.float32)

        for u, v in G.edges():
            u_idx = self.node_id_to_index[u]
            v_idx = self.node_id_to_index[v]
            adj_matrix[u_idx, v_idx] = 1.0
            adj_matrix[v_idx, u_idx] = 1.0  # Make it symmetric

        # Add self-loops
        adj_matrix = adj_matrix + np.eye(self.num_junctions, dtype=np.float32)

        return adj_matrix

    def _normalize_adjacency_matrix(self, adj_matrix):
        """
        Normalize adjacency matrix for GCN: A_hat = D^(-1/2) * A * D^(-1/2)
        """
        # Calculate degree matrix
        rowsum = np.array(adj_matrix.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        # Calculate normalized adjacency
        normalized_adj = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

        return normalized_adj

    def create_node_features(self, active_junctions, batch_size=1):
        """
        Create normalized node features for all junctions
        Features: [normalized_x, normalized_y, intensity, active_status]

        Parameters:
        -----------
        active_junctions: set of active junction IDs
        batch_size: number of identical copies to create (for batch processing)

        Returns:
        --------
        node_features: tensor of shape [batch_size, num_junctions, 4]
        """
        features = np.zeros((self.num_junctions, 4), dtype=np.float32)

        for idx, junction in enumerate(self.prepared_junctions):
            # Normalize spatial coordinates
            norm_x = junction['pixel_x'] / self.heatmap_width
            norm_y = junction['pixel_y'] / self.heatmap_height

            # Get intensity (already normalized 0-1)
            intensity = 1 - junction['intensity']

            # Set active status (0 or 1)
            active = 1.0 if junction['id'] in active_junctions else 0.0

            # Set feature vector
            features[idx] = [norm_x, norm_y, intensity, active]

        # Convert to torch tensor and repeat for batch size
        tensor_features = torch.FloatTensor(features)
        if batch_size > 1:
            tensor_features = tensor_features.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            tensor_features = tensor_features.unsqueeze(0)  # Add batch dimension

        return tensor_features

    def get_normalized_adjacency_tensor(self, batch_size=1):
        """
        Get normalized adjacency matrix as a torch tensor with batch dimension
        """
        adj_tensor = torch.FloatTensor(self.normalized_adj)
        if batch_size > 1:
            adj_tensor = adj_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            adj_tensor = adj_tensor.unsqueeze(0)  # Add batch dimension

        return adj_tensor