import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    """

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        # Graph convolution: X' = A_hat * X * W
        support = torch.bmm(adj, x)  # [batch_size, num_nodes, in_features]
        output = self.linear(support)  # [batch_size, num_nodes, out_features]
        return output


class DenseGCN(nn.Module):
    """
    Graph Convolutional Network with Dense Connections
    Supports variable number of layers
    """

    def __init__(self, in_features, hidden_dim, embedding_dim, num_layers=2, dropout=0.1):
        super(DenseGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Build GCN layers dynamically
        self.gcn_layers = nn.ModuleList()

        # First layer: in_features -> hidden_dim
        self.gcn_layers.append(GCNLayer(in_features, hidden_dim))

        # Middle layers with dense connections
        current_input_dim = in_features + hidden_dim
        for i in range(1, num_layers - 1):
            self.gcn_layers.append(GCNLayer(current_input_dim, hidden_dim))
            current_input_dim += hidden_dim

        # Last layer outputs embedding_dim
        if num_layers > 1:
            self.gcn_layers.append(GCNLayer(current_input_dim, embedding_dim))
            current_input_dim += embedding_dim
        else:
            # If only 1 layer, it should output embedding_dim
            self.gcn_layers = nn.ModuleList([GCNLayer(in_features, embedding_dim)])
            current_input_dim = in_features + embedding_dim

        # Final projection combines all layer outputs
        self.final_projection = nn.Linear(current_input_dim, embedding_dim)

    def forward(self, x, adj):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        layer_outputs = [x]  # Store all layer outputs for dense connections

        for i, gcn_layer in enumerate(self.gcn_layers):
            # Input is concatenation of all previous outputs
            layer_input = torch.cat(layer_outputs, dim=-1)

            # Apply GCN layer
            layer_output = F.relu(gcn_layer(layer_input, adj))
            layer_output = self.dropout(layer_output)

            layer_outputs.append(layer_output)

        # Final projection with all dense connections
        final_input = torch.cat(layer_outputs, dim=-1)
        output = self.final_projection(final_input)

        return output


class DenseHead(nn.Module):
    """
    Dense feedforward head with variable number of layers
    Each layer receives concatenation of all previous layer outputs
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(DenseHead, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Build layers dynamically
        current_input_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_input_dim, hidden_dim))
            current_input_dim += hidden_dim  # Dense connection: add previous dimensions

        # Final output layer
        self.output_layer = nn.Linear(current_input_dim, output_dim)

    def forward(self, x):
        """
        x: Input tensor [batch_size, ..., input_dim]
        """
        layer_outputs = [x]  # Store all layer outputs

        for layer in self.layers:
            # Input is concatenation of all previous outputs
            layer_input = torch.cat(layer_outputs, dim=-1)

            # Apply linear layer + activation + dropout
            layer_output = F.relu(layer(layer_input))
            layer_output = self.dropout(layer_output)

            layer_outputs.append(layer_output)

        # Final output layer
        final_input = torch.cat(layer_outputs, dim=-1)
        output = self.output_layer(final_input)

        return output


class DenseGCNPolicy(nn.Module):
    """
    GCN-based policy network with dense connections for RL
    Supports variable architecture through config
    """

    def __init__(self, in_features, hidden_dim, embedding_dim, num_junctions,
                 num_gcn_layers=2, dropout=0.1, action_head_dims=None):
        super(DenseGCNPolicy, self).__init__()
        self.num_junctions = num_junctions

        # Default action head dimensions if not provided
        if action_head_dims is None:
            action_head_dims = [64, 32]

        # Build GCN backbone
        self.gcn = DenseGCN(in_features, hidden_dim, embedding_dim,
                            num_layers=num_gcn_layers, dropout=dropout)

        # Build action heads using DenseHead
        self.add_head = DenseHead(embedding_dim, action_head_dims, 1, dropout=dropout)
        self.remove_head = DenseHead(embedding_dim, action_head_dims, 1, dropout=dropout)
        self.finish_head = DenseHead(embedding_dim, action_head_dims, 1, dropout=dropout)

    def forward(self, x, adj, action_mask=None):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        action_mask: Binary mask for valid actions [batch_size, num_nodes + 1]
        """
        batch_size = x.size(0)

        # Process node features through Dense GCN
        node_embeddings = self.gcn(x, adj)  # [batch_size, num_nodes, embedding_dim]

        # Get active status for each junction
        active_status = x[:, :, 3]  # [batch_size, num_nodes]

        # Compute add and remove logits
        add_logits = self.add_head(node_embeddings).squeeze(-1)  # [batch_size, num_nodes]
        remove_logits = self.remove_head(node_embeddings).squeeze(-1)  # [batch_size, num_nodes]

        # Ensure logits have correct shape
        if len(add_logits.shape) == 1:
            add_logits = add_logits.unsqueeze(0)
        if len(remove_logits.shape) == 1:
            remove_logits = remove_logits.unsqueeze(0)

        # Combine logits based on active status
        node_logits = torch.where(
            active_status > 0.5,  # If junction is active
            remove_logits,  # Use remove head
            add_logits  # Use add head
        )

        # Compute finish action logit using mean pooling
        graph_embedding = torch.mean(node_embeddings, dim=1)  # [batch_size, embedding_dim]
        finish_logit = self.finish_head(graph_embedding)  # [batch_size, 1]

        # Concatenate node logits with finish logit
        logits = torch.cat([node_logits, finish_logit], dim=1)  # [batch_size, num_nodes + 1]

        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, node_embeddings


class DenseGCNValueNetwork(nn.Module):
    """
    GCN-based value network with dense connections
    Supports variable architecture through config
    """

    def __init__(self, in_features, hidden_dim, embedding_dim,
                 num_gcn_layers=2, dropout=0.1, value_head_dims=None):
        super(DenseGCNValueNetwork, self).__init__()

        # Default value head dimensions if not provided
        if value_head_dims is None:
            value_head_dims = [64, 32]

        # Build GCN backbone
        self.gcn = DenseGCN(in_features, hidden_dim, embedding_dim,
                            num_layers=num_gcn_layers, dropout=dropout)

        # Build value head using DenseHead
        self.value_head = DenseHead(embedding_dim, value_head_dims, 1, dropout=dropout)

    def forward(self, x, adj):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        # Process node features through Dense GCN
        node_embeddings = self.gcn(x, adj)  # [batch_size, num_nodes, embedding_dim]

        # Compute global graph representation through mean pooling
        graph_embedding = torch.mean(node_embeddings, dim=1)  # [batch_size, embedding_dim]

        # Compute value through dense head
        value = self.value_head(graph_embedding)  # [batch_size, 1]

        return value