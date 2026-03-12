import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gcn_model import DenseGCNPolicy, DenseGCNValueNetwork
from data_utils import create_env_from_timestep

class GCNPPOAgent:
    """
    PPO Agent using GCN for processing junction network
    """

    def __init__(
            self,
            env,
            config=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.env = env
        self.device = device

        # Load parameters from config or use defaults/backward compatible values
        if config is not None:
            # Extract from config
            agent_cfg = config['agent']
            model_cfg = config['model']

            # Use the START values for initialization
            self.learning_rate = agent_cfg.get('lr_start', 3e-4)
            self.entropy_coef = agent_cfg['entropy_coef']

            self.gamma = agent_cfg['gamma']
            self.gae_lambda = agent_cfg['gae_lambda']
            self.clip_ratio = agent_cfg['clip_ratio']
            self.value_coef = agent_cfg['value_coef']
            self.max_grad_norm = agent_cfg['max_grad_norm']
            self.ppo_epochs = agent_cfg['ppo_epochs']
            self.ppo_batch_size = agent_cfg['ppo_batch_size']

            self.hidden_dim = model_cfg['hidden_dim']
            self.embedding_dim = model_cfg['embedding_dim']


        # Determine input feature size
        self.in_features = 4  # normalized_x, normalized_y, intensity, active_status
        self.num_junctions = env.num_junctions

        # Get model-specific parameters from config if available
        model_cfg = config['model']
        dropout = model_cfg.get('dropout')
        num_gcn_layers = model_cfg.get('num_gcn_layers')
        action_head_dims = model_cfg.get('action_head_dims')
        value_head_dims = model_cfg.get('value_head_dims')


        # Initialize policy and value networks
        self.policy = DenseGCNPolicy(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_junctions=self.num_junctions,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout,
            action_head_dims=action_head_dims
        ).to(device)

        self.value_net = DenseGCNValueNetwork(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout,
            value_head_dims=value_head_dims
        ).to(device)

        # The optimizers will now be created with the STARTING learning rate
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.action_masks = []

        self.training_step = 0

    def update_schedules(self, current_episode, total_episodes, config):
        """Linearly decays the learning rate and entropy coefficient."""
        agent_cfg = config['agent']
        lr_start = agent_cfg.get('lr_start', 3e-4)
        lr_end = agent_cfg.get('lr_end', 1e-5)

        # Calculate the progress fraction (from 0.0 to 1.0)
        fraction = current_episode / total_episodes

        # Linearly interpolate the new learning rate
        new_lr = lr_start - (lr_start - lr_end) * fraction

        # Update the learning rate in both optimizers
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = new_lr


    def select_action(self, state, greedy=False):
        """
        Select an action using the policy network
        """
        # Get state components
        node_features = state['node_features']
        adjacency = state['adjacency']
        action_mask = state['action_mask']
    
        # SET TO EVAL MODE FOR INFERENCE
        self.policy.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            # Forward pass through policy network
            logits, _ = self.policy(node_features, adjacency, action_mask)
    
            # Calculate value
            value = self.value_net(node_features, adjacency)
    
            # Sample action or take greedy action
            if greedy:
                action_probs = F.softmax(logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from the distribution
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample().item()
    
            # Get log probability of the action
            log_prob = F.log_softmax(logits, dim=-1)[0, action]
    
        return action, log_prob.item(), value.item()

    def store_transition(self, state, action, reward, value, log_prob, mask, action_mask):
        """
        Store transition in experience buffer
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        self.action_masks.append(action_mask)

    def compute_returns_and_advantages(self, next_value):
        """
        Compute returns and advantages using GAE
        """
        returns = []
        advantages = []

        next_return = next_value
        next_advantage = 0

        for step in reversed(range(len(self.rewards))):
            # Calculate returns with discounting
            returns.insert(0, self.rewards[step] + self.gamma * next_return * self.masks[step])

            # Calculate TD error
            delta = self.rewards[step] + self.gamma * next_value * self.masks[step] - self.values[step]

            # Calculate advantage using GAE
            next_advantage = delta + self.gamma * self.gae_lambda * next_advantage * self.masks[step]
            advantages.insert(0, next_advantage)

            next_return = returns[0]
            next_value = self.values[step]

        return returns, advantages

    def update_policy(self, batch_size=None, epochs=None):
        """
        Update policy and value networks using PPO
        """
        # Use config values or fall back to provided parameters
        self.policy.train()
        self.value_net.train()
        
        # Use config values or fall back to provided parameters
        batch_size = batch_size if batch_size is not None else self.ppo_batch_size
        epochs = epochs if epochs is not None else self.ppo_epochs

        # Get the next state value for advantage calculation
        if len(self.states) > 0:
            with torch.no_grad():
                next_state = self.states[-1]
                node_features = next_state['node_features']
                adjacency = next_state['adjacency']
                next_value = self.value_net(node_features, adjacency).item()
        else:
            next_value = 0

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset indices
        indices = np.arange(len(self.states))

        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Mini-batch updates
        for _ in range(epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, len(indices), batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + batch_size]

                # Slice mini-batch data
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]

                # Process mini-batch states
                mb_node_features = torch.cat([self.states[i]['node_features'] for i in idx], dim=0)
                mb_adjacency = torch.cat([self.states[i]['adjacency'] for i in idx], dim=0)
                mb_action_masks = torch.cat([self.action_masks[i] for i in idx], dim=0)

                # Forward pass through networks
                mb_logits, _ = self.policy(mb_node_features, mb_adjacency, mb_action_masks)
                mb_values = self.value_net(mb_node_features, mb_adjacency).squeeze(-1)

                # Calculate log probabilities and entropy
                log_probs_all = F.log_softmax(mb_logits, dim=-1)
                entropy = torch.mean(torch.sum(-torch.exp(log_probs_all) * log_probs_all, dim=-1))

                # Get log probabilities for taken actions
                log_probs = torch.gather(log_probs_all, 1, mb_actions.unsqueeze(1)).squeeze()

                # Calculate probability ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)

                # Calculate surrogate objectives
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages

                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                value_loss = F.mse_loss(mb_values, mb_returns)

                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize policy network
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                # Store losses for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        # Clear experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.action_masks = []

        # Increment training step
        self.training_step += 1

        # Return average losses
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }


    def train(self, timestep_data, train_timesteps, lambda_reg, congestion_threshold, config):
        """
        Train the agent across multiple traffic timesteps.
        Mirrors the training loop in train_single_run.py without monitoring or evaluation.

        Parameters:
        -----------
        timestep_data : dict
            Pre-loaded timestep data from preload_timestep_data()
        train_timesteps : list
            List of timesteps to sample from during training
        lambda_reg : float
            Lambda regularization parameter
        congestion_threshold : float
            Congestion threshold parameter
        config : dict
            Full configuration dictionary
        """
        num_episodes = config['training']['num_episodes']
        max_steps = config['training']['max_steps_per_episode']
        update_freq = config['training']['update_frequency']

        for episode in range(1, num_episodes + 1):
            # Randomly select a training timestep for this episode
            current_timestep = np.random.choice(train_timesteps)

            # Create environment for this timestep
            env = create_env_from_timestep(
                timestep_data, current_timestep, lambda_reg, congestion_threshold, self.device
            )

            state, _ = env.reset()
            gcn_state = env.get_gcn_observation()

            done = False
            steps = 0

            while not done and steps < max_steps:
                action, log_prob, value = self.select_action(gcn_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_gcn_state = env.get_gcn_observation()

                mask = 0.0 if done else 1.0
                self.store_transition(
                    gcn_state, action, reward, value, log_prob, mask,
                    gcn_state['action_mask']
                )

                gcn_state = next_gcn_state
                steps += 1

                if len(self.states) >= update_freq:
                    self.update_policy()

            if len(self.states) > 0:
                self.update_policy()

            self.update_schedules(episode, num_episodes, config)

            env.close()

    def save_models(self, path):
        """
        Save policy and value networks
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)

    def load_models(self, path):
        """
        Load policy and value networks
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])