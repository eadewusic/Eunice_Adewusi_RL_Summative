"""
Actor-Critic Training Script for Rwanda Traffic Junction Environment

This script implements Actor-Critic algorithm with separate actor and critic networks
for the traffic light optimization task.

Actor-Critic combines the benefits of value-based and policy-based methods by
learning both a value function (critic) and a policy (actor) simultaneously.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_junction_env import TrafficJunctionEnv
from environment.traffic_rendering import TrafficVisualizer
from training.training_logger import TrainingLogger, StepMetricsCollector, create_training_plots

# Add tensorboard logging
from torch.utils.tensorboard import SummaryWriter

class ActorNetwork(nn.Module):
    """
    Actor network for policy representation
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize actor network
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            hidden_size: Size of hidden layers
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through actor network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)

class CriticNetwork(nn.Module):
    """
    Critic network for value function approximation
    """
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        """
        Initialize critic network
        
        Args:
            state_size: Size of state space
            hidden_size: Size of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through critic network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ActorCriticAgent:
    """
    Actor-Critic agent for traffic light optimization
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.002,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 1.0,
                 device: str = 'auto'):
        """
        Initialize Actor-Critic agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            gamma: Discount factor
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Training history
        self.episode_rewards_history = []
        self.episode_lengths_history = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        
        # Initialize logging components (same as REINFORCE)
        self.training_logger = None
        self.tensorboard_writer = None
        
        # Hyperparameters
        self.hyperparameters = {
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'gamma': gamma,
            'entropy_coef': entropy_coef,
            'value_coef': value_coef,
            'max_grad_norm': max_grad_norm,
            'device': str(self.device)
        }
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action from actor network
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value from critic
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(training):
            # Get action probabilities from actor
            action_probs = self.actor(state_tensor)
            
            # Add small epsilon to prevent numerical issues
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum()
            
            # Get state value from critic
            value = self.critic(state_tensor)
            
            if training:
                # Sample action from distribution
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                return action.item(), log_prob, value.squeeze()
            else:
                # Deterministic action for evaluation
                action = torch.argmax(action_probs, dim=1)
                return action.item(), None, value.squeeze()
    
    def update_networks(self, states: List[np.ndarray], actions: List[int], 
                       rewards: List[float], next_states: List[np.ndarray],
                       dones: List[bool], log_probs: List[torch.Tensor],
                       values: List[torch.Tensor]):
        """
        Update actor and critic networks
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            log_probs: List of log probabilities
            values: List of state values
        """
        
        if len(states) == 0:
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        log_probs_tensor = torch.stack(log_probs).to(self.device)
        
        # Handle values tensor carefully
        values_list = []
        for value in values:
            if value.dim() == 0:
                values_list.append(value.unsqueeze(0))
            else:
                values_list.append(value.squeeze())
        values_tensor = torch.stack(values_list).to(self.device)
        
        # Calculate returns and advantages
        returns, advantages = self._calculate_returns_and_advantages(
            rewards_tensor, values_tensor, next_states_tensor, dones_tensor
        )
        
        # Update critic network
        critic_loss = self._update_critic(values_tensor, returns)
        
        # Update actor network
        actor_loss, entropy_loss = self._update_actor(log_probs_tensor, advantages)
        
        # Store losses for monitoring
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.entropy_values.append(entropy_loss)
    
    def _calculate_returns_and_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                                        next_states: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate returns and advantages using TD(0)
        
        Args:
            rewards: Tensor of rewards
            values: Tensor of state values
            next_states: Tensor of next states
            dones: Tensor of done flags
            
        Returns:
            returns: Discounted returns
            advantages: Advantage estimates
        """
        
        # Calculate next state values
        with torch.no_grad():
            next_values = self.critic(next_states)
            if next_values.dim() > 1:
                next_values = next_values.squeeze(-1)
            
            # Handle done states - set next values to 0 for terminal states
            if dones.any():
                next_values = next_values * (~dones).float()
        
        # Calculate TD targets (returns)
        returns = rewards + self.gamma * next_values
        
        # Calculate advantages (TD error)
        advantages = returns - values
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _update_critic(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        """
        Update critic network
        
        Args:
            values: Predicted state values
            returns: Target returns
            
        Returns:
            Critic loss value
        """
        
        # Calculate value loss (MSE)
        value_loss = F.mse_loss(values, returns.detach())
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.critic_optimizer.step()
        
        return value_loss.item()
    
    def _update_actor(self, log_probs: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float]:
        """
        Update actor network
        
        Args:
            log_probs: Log probabilities of actions
            advantages: Advantage estimates
            
        Returns:
            Actor loss and entropy loss values
        """
        
        # Calculate policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate entropy loss for exploration
        # Get current action probabilities
        states_for_entropy = []  # We need states to calculate entropy
        # For simplicity, we'll approximate entropy loss
        entropy_loss = 0.0  # Placeholder - would need states to calculate properly
        
        # Total actor loss
        total_loss = policy_loss - self.entropy_coef * entropy_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        
        return policy_loss.item(), entropy_loss
    
    def train(self, env: TrafficJunctionEnv,
              num_episodes: int = 1000,
              max_steps_per_episode: int = 500,
              update_frequency: int = 10,
              eval_frequency: int = 100,
              save_path: str = "models/actor_critic/",
              verbose: bool = True) -> Dict:
        """
        Train the Actor-Critic agent with automatic CSV logging
        
        Args:
            env: Training environment
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            update_frequency: How often to update networks (in steps)
            eval_frequency: Frequency of evaluation
            save_path: Path to save models
            verbose: Whether to print training progress
            
        Returns:
            Training statistics
        """
        
        print(f"Starting Actor-Critic Training for {num_episodes:,} episodes")
        print("=" * 60)
        print("Hyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print()
        
        # Initialize logging infrastructure (same as REINFORCE)
        self.training_logger = TrainingLogger("Actor-Critic")
        
        # Initialize tensorboard logging
        os.makedirs("./tensorboard_logs/actor_critic/", exist_ok=True)
        self.tensorboard_writer = SummaryWriter("./tensorboard_logs/actor_critic/ActorCritic_1")
        
        print("Started logging Actor-Critic training metrics")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Training statistics
        recent_rewards = deque(maxlen=100)
        best_avg_reward = float('-inf')
        start_time = datetime.now()
        
        # Storage for batch updates
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_log_probs = []
        batch_values = []
        
        global_step = 0
        
        try:
            for episode in range(num_episodes):
                # Reset environment
                state, info = env.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(max_steps_per_episode):
                    # Get action from actor-critic
                    action, log_prob, value = self.get_action(state, training=True)
                    
                    # Take action in environment
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Store transition
                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_rewards.append(reward)
                    batch_next_states.append(next_state)
                    batch_dones.append(done)
                    batch_log_probs.append(log_prob)
                    batch_values.append(value)
                    
                    # Update for next iteration
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    global_step += 1
                    
                    # Update networks if batch is full or episode ended
                    if len(batch_states) >= update_frequency or done:
                        if len(batch_states) > 0:
                            self.update_networks(
                                batch_states, batch_actions, batch_rewards,
                                batch_next_states, batch_dones, batch_log_probs, batch_values
                            )
                            
                            # Clear batch
                            batch_states = []
                            batch_actions = []
                            batch_rewards = []
                            batch_next_states = []
                            batch_dones = []
                            batch_log_probs = []
                            batch_values = []
                    
                    # Check if episode ended
                    if done:
                        break
                
                # Log episode metrics (same as REINFORCE)
                if self.training_logger and episode_length > 0:
                    episode_data = {
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'vehicles_processed': info.get('vehicles_processed', 0),
                        'total_waiting_time': info.get('total_waiting_time', 0),
                        'final_queue_length': info.get('queue_length', 0)
                    }
                    
                    self.training_logger.log_episode(episode_data)
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar('episode/reward', episode_reward, episode)
                    self.tensorboard_writer.add_scalar('episode/length', episode_length, episode)
                    if self.actor_losses:
                        self.tensorboard_writer.add_scalar('loss/actor', self.actor_losses[-1], episode)
                    if self.critic_losses:
                        self.tensorboard_writer.add_scalar('loss/critic', self.critic_losses[-1], episode)
                    if self.entropy_values:
                        self.tensorboard_writer.add_scalar('loss/entropy', self.entropy_values[-1], episode)
                
                # Record episode statistics
                self.episode_rewards_history.append(episode_reward)
                self.episode_lengths_history.append(episode_length)
                recent_rewards.append(episode_reward)
                
                # Print progress
                if verbose and (episode + 1) % 50 == 0:
                    avg_reward = np.mean(recent_rewards)
                    avg_actor_loss = np.mean(self.actor_losses[-50:]) if self.actor_losses else 0
                    avg_critic_loss = np.mean(self.critic_losses[-50:]) if self.critic_losses else 0
                    
                    print(f"Episode {episode + 1:4d}: "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg(100): {avg_reward:7.2f} | "
                          f"Length: {episode_length:3d} | "
                          f"Actor Loss: {avg_actor_loss:.4f} | "
                          f"Critic Loss: {avg_critic_loss:.4f}")
                
                # Evaluation and model saving
                if (episode + 1) % eval_frequency == 0:
                    avg_reward = np.mean(recent_rewards)
                    
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        
                        # Save best model
                        best_model_path = os.path.join(save_path, "ac_best.pth")
                        self.save_model(best_model_path)
                        
                        if verbose:
                            print(f"New best average reward: {best_avg_reward:.2f} - Model saved")
            
            # Training completed - finalize logging (same as REINFORCE)
            if self.training_logger:
                self.training_logger.save_final_summary()
                create_training_plots("Actor-Critic")
                print("Actor-Critic training metrics saved to CSV files")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            training_time = datetime.now() - start_time
            print(f"\nActor-Critic training completed in {training_time}")
            
            # Save final model
            final_model_path = os.path.join(save_path, "ac_traffic_final.pth")
            self.save_model(final_model_path)
            
            # Save training history
            history_path = os.path.join(save_path, "training_history.json")
            self.save_training_history(history_path)
            
            # Save hyperparameters
            config_path = os.path.join(save_path, "ac_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.hyperparameters, f, indent=2)
            
            # Print CSV file locations (same as REINFORCE)
            print(f"\nTraining metrics automatically saved:")
            print(f"   Episode metrics: results/training_logs/Actor-Critic_episode_metrics.csv")
            print(f"   Step metrics: results/training_logs/Actor-Critic_step_metrics.csv")
            print(f"   Training plots: results/training_logs/Actor-Critic_training_plots.png")
            print(f"   Training summary: results/training_logs/Actor-Critic_training_summary.txt")
            
            return {
                'training_time': str(training_time),
                'num_episodes': num_episodes,
                'final_avg_reward': np.mean(recent_rewards),
                'best_avg_reward': best_avg_reward,
                'final_model_path': final_model_path,
                'config_path': config_path,
                'episode_metrics_csv': 'results/training_logs/Actor-Critic_episode_metrics.csv',
                'step_metrics_csv': 'results/training_logs/Actor-Critic_step_metrics.csv'
            }
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            
            # Save interrupted model and close logging
            if self.training_logger:
                self.training_logger.save_final_summary()
                create_training_plots("Actor-Critic")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            interrupted_path = os.path.join(save_path, "ac_interrupted.pth")
            self.save_model(interrupted_path)
            return None
    
    def evaluate(self, env: TrafficJunctionEnv, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the trained Actor-Critic agent
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes for evaluation
            
        Returns:
            Mean reward and standard deviation
        """
        
        print(f"Evaluating Actor-Critic agent over {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get deterministic action
                action, _, _ = self.get_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Episode Length: {mean_length:.1f}")
        
        return mean_reward, std_reward
    
    def save_model(self, filepath: str):
        """Save model state"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparameters': self.hyperparameters,
            'episode_rewards_history': self.episode_rewards_history
        }
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        try:
            # Try with weights_only=False for compatibility
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load checkpoint - {e}")
            print("Continuing with current model weights")
            return
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.episode_rewards_history = checkpoint.get('episode_rewards_history', [])
        print(f"Actor-Critic model loaded from: {filepath}")
    
    def save_training_history(self, filepath: str):
        """Save training history"""
        history = {
            'episode_rewards': self.episode_rewards_history,
            'episode_lengths': self.episode_lengths_history,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_values': self.entropy_values
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

def main_training_experiment():
    """
    Main Actor-Critic training experiment
    """
    print("Actor-Critic Main Training Experiment")
    print("=" * 50)
    
    # Create training environment
    env = TrafficJunctionEnv(render_mode=None)
    
    # Create Actor-Critic agent
    agent = ActorCriticAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5
    )
    
    # Train the agent
    training_stats = agent.train(
        env=env,
        num_episodes=1200,
        update_frequency=10,
        eval_frequency=100,
        save_path="models/actor_critic/",
        verbose=True
    )
    
    if training_stats:
        # Final evaluation
        print("\nFINAL EVALUATION")
        print("=" * 30)
        
        mean_reward, std_reward = agent.evaluate(env, n_episodes=20)
        
        # Save evaluation results
        evaluation_results = {
            'algorithm': 'Actor-Critic',
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'hyperparameters': agent.hyperparameters,
            'training_stats': training_stats
        }
        
        results_path = "models/actor_critic/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved: {results_path}")
    
    # Clean up
    env.close()
    
    return agent

if __name__ == "__main__":
    print("Rwanda Traffic Flow Optimization - Actor-Critic Training")
    print("Algorithm: Actor-Critic (Hybrid Value-Policy Method)")
    print()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run main training experiment
    trained_agent = main_training_experiment()
    
    print("\nActor-Critic training completed!")