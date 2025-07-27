"""
REINFORCE Training Script for Rwanda Traffic Junction Environment

This script implements REINFORCE (Monte Carlo Policy Gradient) algorithm
for the traffic light optimization task.

REINFORCE is a policy gradient method that learns directly from episodes
using Monte Carlo sampling, making it suitable for episodic tasks.
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

class PolicyNetwork(nn.Module):
    """
    Policy network for REINFORCE algorithm
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize policy network
        
        Args:
            state_size: Size of state space
            action_size: Size of action space  
            hidden_size: Size of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        """Forward pass through network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """
    Value network for baseline (variance reduction)
    """
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        """
        Initialize value network for baseline
        
        Args:
            state_size: Size of state space
            hidden_size: Size of hidden layers
        """
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        """Forward pass through value network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class REINFORCEAgent:
    """
    REINFORCE agent for traffic light optimization with improved stability
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.0005,  # Reduced from 0.001 for stability
                 gamma: float = 0.99,
                 use_baseline: bool = True,
                 device: str = 'auto'):
        """
        Initialize REINFORCE agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate for optimizer (reduced for stability)
            gamma: Discount factor
            use_baseline: Whether to use value function baseline
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_size, action_size).to(self.device)
        
        if use_baseline:
            self.value_net = ValueNetwork(state_size).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Learning rate schedulers for stability
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=300, gamma=0.8)
        if use_baseline:
            self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=300, gamma=0.8)
        
        # Storage for episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Training history
        self.episode_rewards_history = []
        self.episode_lengths_history = []
        self.policy_losses = []
        self.value_losses = []
        
        # Initialize logging components
        self.training_logger = None
        self.tensorboard_writer = None
        
        # Early stopping variables
        self.best_avg_reward = float('-inf')
        self.patience = 5  # Number of evaluations to wait for improvement
        self.patience_counter = 0
        self.early_stop = False
        
        # Hyperparameters
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'use_baseline': use_baseline,
            'network_architecture': '128-128-64',
            'device': str(self.device),
            'patience': self.patience
        }
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, torch.Tensor]:
        """
        Get action from policy network
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action
            log_prob: Log probability of action (for training)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(training):
            action_probs = self.policy_net(state_tensor)
            
            # Add small epsilon to prevent numerical issues
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum()
            
            if training:
                # Sample action from probability distribution
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                return action.item(), log_prob
            else:
                # Deterministic action for evaluation
                action = torch.argmax(action_probs, dim=1)
                return action.item(), None
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, log_prob: torch.Tensor):
        """
        Store transition data for episode
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
    
    def calculate_returns(self) -> List[float]:
        """
        Calculate discounted returns for episode
        
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        
        # Calculate returns backwards from end of episode
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns for stability
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """
        Update policy network using REINFORCE algorithm
        """
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate returns
        returns = self.calculate_returns()
        
        # Convert episode data to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        log_probs = torch.stack(self.episode_log_probs).to(self.device)
        
        # Calculate baseline if using value function
        if self.use_baseline:
            with torch.no_grad():
                baselines = self.value_net(states).squeeze()
            advantages = returns - baselines
            
            # Update value network
            value_loss = F.mse_loss(self.value_net(states).squeeze(), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            
            # Gradient clipping for value network
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            
            self.value_optimizer.step()
            self.value_losses.append(value_loss.item())
        else:
            advantages = returns
        
        # Calculate policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability (reduced from 1.0 to 0.5)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        
        self.policy_optimizer.step()
        self.policy_losses.append(policy_loss.item())
        
        # Update learning rate schedulers
        self.policy_scheduler.step()
        if self.use_baseline:
            self.value_scheduler.step()
    
    def reset_episode(self):
        """Reset episode storage"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def check_early_stopping(self, current_avg_reward: float, episode: int) -> bool:
        """
        Check if training should stop early based on performance
        
        Args:
            current_avg_reward: Current average reward
            episode: Current episode number
            
        Returns:
            True if training should stop
        """
        if current_avg_reward > self.best_avg_reward:
            self.best_avg_reward = current_avg_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            
            # Check for severe performance degradation
            if episode > 500 and current_avg_reward < (self.best_avg_reward - 1000):
                print(f"\nSevere performance degradation detected!")
                print(f"Current: {current_avg_reward:.2f}, Best: {self.best_avg_reward:.2f}")
                return True
            
            # Check patience
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {self.patience} evaluations without improvement")
                print(f"Best average reward: {self.best_avg_reward:.2f}")
                return True
            
            return False
    
    def train(self, env: TrafficJunctionEnv, 
              num_episodes: int = 1000,  # Reduced from 1500
              max_steps_per_episode: int = 500,
              eval_frequency: int = 50,  # More frequent evaluation
              save_path: str = "models/reinforce/",
              verbose: bool = True) -> Dict:
        """
        Train the REINFORCE agent with improved stability and early stopping
        
        Args:
            env: Training environment
            num_episodes: Number of training episodes (reduced)
            max_steps_per_episode: Maximum steps per episode
            eval_frequency: Frequency of evaluation (increased)
            save_path: Path to save models
            verbose: Whether to print training progress
            
        Returns:
            Training statistics
        """
        
        print(f"Starting REINFORCE Training for {num_episodes:,} episodes")
        print("=" * 60)
        print("Improvements:")
        print("  • Lower learning rate (0.0005) for stability")
        print("  • Learning rate scheduling")
        print("  • Early stopping with patience")
        print("  • Improved gradient clipping")
        print("  • More frequent evaluation")
        print("  • Performance degradation detection")
        print()
        print("Hyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print()
        
        # Initialize logging infrastructure
        self.training_logger = TrainingLogger("REINFORCE")
        
        # Initialize tensorboard logging
        os.makedirs("./tensorboard_logs/reinforce/", exist_ok=True)
        self.tensorboard_writer = SummaryWriter("./tensorboard_logs/reinforce/REINFORCE_1")
        
        print("Started logging REINFORCE training metrics")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Training statistics
        recent_rewards = deque(maxlen=100)
        start_time = datetime.now()
        
        global_step = 0
        
        try:
            for episode in range(num_episodes):
                # Reset environment and episode storage
                state, info = env.reset()
                self.reset_episode()
                
                episode_reward = 0
                episode_length = 0
                
                for step in range(max_steps_per_episode):
                    # Get action from policy
                    action, log_prob = self.get_action(state, training=True)
                    
                    # Take action in environment
                    next_state, reward, terminated, truncated, info = env.step(action)
                    
                    # Store transition
                    self.store_transition(state, action, reward, log_prob)
                    
                    # Update for next iteration
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    global_step += 1
                    
                    # Check if episode ended
                    if terminated or truncated:
                        break
                
                # Update policy after episode
                self.update_policy()
                
                # Log episode metrics
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
                    self.tensorboard_writer.add_scalar('learning_rate/policy', 
                                                     self.policy_optimizer.param_groups[0]['lr'], episode)
                    if self.policy_losses:
                        self.tensorboard_writer.add_scalar('loss/policy', self.policy_losses[-1], episode)
                    if self.value_losses and self.use_baseline:
                        self.tensorboard_writer.add_scalar('loss/value', self.value_losses[-1], episode)
                
                # Record episode statistics
                self.episode_rewards_history.append(episode_reward)
                self.episode_lengths_history.append(episode_length)
                recent_rewards.append(episode_reward)
                
                # Print progress
                if verbose and (episode + 1) % 50 == 0:
                    avg_reward = np.mean(recent_rewards)
                    current_lr = self.policy_optimizer.param_groups[0]['lr']
                    
                    print(f"Episode {episode + 1:4d}: "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg(100): {avg_reward:7.2f} | "
                          f"Length: {episode_length:3d} | "
                          f"LR: {current_lr:.6f} | "
                          f"Policy Loss: {self.policy_losses[-1] if self.policy_losses else 0:.4f}")
                
                # Evaluation and early stopping check
                if (episode + 1) % eval_frequency == 0:
                    avg_reward = np.mean(recent_rewards)
                    
                    # Save best model
                    if avg_reward > self.best_avg_reward:
                        best_model_path = os.path.join(save_path, "reinforce_best.pth")
                        self.save_model(best_model_path)
                        
                        if verbose:
                            print(f"New best average reward: {avg_reward:.2f} - Model saved")
                    
                    # Check for early stopping
                    if self.check_early_stopping(avg_reward, episode):
                        self.early_stop = True
                        print(f"Early stopping at episode {episode + 1}")
                        break
            
            # Training completed - finalize logging
            if self.training_logger:
                self.training_logger.save_final_summary()
                create_training_plots("REINFORCE")
                print("REINFORCE training metrics saved to CSV files")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            training_time = datetime.now() - start_time
            print(f"\nREINFORCE training completed in {training_time}")
            
            if self.early_stop:
                print(f"Training stopped early with best performance: {self.best_avg_reward:.2f}")
            
            # Save final model
            final_model_path = os.path.join(save_path, "reinforce_traffic_final.pth")
            self.save_model(final_model_path)
            
            # Save training history
            history_path = os.path.join(save_path, "training_history.json")
            self.save_training_history(history_path)
            
            # Save hyperparameters
            config_path = os.path.join(save_path, "reinforce_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.hyperparameters, f, indent=2)
            
            # Print CSV file locations
            print(f"\nTraining metrics automatically saved:")
            print(f"   Episode metrics: results/training_logs/REINFORCE_episode_metrics.csv")
            print(f"   Step metrics: results/training_logs/REINFORCE_step_metrics.csv")
            print(f"   Training plots: results/training_logs/REINFORCE_training_plots.png")
            print(f"   Training summary: results/training_logs/REINFORCE_training_summary.txt")
            
            return {
                'training_time': str(training_time),
                'num_episodes': episode + 1,
                'final_avg_reward': np.mean(recent_rewards),
                'best_avg_reward': self.best_avg_reward,
                'early_stop': self.early_stop,
                'final_model_path': final_model_path,
                'config_path': config_path,
                'episode_metrics_csv': 'results/training_logs/REINFORCE_episode_metrics.csv',
                'step_metrics_csv': 'results/training_logs/REINFORCE_step_metrics.csv'
            }
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            
            # Save interrupted model and close logging
            if self.training_logger:
                self.training_logger.save_final_summary()
                create_training_plots("REINFORCE")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            interrupted_path = os.path.join(save_path, "reinforce_interrupted.pth")
            self.save_model(interrupted_path)
            return None
    
    def evaluate(self, env: TrafficJunctionEnv, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the trained REINFORCE agent
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes for evaluation
            
        Returns:
            Mean reward and standard deviation
        """
        
        print(f"Evaluating REINFORCE agent over {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get deterministic action
                action, _ = self.get_action(state, training=False)
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
            'policy_net_state_dict': self.policy_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
            'hyperparameters': self.hyperparameters,
            'episode_rewards_history': self.episode_rewards_history,
            'best_avg_reward': self.best_avg_reward
        }
        
        if self.use_baseline:
            checkpoint['value_net_state_dict'] = self.value_net.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            checkpoint['value_scheduler_state_dict'] = self.value_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
        
        if self.use_baseline and 'value_net_state_dict' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
        
        self.episode_rewards_history = checkpoint.get('episode_rewards_history', [])
        self.best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
        print(f"REINFORCE model loaded from: {filepath}")
    
    def save_training_history(self, filepath: str):
        """Save training history"""
        history = {
            'episode_rewards': self.episode_rewards_history,
            'episode_lengths': self.episode_lengths_history,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses if self.use_baseline else [],
            'best_avg_reward': self.best_avg_reward,
            'early_stop': self.early_stop
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

def hyperparameter_tuning_experiment():
    """
    Run hyperparameter tuning experiment for REINFORCE
    """
    print("REINFORCE Hyperparameter Tuning Experiment")
    print("=" * 50)
    
    # Create environment for tuning
    env = TrafficJunctionEnv(render_mode=None)
    
    # Hyperparameter combinations to test (with improved stability)
    hyperparameter_configs = [
        # Configuration 1: Conservative (very stable)
        {
            'name': 'conservative',
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'use_baseline': True
        },
        # Configuration 2: Standard (balanced)
        {
            'name': 'standard',
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'use_baseline': True
        },
        # Configuration 3: Moderate (slightly faster)
        {
            'name': 'moderate',
            'learning_rate': 0.0008,
            'gamma': 0.99,
            'use_baseline': True
        }
    ]
    
    results = []
    
    for config in hyperparameter_configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Create agent with specific hyperparameters
        agent = REINFORCEAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            use_baseline=config['use_baseline']
        )
        
        # Train for shorter duration for tuning
        training_stats = agent.train(
            env=env,
            num_episodes=500,
            eval_frequency=50,
            save_path=f"models/reinforce_tuning/{config['name']}/",
            verbose=False
        )
        
        if training_stats:
            # Evaluate performance
            mean_reward, std_reward = agent.evaluate(env, n_episodes=5)
            
            results.append({
                'config_name': config['name'],
                'hyperparameters': config,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'best_avg_reward': training_stats['best_avg_reward']
            })
    
    # Report results
    print("\nHYPERPARAMETER TUNING RESULTS")
    print("=" * 50)
    
    best_config = None
    best_reward = float('-inf')
    
    for result in results:
        print(f"\nConfiguration: {result['config_name']}")
        print(f"  Final Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Best Training: {result['best_avg_reward']:.2f}")
        
        if result['best_avg_reward'] > best_reward:
            best_reward = result['best_avg_reward']
            best_config = result
    
    if best_config:
        print(f"\nBest Configuration: {best_config['config_name']}")
        print(f"   Best Training Reward: {best_config['best_avg_reward']:.2f}")
        print("   Hyperparameters:")
        for key, value in best_config['hyperparameters'].items():
            if key != 'name':
                print(f"     {key}: {value}")
    
    env.close()
    return results

def main_training_experiment():
    """
    Main REINFORCE training experiment with improved stability
    """
    print("REINFORCE Main Training Experiment")
    print("=" * 50)
    
    # Create training environment
    env = TrafficJunctionEnv(render_mode=None)
    
    # Create REINFORCE agent with stability improvements
    agent = REINFORCEAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=0.0005,  # Reduced for stability
        gamma=0.99,
        use_baseline=True
    )
    
    # Train the agent
    training_stats = agent.train(
        env=env,
        num_episodes=1000,  # Reduced from 1500
        eval_frequency=50,  # More frequent evaluation
        save_path="models/reinforce/",
        verbose=True
    )
    
    if training_stats:
        # Final evaluation using best model
        print("\nFINAL EVALUATION")
        print("=" * 30)
        
        # Load best model for evaluation
        best_model_path = "models/reinforce/reinforce_best.pth"
        if os.path.exists(best_model_path):
            agent.load_model(best_model_path)
            print("Loaded best model for evaluation")
        
        mean_reward, std_reward = agent.evaluate(env, n_episodes=20)
        
        # Save evaluation results
        evaluation_results = {
            'algorithm': 'REINFORCE',
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'best_training_reward': training_stats['best_avg_reward'],
            'early_stop': training_stats['early_stop'],
            'hyperparameters': agent.hyperparameters,
            'training_stats': training_stats
        }
        
        results_path = "models/reinforce/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved: {results_path}")
        
        # Print summary
        print(f"\nTRAINING SUMMARY:")
        print(f"  Best Training Performance: {training_stats['best_avg_reward']:.2f}")
        print(f"  Final Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Early Stop: {training_stats['early_stop']}")
        print(f"  Episodes Completed: {training_stats['num_episodes']}")
    
    # Clean up
    env.close()
    
    return agent

if __name__ == "__main__":
    print("Rwanda Traffic Flow Optimization - REINFORCE Training")
    print("Algorithm: REINFORCE (Policy Gradient Method) with Stability Improvements")
    print()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run hyperparameter tuning first
    print("\n" + "="*60)
    tuning_results = hyperparameter_tuning_experiment()
    
    # Run main training experiment
    print("\n" + "="*60)
    trained_agent = main_training_experiment()
    
    print("\nREINFORCE training pipeline completed!")