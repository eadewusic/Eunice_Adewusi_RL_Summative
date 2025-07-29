"""

Complete REINFORCE Hyperparameter Recovery Script
Recovers the 2 missing hyperparameter configurations identified by diagnosis

Matches main training parameters for proper comparison:
- 1000 episodes (same as main)
- eval_frequency=50 (same as main)
- max_steps_per_episode=500 (same as main)
- Early stopping with patience (same as main)
- Same logging and evaluation as main
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
from datetime import datetime
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_junction_env import TrafficJunctionEnv
from training.training_logger import TrainingLogger, StepMetricsCollector, create_training_plots
from torch.utils.tensorboard import SummaryWriter

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """Value network for baseline (variance reduction)"""
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class REINFORCEAgent:
    """REINFORCE agent for hyperparameter comparison"""
    
    def __init__(self, state_size: int, action_size: int, config_name: str, hyperparams: dict, seed: int = 42):
        self.config_name = config_name
        self.hyperparameters = hyperparams
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_size, action_size).to(self.device)
        
        self.use_baseline = hyperparams['use_baseline']
        if self.use_baseline:
            self.value_net = ValueNetwork(state_size).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=hyperparams['learning_rate'])
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=hyperparams['learning_rate'])
        
        # Learning rate schedulers (same as main)
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=300, gamma=0.8)
        if self.use_baseline:
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
        
        # Logging components
        self.training_logger = None
        self.tensorboard_writer = None
        
        # Early stopping variables (same as main)
        self.best_avg_reward = float('-inf')
        self.patience = 5
        self.patience_counter = 0
        self.early_stop = False
    
    def get_action(self, state: np.ndarray, training: bool = True):
        """Get action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(training):
            action_probs = self.policy_net(state_tensor)
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum()
            
            if training:
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                return action.item(), log_prob
            else:
                action = torch.argmax(action_probs, dim=1)
                return action.item(), None
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, log_prob: torch.Tensor):
        """Store transition data for episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
    
    def calculate_returns(self):
        """Calculate discounted returns for episode"""
        returns = []
        G = 0
        
        for reward in reversed(self.episode_rewards):
            G = reward + self.hyperparameters['gamma'] * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """Update policy network using REINFORCE algorithm"""
        if len(self.episode_rewards) == 0:
            return
        
        returns = self.calculate_returns()
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
        """Check if training should stop early (same as main)"""
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
    
    def train_and_save(self, env, save_base_path: str = "models/reinforce_tuning"):
        """Train with main training parameters for comparison"""
        
        print(f"\nTRAINING: {self.config_name.upper()}")
        print("-" * 60)
        print(f"Using main REINFORCE training parameters for comparison:")
        print(f"  - Episodes: 1000 (same as main)")
        print(f"  - Eval frequency: 50 (same as main)")
        print(f"  - Max steps per episode: 500 (same as main)")
        print(f"  - Early stopping with patience: 5 (same as main)")
        print(f"  - Learning rate scheduling (same as main)")
        print(f"  - 20 final evaluation episodes (same as main)")
        print(f"  - Full logging enabled")
        print()
        print(f"Hyperparameters: {self.hyperparameters}")
        
        # Create save directory
        save_dir = Path(save_base_path) / self.config_name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save directory: {save_dir}")
        
        # Initialize logging (same as main)
        self.training_logger = TrainingLogger(f"REINFORCE_{self.config_name.upper()}")
        
        # Initialize tensorboard
        os.makedirs(f"./tensorboard_logs/reinforce_tuning/{self.config_name}/", exist_ok=True)
        self.tensorboard_writer = SummaryWriter(f"./tensorboard_logs/reinforce_tuning/{self.config_name}/REINFORCE_1")
        
        print(f"Started logging REINFORCE_{self.config_name.upper()} training metrics")
        
        # Training parameters (same as main)
        num_episodes = 1000
        max_steps_per_episode = 500
        eval_frequency = 50
        
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
                    
                    if terminated or truncated:
                        break
                
                # Update policy after episode
                self.update_policy()
                
                # Log episode metrics (same as main)
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
                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(recent_rewards)
                    current_lr = self.policy_optimizer.param_groups[0]['lr']
                    
                    print(f"Episode {episode + 1:4d}: "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg(100): {avg_reward:7.2f} | "
                          f"Length: {episode_length:3d} | "
                          f"LR: {current_lr:.6f} | "
                          f"Policy Loss: {self.policy_losses[-1] if self.policy_losses else 0:.4f}")
                
                # Evaluation and early stopping check (same as main)
                if (episode + 1) % eval_frequency == 0:
                    avg_reward = np.mean(recent_rewards)
                    
                    # Save best model
                    if avg_reward > self.best_avg_reward:
                        best_model_path = save_dir / f"reinforce_{self.config_name}_best.pth"
                        self.save_model(str(best_model_path))
                        print(f"New best average reward: {avg_reward:.2f} - Model saved")
                    
                    # Check for early stopping
                    if self.check_early_stopping(avg_reward, episode):
                        self.early_stop = True
                        print(f"Early stopping at episode {episode + 1}")
                        break
            
            # Training completed - finalize logging (same as main)
            if self.training_logger:
                self.training_logger.save_final_summary()
                create_training_plots(f"REINFORCE_{self.config_name.upper()}")
                print(f"REINFORCE_{self.config_name.upper()} training metrics saved to CSV files")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            training_time = datetime.now() - start_time
            print(f"Training completed in: {training_time}")
            
            if self.early_stop:
                print(f"Training stopped early with best performance: {self.best_avg_reward:.2f}")
            
            # Save final model
            final_model_path = save_dir / f"reinforce_{self.config_name}_final.pth"
            self.save_model(str(final_model_path))
            
            # Final evaluation with SAME parameters as main training
            print(f"Evaluating {self.config_name} performance over 20 episodes...")
            
            # Load best model for evaluation (same as main)
            best_model_path = save_dir / f"reinforce_{self.config_name}_best.pth"
            if best_model_path.exists():
                self.load_model(str(best_model_path))
                print("Loaded best model for evaluation")
            
            mean_reward, std_reward = self.evaluate(env, n_episodes=20)
            
            # Save comprehensive results
            evaluation_results = {
                'algorithm': f'REINFORCE_{self.config_name.upper()}',
                'config_name': self.config_name,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'best_training_reward': self.best_avg_reward,
                'early_stop': self.early_stop,
                'hyperparameters': self.hyperparameters,
                'training_stats': {
                    'training_time': str(training_time),
                    'num_episodes': episode + 1,
                    'final_model_path': str(final_model_path),
                    'episode_metrics_csv': f'results/training_logs/REINFORCE_{self.config_name.upper()}_episode_metrics.csv',
                    'step_metrics_csv': f'results/training_logs/REINFORCE_{self.config_name.upper()}_step_metrics.csv'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = save_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            # Save hyperparameters config
            config_path = save_dir / f"reinforce_{self.config_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.hyperparameters, f, indent=2)
            
            # Save training history
            history_path = save_dir / "training_history.json"
            history = {
                'episode_rewards': self.episode_rewards_history,
                'episode_lengths': self.episode_lengths_history,
                'policy_losses': self.policy_losses,
                'value_losses': self.value_losses if self.use_baseline else [],
                'best_avg_reward': self.best_avg_reward,
                'early_stop': self.early_stop
            }
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Verify files were saved
            saved_files = list(save_dir.glob("*"))
            
            print(f"\n{self.config_name.upper()} COMPLETED SUCCESSFULLY!")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Best Training Reward: {self.best_avg_reward:.2f}")
            print(f"   Early Stop: {self.early_stop}")
            print(f"   Episodes Completed: {episode + 1}")
            print(f"   Training Time: {training_time}")
            print(f"   Files Saved: {len(saved_files)}")
            for file_path in saved_files:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"     - {file_path.name} ({size:,} bytes)")
            
            print(f"\n   Training metrics saved:")
            print(f"     Episode metrics: results/training_logs/REINFORCE_{self.config_name.upper()}_episode_metrics.csv")
            print(f"     Step metrics: results/training_logs/REINFORCE_{self.config_name.upper()}_step_metrics.csv")
            print(f"     Training plots: results/training_logs/REINFORCE_{self.config_name.upper()}_training_plots.png")
            
            return evaluation_results
            
        except Exception as e:
            print(f"ERROR training {self.config_name}: {e}")
            import traceback
            traceback.print_exc()
            
            if self.training_logger:
                self.training_logger.save_final_summary()
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            return None
    
    def evaluate(self, env, n_episodes: int = 20):
        """Evaluate the trained agent"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
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
        
        return mean_reward, std_reward
    
    def save_model(self, filepath: str):
        """Save model state"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
            'hyperparameters': self.hyperparameters,
            'episode_rewards_history': self.episode_rewards_history,
            'best_avg_reward': self.best_avg_reward,
            'config_name': self.config_name
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

def main():
    """Main comparison function"""
    
    print("REINFORCE HYPERPARAMETER COMPARISON")
    print("=" * 60)
    print("Training ALL configurations with main training parameters:")
    print("  - 1000 episodes (same as main)")
    print("  - Eval frequency: 50 (same as main)")
    print("  - Max steps per episode: 500 (same as main)")
    print("  - Early stopping with patience: 5 (same as main)")
    print("  - Learning rate scheduling (same as main)")
    print("  - 20 final evaluation episodes (same as main)")
    print("  - Full CSV logging enabled (same as main)")
    print()
    print("This will provide fair comparison!")
    print()
    
    # Check current device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create environment
    env = TrafficJunctionEnv(render_mode=None)
    
    # Define configurations to test (with improved stability like main)
    configs = {
        'conservative': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'use_baseline': True,
            'seed': 42
        },
        'moderate': {
            'learning_rate': 0.0008,
            'gamma': 0.99,
            'use_baseline': True,
            'seed': 456
        }
    }
    
    print(f"Configurations to train: {list(configs.keys())}")
    print(f"Expected training time: ~10-15 minutes per config (with early stopping)")
    print()
    
    results = []
    
    for i, (config_name, config_params) in enumerate(configs.items(), 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {i}/3: {config_name.upper()}")
        print(f"{'='*80}")
        
        agent = REINFORCEAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config_name=config_name,
            hyperparams=config_params,
            seed=config_params['seed']
        )
        
        result = agent.train_and_save(env)
        if result:
            results.append(result)
        else:
            print(f"Failed to train {config_name}")
    
    # Generate final comparison results
    if results:
        print(f"\n{'='*80}")
        print("REINFORCE COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"Successfully trained: {len(results)}/3 configurations")
        print()
        
        # Add main training result for comparison
        main_result = {
            'config_name': 'main_training',
            'mean_reward': -188.95,
            'std_reward': 25.28532182907704,
            'hyperparameters': {
                'learning_rate': 0.0005,
                'gamma': 0.99,
                'use_baseline': True,
                'network_architecture': '128-128-64',
                'device': 'cpu',
                'patience': 5
            }
        }
        
        # Sort by performance
        all_results = results + [main_result]
        all_results.sort(key=lambda x: x['mean_reward'], reverse=True)
        
        print("FINAL PERFORMANCE RANKING:")
        for i, result in enumerate(all_results, 1):
            config_name = result['config_name']
            reward = result['mean_reward']
            std = result['std_reward']
            if config_name == 'main_training':
                print(f"  {i}. {config_name:>15}: {reward:>8.2f} ± {std:.2f} [REFERENCE]")
            else:
                print(f"  {i}. {config_name:>15}: {reward:>8.2f} ± {std:.2f}")
        
        best_config = all_results[0]
        print(f"\nBEST CONFIGURATION: {best_config['config_name'].upper()}")
        print(f"   Reward: {best_config['mean_reward']:.2f}")
        
        # Save comprehensive summary
        summary_path = Path("models/reinforce_tuning/comparison_summary.json")
        summary_data = {
            'comparison_type': 'Comparison - Same Training Conditions',
            'training_parameters': {
                'num_episodes': 1000,
                'eval_frequency': 50,
                'max_steps_per_episode': 500,
                'early_stopping_patience': 5,
                'final_eval_episodes': 20,
                'learning_rate_scheduling': True
            },
            'timestamp': datetime.now().isoformat(),
            'total_configurations': len(configs),
            'successful_configurations': len(results),
            'main_training_reference': main_result,
            'hyperparameter_results': results,
            'all_results_ranked': all_results
        }
        
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nComparison summary saved: {summary_path}")
        
        print(f"\nREINFORCE COMPARISON COMPLETED!")
        print(f"   Main training results: models/reinforce/ (unchanged)")
        print(f"   Hyperparameter results: models/reinforce_tuning/")
        
    else:
        print(f"\nCOMPARISON FAILED")
        print("No configurations were successfully trained.")
    
    env.close()

if __name__ == "__main__":
    main()