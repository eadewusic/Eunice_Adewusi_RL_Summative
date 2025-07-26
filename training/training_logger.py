"""
Training Metrics Logger

Automatically saves training metrics to CSV files during training.
Creates detailed episode-by-episode logs for analysis.
"""

import os
import csv
import time
from typing import Dict, List, Optional
import numpy as np
from collections import deque

class TrainingLogger:
    """
    Logs training metrics automatically to CSV files
    """
    
    def __init__(self, algorithm_name: str, log_dir: str = "results/training_logs/"):
        """
        Initialize training logger
        
        Args:
            algorithm_name: Name of the RL algorithm
            log_dir: Directory to save log files
        """
        self.algorithm_name = algorithm_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file paths
        self.episode_log_path = os.path.join(log_dir, f"{algorithm_name}_episode_metrics.csv")
        self.step_log_path = os.path.join(log_dir, f"{algorithm_name}_step_metrics.csv")
        
        # Tracking variables
        self.global_step = 0
        self.episode_count = 0
        self.start_time = time.time()
        self.recent_rewards = deque(maxlen=100)
        
        # Initialize CSV files
        self._initialize_csv_files()
        
        print(f"Training logger initialized for {algorithm_name}")
        print(f"   Episode log: {self.episode_log_path}")
        print(f"   Step log: {self.step_log_path}")
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        
        # Episode metrics CSV
        episode_headers = [
            'episode', 'total_steps', 'episode_reward', 'episode_length',
            'mean_reward_100', 'vehicles_processed', 'total_waiting_time',
            'final_queue_length', 'convergence_metric', 'training_time_hours'
        ]
        
        with open(self.episode_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_headers)
        
        # Step metrics CSV  
        step_headers = [
            'global_step', 'episode', 'step_in_episode', 'reward', 'cumulative_reward',
            'action_taken', 'queue_north', 'queue_south', 'queue_east', 'queue_west',
            'total_queue', 'vehicles_processed', 'light_state', 'hidden_state'
        ]
        
        with open(self.step_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(step_headers)
    
    def log_step(self, step_data: Dict):
        """
        Log step-level metrics
        
        Args:
            step_data: Dictionary containing step information
        """
        self.global_step += 1
        
        step_row = [
            self.global_step,
            step_data.get('episode', self.episode_count),
            step_data.get('step_in_episode', 0),
            step_data.get('reward', 0),
            step_data.get('cumulative_reward', 0),
            step_data.get('action', 0),
            step_data.get('queue_north', 0),
            step_data.get('queue_south', 0), 
            step_data.get('queue_east', 0),
            step_data.get('queue_west', 0),
            step_data.get('total_queue', 0),
            step_data.get('vehicles_processed', 0),
            step_data.get('light_state', 'unknown'),
            step_data.get('hidden_state', 'unknown')
        ]
        
        # Append to step CSV
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(step_row)
    
    def log_episode(self, episode_data: Dict):
        """
        Log episode-level metrics
        
        Args:
            episode_data: Dictionary containing episode information
        """
        self.episode_count += 1
        episode_reward = episode_data.get('episode_reward', 0)
        self.recent_rewards.append(episode_reward)
        
        # Calculate metrics
        mean_reward_100 = np.mean(self.recent_rewards) if self.recent_rewards else 0
        training_time_hours = (time.time() - self.start_time) / 3600
        
        # Convergence metric (improvement rate)
        if len(self.recent_rewards) >= 10:
            recent_10 = list(self.recent_rewards)[-10:]
            prev_10 = list(self.recent_rewards)[-20:-10] if len(self.recent_rewards) >= 20 else recent_10
            convergence_metric = np.mean(recent_10) - np.mean(prev_10)
        else:
            convergence_metric = 0
        
        episode_row = [
            self.episode_count,
            self.global_step,
            episode_reward,
            episode_data.get('episode_length', 0),
            mean_reward_100,
            episode_data.get('vehicles_processed', 0),
            episode_data.get('total_waiting_time', 0),
            episode_data.get('final_queue_length', 0),
            convergence_metric,
            training_time_hours
        ]
        
        # Append to episode CSV
        with open(self.episode_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_row)
        
        # Print progress every 50 episodes
        if self.episode_count % 50 == 0:
            print(f"Episode {self.episode_count}: Reward={episode_reward:.2f}, "
                  f"Mean(100)={mean_reward_100:.2f}, Steps={self.global_step}")
    
    def get_current_stats(self) -> Dict:
        """Get current training statistics"""
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.global_step,
            'mean_reward_100': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'training_time_hours': (time.time() - self.start_time) / 3600,
            'episode_log_path': self.episode_log_path,
            'step_log_path': self.step_log_path
        }
    
    def save_final_summary(self, final_stats: Optional[Dict] = None):
        """Save final training summary"""
        summary_path = os.path.join(self.log_dir, f"{self.algorithm_name}_training_summary.txt")
        
        stats = self.get_current_stats()
        if final_stats:
            stats.update(final_stats)
        
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary - {self.algorithm_name}\n")
            f.write("="*50 + "\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Training summary saved: {summary_path}")

class StepMetricsCollector:
    """
    Helper class to collect step metrics from environment info
    """
    
    @staticmethod
    def extract_step_metrics(env_info: Dict, episode: int, step_in_episode: int, 
                           action: int, reward: float, cumulative_reward: float) -> Dict:
        """
        Extract step metrics from environment info
        
        Args:
            env_info: Environment info dictionary
            episode: Current episode number
            step_in_episode: Step within current episode
            action: Action taken
            reward: Step reward
            cumulative_reward: Cumulative episode reward
            
        Returns:
            Dictionary of step metrics
        """
        
        # Parse queue information (assuming env stores this in info)
        total_waiting = env_info.get('total_vehicles_waiting', 0)
        
        return {
            'episode': episode,
            'step_in_episode': step_in_episode,
            'reward': reward,
            'cumulative_reward': cumulative_reward,
            'action': action,
            'queue_north': env_info.get('queue_north', 0),
            'queue_south': env_info.get('queue_south', 0),
            'queue_east': env_info.get('queue_east', 0),
            'queue_west': env_info.get('queue_west', 0),
            'total_queue': total_waiting,
            'vehicles_processed': env_info.get('vehicles_processed', 0),
            'light_state': env_info.get('current_light', 'unknown'),
            'hidden_state': env_info.get('hidden_state', 'unknown')
        }
    
    @staticmethod
    def extract_episode_metrics(episode_reward: float, episode_length: int, 
                              final_info: Dict) -> Dict:
        """
        Extract episode metrics
        
        Args:
            episode_reward: Total episode reward
            episode_length: Episode length in steps
            final_info: Final environment info
            
        Returns:
            Dictionary of episode metrics
        """
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'vehicles_processed': final_info.get('vehicles_processed', 0),
            'total_waiting_time': episode_length * final_info.get('total_vehicles_waiting', 0),
            'final_queue_length': final_info.get('total_vehicles_waiting', 0)
        }

def create_training_plots(algorithm_name: str, log_dir: str = "results/training_logs/"):
    """
    Create training plots from logged CSV data
    
    Args:
        algorithm_name: Name of algorithm
        log_dir: Directory containing log files
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    episode_log_path = os.path.join(log_dir, f"{algorithm_name}_episode_metrics.csv")
    
    if not os.path.exists(episode_log_path):
        print(f"No episode log found for {algorithm_name}")
        return
    
    # Load data
    df = pd.read_csv(episode_log_path)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{algorithm_name} Training Metrics', fontsize=16)
    
    # Episode rewards
    axes[0,0].plot(df['episode'], df['episode_reward'], alpha=0.6, label='Episode Reward')
    axes[0,0].plot(df['episode'], df['mean_reward_100'], color='red', linewidth=2, label='Mean (100 episodes)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].set_title('Episode Rewards Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0,1].plot(df['episode'], df['episode_length'])
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Episode Length')
    axes[0,1].set_title('Episode Length Over Time')
    axes[0,1].grid(True, alpha=0.3)
    
    # Vehicles processed
    axes[1,0].plot(df['episode'], df['vehicles_processed'])
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Vehicles Processed')
    axes[1,0].set_title('Traffic Efficiency Over Time')
    axes[1,0].grid(True, alpha=0.3)
    
    # Training progress
    axes[1,1].plot(df['episode'], df['convergence_metric'])
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Convergence Metric')
    axes[1,1].set_title('Training Convergence')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(log_dir, f"{algorithm_name}_training_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved: {plot_path}")

if __name__ == "__main__":
    # Example usage
    logger = TrainingLogger("DQN_test")
    
    # Simulate some training data
    for episode in range(10):
        episode_reward = 0
        for step in range(50):
            step_data = {
                'episode': episode,
                'step_in_episode': step,
                'reward': np.random.randn(),
                'action': np.random.randint(0, 9),
                'total_queue': np.random.randint(0, 20)
            }
            episode_reward += step_data['reward']
            step_data['cumulative_reward'] = episode_reward
            logger.log_step(step_data)
        
        episode_data = {
            'episode_reward': episode_reward,
            'episode_length': 50,
            'vehicles_processed': np.random.randint(10, 30)
        }
        logger.log_episode(episode_data)
    
    # Create plots
    create_training_plots("DQN_test")
    
    print("Training logger test completed")