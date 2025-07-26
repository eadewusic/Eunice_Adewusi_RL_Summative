"""
DQN Training Script for Rwanda Traffic Junction Environment

This script implements Deep Q-Network (DQN) training using Stable Baselines3
for the traffic light optimization task.

DQN is a value-based method that learns Q-values for state-action pairs,
making it suitable for discrete action spaces like traffic light control.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Environment imports
from environment.traffic_junction_env import TrafficJunctionEnv
from training.training_logger import TrainingLogger, StepMetricsCollector, create_training_plots

class DQNLoggingCallback(BaseCallback):
    """
    Custom callback for DQN that logs detailed training metrics to CSV
    """
    
    def __init__(self, algorithm_name: str = "DQN", verbose: int = 0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.training_logger = None  # Changed from self.logger to self.training_logger
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_start_step = 0
        
    def _on_training_start(self) -> None:
        """Initialize logger when training starts"""
        self.training_logger = TrainingLogger(self.algorithm_name)  # Updated reference
        print(f"Started logging {self.algorithm_name} training metrics")
    
    def _on_step(self) -> bool:
        """Called after each step"""
        # Get current environment info
        if hasattr(self.training_env, 'get_attr'):
            # For vectorized environments
            env_infos = self.training_env.get_attr('unwrapped')[0]
            if hasattr(env_infos, 'get_info'):
                info = env_infos.get_info()
            else:
                info = {}
        else:
            info = {}
        
        # Extract step metrics
        step_data = StepMetricsCollector.extract_step_metrics(
            env_info=info,
            episode=self.episode_count,
            step_in_episode=self.current_episode_length,
            action=0,  # Would need to track this separately for DQN
            reward=0,  # Would need to track this separately 
            cumulative_reward=self.current_episode_reward
        )
        
        # Log step
        if self.training_logger:  # Updated reference
            self.training_logger.log_step(step_data)
        
        self.current_episode_length += 1
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (episode for DQN)"""
        # Log episode completion
        if self.training_logger and self.current_episode_length > 0:  # Updated reference
            episode_data = {
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'vehicles_processed': 0,  # Would need to extract from env
                'total_waiting_time': 0,
                'final_queue_length': 0
            }
            
            self.training_logger.log_episode(episode_data)  # Updated reference
            self.episode_count += 1
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def _on_training_end(self) -> None:
        """Called when training ends"""
        if self.training_logger:  # Updated reference
            self.training_logger.save_final_summary()
            create_training_plots(self.algorithm_name)
            print(f"{self.algorithm_name} training metrics saved to CSV files")

class DQNTrafficAgent:
    """
    DQN agent for traffic light optimization
    """
    
    def __init__(self, 
                 learning_rate: float = 0.0005,
                 buffer_size: int = 50000,
                 learning_starts: int = 1000,
                 batch_size: int = 64,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 target_update_interval: int = 1000,
                 exploration_fraction: float = 0.3,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 verbose: int = 1):
        """
        Initialize DQN agent with hyperparameters
        
        Args:
            learning_rate: Learning rate for the optimizer
            buffer_size: Size of the replay buffer
            learning_starts: Number of steps before learning starts
            batch_size: Batch size for training
            tau: Soft update coefficient for target network
            gamma: Discount factor
            target_update_interval: Steps between target network updates
            exploration_fraction: Fraction of training time for exploration
            exploration_initial_eps: Initial epsilon for epsilon-greedy
            exploration_final_eps: Final epsilon for epsilon-greedy
            verbose: Verbosity level
        """
        
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'target_update_interval': target_update_interval,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps
        }
        
        self.verbose = verbose
        self.model = None
        self.training_history = []
        
    def create_model(self, env) -> DQN:
        """
        Create DQN model with specified hyperparameters
        
        Args:
            env: Training environment
            
        Returns:
            DQN model instance
        """
        
        # Neural network architecture for DQN
        policy_kwargs = dict(
            net_arch=[256, 128, 64],  # Three hidden layers
            activation_fn=torch.nn.ReLU
        )
        
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.hyperparameters['learning_rate'],
            buffer_size=self.hyperparameters['buffer_size'],
            learning_starts=self.hyperparameters['learning_starts'],
            batch_size=self.hyperparameters['batch_size'],
            tau=self.hyperparameters['tau'],
            gamma=self.hyperparameters['gamma'],
            target_update_interval=self.hyperparameters['target_update_interval'],
            exploration_fraction=self.hyperparameters['exploration_fraction'],
            exploration_initial_eps=self.hyperparameters['exploration_initial_eps'],
            exploration_final_eps=self.hyperparameters['exploration_final_eps'],
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
            tensorboard_log="./tensorboard_logs/dqn/",
            device='auto'  # Use GPU if available
        )
        
        return self.model
    
    def train(self, 
              total_timesteps: int = 100000,
              eval_freq: int = 5000,
              n_eval_episodes: int = 10,
              eval_env=None,
              save_path: str = "models/dqn/") -> Dict:
        """
        Train the DQN agent with automatic CSV logging
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of episodes for evaluation
            eval_env: Environment for evaluation
            save_path: Path to save trained model
            
        Returns:
            Training statistics dictionary
        """
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        print(f"Starting DQN Training for {total_timesteps:,} timesteps")
        print("=" * 60)
        print("Hyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print()
        
        # Create evaluation callback
        os.makedirs(save_path, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env if eval_env else self.model.get_env(),
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=self.verbose
        )
        
        # Create logging callback for CSV metrics
        logging_callback = DQNLoggingCallback(algorithm_name="DQN", verbose=self.verbose)
        
        # Combine callbacks
        callbacks = [eval_callback, logging_callback]
        
        # Train the model
        start_time = datetime.now()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            training_time = datetime.now() - start_time
            print(f"\nTraining completed in {training_time}")
            
            # Save final model
            final_model_path = os.path.join(save_path, "dqn_traffic_final.zip")
            self.model.save(final_model_path)
            print(f"Final model saved: {final_model_path}")
            
            # Save hyperparameters
            config_path = os.path.join(save_path, "dqn_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.hyperparameters, f, indent=2)
            
            # Print CSV file locations
            print(f"\nTraining metrics automatically saved:")
            print(f"   Episode metrics: results/training_logs/DQN_episode_metrics.csv")
            print(f"   Step metrics: results/training_logs/DQN_step_metrics.csv")
            print(f"   Training plots: results/training_logs/DQN_training_plots.png")
            
            return {
                'training_time': str(training_time),
                'total_timesteps': total_timesteps,
                'final_model_path': final_model_path,
                'config_path': config_path,
                'episode_metrics_csv': 'results/training_logs/DQN_episode_metrics.csv',
                'step_metrics_csv': 'results/training_logs/DQN_step_metrics.csv'
            }
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            # Still save the current model
            interrupted_path = os.path.join(save_path, "dqn_traffic_interrupted.zip")
            self.model.save(interrupted_path)
            print(f"Interrupted model saved: {interrupted_path}")
            return None
    
    def evaluate(self, env, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the trained DQN agent
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes for evaluation
            
        Returns:
            Mean reward and standard deviation
        """
        
        if self.model is None:
            raise ValueError("No trained model available")
        
        print(f"Evaluating DQN agent over {n_episodes} episodes...")
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            render=False
        )
        
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return mean_reward, std_reward
    
    def load_model(self, model_path: str):
        """Load a pre-trained DQN model"""
        self.model = DQN.load(model_path)
        print(f"DQN model loaded from: {model_path}")
    
    def predict(self, observation, deterministic: bool = True):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("No trained model available")
        
        return self.model.predict(observation, deterministic=deterministic)

def hyperparameter_tuning_experiment():
    """
    Run hyperparameter tuning experiment for DQN
    """
    print("DQN Hyperparameter Tuning Experiment")
    print("=" * 50)
    
    # Create environment for tuning
    env = TrafficJunctionEnv(render_mode=None)
    env = Monitor(env)
    
    # Hyperparameter combinations to test
    hyperparameter_configs = [
        # Configuration 1: Conservative
        {
            'name': 'conservative',
            'learning_rate': 0.0001,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_initial_eps': 0.8,
            'exploration_final_eps': 0.1
        },
        # Configuration 2: Standard
        {
            'name': 'standard',
            'learning_rate': 0.0005,
            'batch_size': 64,
            'gamma': 0.99,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05
        },
        # Configuration 3: Aggressive
        {
            'name': 'aggressive',
            'learning_rate': 0.001,
            'batch_size': 128,
            'gamma': 0.95,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.02
        }
    ]
    
    results = []
    
    for config in hyperparameter_configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Create agent with specific hyperparameters
        agent = DQNTrafficAgent(
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            exploration_initial_eps=config['exploration_initial_eps'],
            exploration_final_eps=config['exploration_final_eps'],
            verbose=0  # Reduced verbosity for tuning
        )
        
        # Create model
        agent.create_model(env)
        
        # Train for shorter duration for tuning
        training_stats = agent.train(
            total_timesteps=20000,
            eval_freq=2000,
            save_path=f"models/dqn_tuning/{config['name']}/"
        )
        
        if training_stats:
            # Evaluate performance
            mean_reward, std_reward = agent.evaluate(env, n_episodes=5)
            
            results.append({
                'config_name': config['name'],
                'hyperparameters': config,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            })
    
    # Report results
    print("\nHYPERPARAMETER TUNING RESULTS")
    print("=" * 50)
    
    best_config = None
    best_reward = float('-inf')
    
    for result in results:
        print(f"\nConfiguration: {result['config_name']}")
        print(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        
        if result['mean_reward'] > best_reward:
            best_reward = result['mean_reward']
            best_config = result
    
    if best_config:
        print(f"\nBest Configuration: {best_config['config_name']}")
        print(f"   Reward: {best_config['mean_reward']:.2f}")
        print("   Hyperparameters:")
        for key, value in best_config['hyperparameters'].items():
            if key != 'name':
                print(f"     {key}: {value}")
    
    env.close()
    return results

def main_training_experiment():
    """
    Main DQN training experiment with best hyperparameters
    """
    print("DQN Main Training Experiment")
    print("=" * 50)
    
    # Create training environment
    env = TrafficJunctionEnv(render_mode=None)
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = TrafficJunctionEnv(render_mode=None)
    eval_env = Monitor(eval_env)
    
    # Create DQN agent with optimized hyperparameters
    agent = DQNTrafficAgent(
        learning_rate=0.0005,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1
    )
    
    # Create model
    agent.create_model(env)
    
    # Train the agent
    training_stats = agent.train(
        total_timesteps=100000,
        eval_freq=5000,
        n_eval_episodes=10,
        eval_env=eval_env,
        save_path="models/dqn/"
    )
    
    if training_stats:
        # Final evaluation
        print("\nFINAL EVALUATION")
        print("=" * 30)
        
        mean_reward, std_reward = agent.evaluate(eval_env, n_episodes=20)
        
        # Save evaluation results
        evaluation_results = {
            'algorithm': 'DQN',
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'hyperparameters': agent.hyperparameters,
            'training_stats': training_stats
        }
        
        results_path = "models/dqn/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved: {results_path}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    return agent

if __name__ == "__main__":
    import torch
    
    print("Rwanda Traffic Flow Optimization - DQN Training")
    print("Assignment: Mission-Based Reinforcement Learning")
    print("Algorithm: Deep Q-Network (Value-Based Method)")
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
    
    print("\nDQN training pipeline completed!")