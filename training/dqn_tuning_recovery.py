"""
Complete DQN Hyperparameter Recovery Script
Recovers the 2 missing hyperparameter configurations identified by diagnosis

Matches main training parameters for proper comparison:
- 100,000 timesteps (same as main)
- 5,000 eval frequency (same as main)
- 10 eval episodes (same as main)
- Same logging and callbacks as main
"""


import sys
import os
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment.traffic_junction_env import TrafficJunctionEnv
from training.training_logger import TrainingLogger, StepMetricsCollector, create_training_plots

class DQNLoggingCallback(BaseCallback):
    """
    Custom callback for DQN that logs detailed training metrics to CSV
    (Same as main training)
    """
    
    def __init__(self, algorithm_name: str = "DQN", verbose: int = 0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.training_logger = None
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_training_start(self) -> None:
        """Initialize logger when training starts"""
        self.training_logger = TrainingLogger(self.algorithm_name)
        print(f"Started logging {self.algorithm_name} training metrics")
    
    def _on_step(self) -> bool:
        """Called after each step"""
        # Get current environment info
        if hasattr(self.training_env, 'get_attr'):
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
            action=0,
            reward=0,
            cumulative_reward=self.current_episode_reward
        )
        
        # Log step
        if self.training_logger:
            self.training_logger.log_step(step_data)
        
        self.current_episode_length += 1
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        if self.training_logger and self.current_episode_length > 0:
            episode_data = {
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'vehicles_processed': 0,
                'total_waiting_time': 0,
                'final_queue_length': 0
            }
            
            self.training_logger.log_episode(episode_data)
            self.episode_count += 1
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def _on_training_end(self) -> None:
        """Called when training ends"""
        if self.training_logger:
            self.training_logger.save_final_summary()
            create_training_plots(self.algorithm_name)
            print(f"{self.algorithm_name} training metrics saved to CSV files")

def train_single_config(config_name, config_params):
    """Train a single hyperparameter configuration with main training parameters"""
    
    print(f"\nTRAINING: {config_name.upper()}")
    print("-" * 60)
    print(f"Using MAIN TRAINING parameters for fair comparison:")
    print(f"  - Timesteps: 100,000 (same as main)")
    print(f"  - Eval frequency: 5,000 (same as main)")
    print(f"  - Eval episodes: 10 (same as main)")
    print(f"  - Full logging enabled")
    print()
    print(f"Hyperparameters: {config_params}")
    
    # Set seeds for reproducibility
    seed = config_params['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create save directory
    save_dir = Path("models/dqn_tuning") / config_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Create training environment
    env = TrafficJunctionEnv(render_mode=None)
    env = Monitor(env)
    
    # Create evaluation environment (separate like main training)
    eval_env = TrafficJunctionEnv(render_mode=None)
    eval_env = Monitor(eval_env)

    # Set seeds using gymnasium method
    try:
        env.reset(seed=seed)
        eval_env.reset(seed=seed + 1)  # Different seed for eval
    except:
        try:
            env.seed(seed)
            eval_env.seed(seed + 1)
        except:
            print(f"Warning: Could not set environment seed for {config_name}")
            pass

    try:
        # Create DQN model with SAME parameters as main training
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config_params['learning_rate'],
            buffer_size=50000,  # Same as main
            learning_starts=1000,  # Same as main
            batch_size=config_params['batch_size'],
            tau=1.0,  # Same as main
            gamma=config_params['gamma'],
            target_update_interval=1000,  # Same as main
            exploration_fraction=0.3,  # Same as main
            exploration_initial_eps=config_params['exploration_initial_eps'],
            exploration_final_eps=config_params['exploration_final_eps'],
            policy_kwargs=dict(net_arch=[256, 128, 64], activation_fn=torch.nn.ReLU),  # Same as main
            verbose=1,
            seed=seed,
            tensorboard_log=f"./tensorboard_logs/dqn_tuning/{config_name}/",
            device='auto'
        )
        
        # Create evaluation callback with SAME parameters as main training
        eval_callback = EvalCallback(
            eval_env,  # Separate eval environment like main
            best_model_save_path=str(save_dir),
            log_path=str(save_dir),
            eval_freq=5000,  # Same as main (was 2000 in old version)
            n_eval_episodes=10,  # Same as main (was 5 in old version)
            deterministic=True,
            render=False,
            verbose=1  # Same as main
        )
        
        # Create logging callback like main training
        logging_callback = DQNLoggingCallback(
            algorithm_name=f"DQN_{config_name.upper()}", 
            verbose=1
        )
        
        # Combine callbacks like main training
        callbacks = [eval_callback, logging_callback]
        
        print(f"Starting training for {config_name} with 100,000 timesteps...")
        start_time = datetime.now()
        
        # Train the model with SAME timesteps as main training
        model.learn(
            total_timesteps=100000,  # Same as main (was 20000 in old version)
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = datetime.now() - start_time
        print(f"Training completed in: {training_time}")
        
        # Save final model
        final_model_path = save_dir / f"dqn_{config_name}_final.zip"
        model.save(str(final_model_path))
        
        # Final evaluation with SAME parameters as main training
        print(f"Evaluating {config_name} performance over 20 episodes...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True, render=False  # Same as main
        )
        
        # Save hyperparameters config like main training
        hyperparams_path = save_dir / f"dqn_{config_name}_config.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(config_params, f, indent=2)
        
        # Save comprehensive results like main training
        evaluation_results = {
            'algorithm': f'DQN_{config_name.upper()}',
            'config_name': config_name,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'hyperparameters': config_params,
            'training_stats': {
                'training_time': str(training_time),
                'total_timesteps': 100000,
                'final_model_path': str(final_model_path),
                'config_path': str(hyperparams_path),
                'episode_metrics_csv': f'results/training_logs/DQN_{config_name.upper()}_episode_metrics.csv',
                'step_metrics_csv': f'results/training_logs/DQN_{config_name.upper()}_step_metrics.csv'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = save_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Verify files were saved
        saved_files = list(save_dir.glob("*"))
        
        print(f"\n{config_name.upper()} COMPLETED SUCCESSFULLY!")
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Training Time: {training_time}")
        print(f"   Files Saved: {len(saved_files)}")
        for file_path in saved_files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"     - {file_path.name} ({size:,} bytes)")
        
        print(f"\n   Training metrics saved:")
        print(f"     Episode metrics: results/training_logs/DQN_{config_name.upper()}_episode_metrics.csv")
        print(f"     Step metrics: results/training_logs/DQN_{config_name.upper()}_step_metrics.csv")
        print(f"     Training plots: results/training_logs/DQN_{config_name.upper()}_training_plots.png")
        
        env.close()
        eval_env.close()
        return evaluation_results
        
    except Exception as e:
        print(f"ERROR training {config_name}: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        eval_env.close()
        return None

def main():
    """Main recovery function with comparison"""
    
    print("DQN HYPERPARAMETER COMPARISON")
    print("=" * 60)
    print("Training ALL configurations with MAIN TRAINING parameters:")
    print("  - 100,000 timesteps (same as main)")
    print("  - 5,000 evaluation frequency (same as main)")
    print("  - 10 evaluation episodes (same as main)")
    print("  - 20 final evaluation episodes (same as main)")
    print("  - Full CSV logging enabled (same as main)")
    print("  - Separate eval environment (same as main)")
    print()
    print("This will take significantly longer but provide fair comparison!")
    print()
    
    # Check current device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define the configurations to test (same as before)
    configs = {
        'conservative': {
            'learning_rate': 0.0001,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_initial_eps': 0.8,
            'exploration_final_eps': 0.1,
            'seed': 42
        },
        'aggressive': {
            'learning_rate': 0.001,
            'batch_size': 128,
            'gamma': 0.95,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.02,
            'seed': 456
        }
    }
    
    print(f"Configurations to train: {list(configs.keys())}")
    print(f"Expected training time: ~17 minutes per config (3 configs × 5:37 each)")
    print()
    
    # Train each configuration
    results = []
    
    for i, (config_name, config_params) in enumerate(configs.items(), 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {i}/3: {config_name.upper()}")
        print(f"{'='*80}")
        
        result = train_single_config(config_name, config_params)
        if result:
            results.append(result)
        else:
            print(f"Failed to train {config_name}")
    
    # Generate final summary with main training comparison
    if results:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"Successfully trained: {len(results)}/3 configurations")
        print()
        
        # Add main training result for comparison
        main_result = {
            'config_name': 'main_training',
            'mean_reward': -1371.20,
            'std_reward': 644.80,
            'hyperparameters': {
                'learning_rate': 0.0005,
                'batch_size': 64,
                'gamma': 0.99,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05
            }
        }
        
        # Sort by performance (include main training)
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
        print(f"   Hyperparameters:")
        for key, value in best_config['hyperparameters'].items():
            if key != 'seed':
                print(f"     {key}: {value}")
        
        # Save comprehensive summary
        summary_path = Path("models/dqn_tuning/hyperparameter_comparison.json")
        summary_data = {
            'comparison_type': 'Comparison - Same Training Conditions',
            'training_parameters': {
                'total_timesteps': 100000,
                'eval_freq': 5000,
                'n_eval_episodes': 10,
                'final_eval_episodes': 20,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'target_update_interval': 1000,
                'exploration_fraction': 0.3
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
        
        print(f"\nCOMPARISON COMPLETED!")
        print(f"   Main training results: models/dqn/ (unchanged)")
        print(f"   Hyperparameter results: models/dqn_tuning/")
        print(f"   Now you have a true hyperparameter comparison!")
        
    else:
        print(f"\nCOMPARISON FAILED")
        print("No configurations were successfully trained.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()