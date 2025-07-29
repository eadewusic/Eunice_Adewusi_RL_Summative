"""
Complete PPO Hyperparameter Recovery Script
Recovers the 2 missing hyperparameter configurations identified by diagnosis

Matches main training parameters for proper comparison:
- 200,000 timesteps (same as main)
- eval_freq=10,000 (same as main)
- n_eval_episodes=10 (same as main)
- Vectorized environments (same as main)
- Same logging and evaluation as main
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

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

# Environment imports
from environment.traffic_junction_env import TrafficJunctionEnv
from training.training_logger import TrainingLogger, StepMetricsCollector, create_training_plots

class PPOLoggingCallback(BaseCallback):
    """Custom callback for PPO logging"""
    
    def __init__(self, algorithm_name: str = "PPO", verbose: int = 0):
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
        self.current_episode_length += 1
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        if self.training_logger and self.current_episode_length > 0:
            # Get environment info
            if hasattr(self.training_env, 'get_attr'):
                try:
                    env_infos = self.training_env.get_attr('unwrapped')
                    if env_infos and hasattr(env_infos[0], 'get_info'):
                        info = env_infos[0].get_info()
                    else:
                        info = {}
                except:
                    info = {}
            else:
                info = {}
            
            episode_data = {
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'vehicles_processed': info.get('vehicles_processed', 0),
                'total_waiting_time': info.get('total_waiting_time', 0),
                'final_queue_length': info.get('queue_length', 0)
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

class PPOAgent:
    """PPO agent for hyperparameter comparison"""
    
    def __init__(self, config_name: str, hyperparams: dict, seed: int = 42):
        self.config_name = config_name
        self.hyperparameters = hyperparams
        self.seed = seed
        self.model = None
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def create_vectorized_env(self, n_envs: int = 4):
        """Create vectorized environment (same as main training)"""
        def make_env():
            def _init():
                env = TrafficJunctionEnv(render_mode=None)
                env = Monitor(env)
                return env
            return _init
        
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        return env
    
    def create_model(self, env):
        """Create PPO model with specified hyperparameters"""
        
        # Neural network architecture (same as main)
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 128, 64],  # Policy network architecture
                vf=[256, 128, 64]   # Value function network architecture
            ),
            activation_fn=torch.nn.ReLU
        )
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.hyperparameters['learning_rate'],
            n_steps=self.hyperparameters['n_steps'],
            batch_size=self.hyperparameters['batch_size'],
            n_epochs=self.hyperparameters['n_epochs'],
            gamma=self.hyperparameters['gamma'],
            gae_lambda=self.hyperparameters['gae_lambda'],
            clip_range=self.hyperparameters['clip_range'],
            ent_coef=self.hyperparameters['ent_coef'],
            vf_coef=self.hyperparameters['vf_coef'],
            max_grad_norm=self.hyperparameters['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=self.seed,
            tensorboard_log=f"./tensorboard_logs/ppo_tuning/{self.config_name}/",
            device='auto'
        )
        
        return self.model
    
    def train_and_save(self, save_base_path: str = "models/ppo_tuning"):
        """Train with main training parameters for comparison"""
        
        print(f"\nTRAINING: {self.config_name.upper()}")
        print("-" * 60)
        print(f"Using MAIN TRAINING parameters for comparison:")
        print(f"  - Timesteps: 200,000 (same as main)")
        print(f"  - Eval frequency: 10,000 (same as main)")
        print(f"  - Eval episodes: 10 (same as main)")
        print(f"  - Final eval episodes: 20 (same as main)")
        print(f"  - Vectorized envs: 4 (same as main)")
        print(f"  - Full logging enabled")
        print()
        print(f"Hyperparameters: {self.hyperparameters}")
        
        # Create save directory
        save_dir = Path(save_base_path) / self.config_name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save directory: {save_dir}")
        
        # Create vectorized training environment (same as main)
        train_env = self.create_vectorized_env(n_envs=4)
        
        # Create evaluation environment (same as main)
        eval_env = TrafficJunctionEnv(render_mode=None)
        eval_env = Monitor(eval_env)
        
        try:
            # Create model
            self.create_model(train_env)
            
            # Create evaluation callback (same parameters as main)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir),
                log_path=str(save_dir),
                eval_freq=10000,  # Same as main
                n_eval_episodes=10,  # Same as main
                deterministic=True,
                render=False,
                verbose=1
            )
            
            # Create logging callback
            logging_callback = PPOLoggingCallback(
                algorithm_name=f"PPO_{self.config_name.upper()}", 
                verbose=1
            )
            
            # Combine callbacks
            callbacks = [eval_callback, logging_callback]
            
            print(f"Starting training for {self.config_name} with 200,000 timesteps...")
            start_time = datetime.now()
            
            # Train the model (same timesteps as main)
            self.model.learn(
                total_timesteps=200000,  # Same as main
                callback=callbacks,
                progress_bar=True
            )
            
            training_time = datetime.now() - start_time
            print(f"Training completed in: {training_time}")
            
            # Save final model
            final_model_path = save_dir / f"ppo_{self.config_name}_final.zip"
            self.model.save(str(final_model_path))
            
            # Final evaluation with SAME parameters as main training
            print(f"Evaluating {self.config_name} performance over 20 episodes...")
            mean_reward, std_reward = evaluate_policy(
                self.model, eval_env, n_eval_episodes=20, deterministic=True, render=False
            )
            
            # Save hyperparameters config
            config_path = save_dir / f"ppo_{self.config_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.hyperparameters, f, indent=2)
            
            # Save comprehensive results
            evaluation_results = {
                'algorithm': f'PPO_{self.config_name.upper()}',
                'config_name': self.config_name,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'hyperparameters': self.hyperparameters,
                'training_stats': {
                    'training_time': str(training_time),
                    'total_timesteps': 200000,
                    'final_model_path': str(final_model_path),
                    'config_path': str(config_path),
                    'episode_metrics_csv': f'results/training_logs/PPO_{self.config_name.upper()}_episode_metrics.csv',
                    'step_metrics_csv': f'results/training_logs/PPO_{self.config_name.upper()}_step_metrics.csv'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = save_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            # Verify files were saved
            saved_files = list(save_dir.glob("*"))
            
            print(f"\n{self.config_name.upper()} COMPLETED SUCCESSFULLY!")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Training Time: {training_time}")
            print(f"   Files Saved: {len(saved_files)}")
            for file_path in saved_files:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"     - {file_path.name} ({size:,} bytes)")
            
            print(f"\n   Training metrics saved:")
            print(f"     Episode metrics: results/training_logs/PPO_{self.config_name.upper()}_episode_metrics.csv")
            print(f"     Step metrics: results/training_logs/PPO_{self.config_name.upper()}_step_metrics.csv")
            print(f"     Training plots: results/training_logs/PPO_{self.config_name.upper()}_training_plots.png")
            
            # Clean up environments
            train_env.close()
            eval_env.close()
            
            return evaluation_results
            
        except Exception as e:
            print(f"ERROR training {self.config_name}: {e}")
            import traceback
            traceback.print_exc()
            
            train_env.close()
            eval_env.close()
            return None

def main():
    """Main comparison function"""
    
    print("PPO HYPERPARAMETER COMPARISON")
    print("=" * 60)
    print("Training ALL configurations with MAIN TRAINING parameters:")
    print("  - 200,000 timesteps (same as main)")
    print("  - Eval frequency: 10,000 (same as main)")
    print("  - Eval episodes: 10 during training (same as main)")
    print("  - 20 final evaluation episodes (same as main)")
    print("  - Vectorized environments: 4 (same as main)")
    print("  - Full CSV logging enabled (same as main)")
    print()
    print("This will provide fair comparison!")
    print()
    
    # Check current device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define configurations to test
    configs = {
        'conservative': {
            'learning_rate': 0.0001,
            'n_steps': 1024,
            'batch_size': 32,
            'n_epochs': 4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.1,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'seed': 42
        },
        'aggressive': {
            'learning_rate': 0.001,
            'n_steps': 4096,
            'batch_size': 128,
            'n_epochs': 20,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.3,
            'ent_coef': 0.001,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'seed': 456
        },
        'high_entropy': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.1,  # Higher entropy for more exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'seed': 789
        }
    }
    
    print(f"Configurations to train: {list(configs.keys())}")
    print(f"Expected training time: ~15 minutes per config")
    print()
    
    results = []
    
    for i, (config_name, config_params) in enumerate(configs.items(), 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {i}/4: {config_name.upper()}")
        print(f"{'='*80}")
        
        agent = PPOAgent(
            config_name=config_name,
            hyperparams=config_params,
            seed=config_params['seed']
        )
        
        result = agent.train_and_save()
        if result:
            results.append(result)
        else:
            print(f"Failed to train {config_name}")
    
    # Generate final comparison results
    if results:
        print(f"\n{'='*80}")
        print("PPO COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"Successfully trained: {len(results)}/4 configurations")
        print()
        
        # main training result for comparison
        main_result = {
            'config_name': 'main_training',
            'mean_reward': -158.25,
            'std_reward': 20.41292482717751,
            'hyperparameters': {
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
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
        summary_path = Path("models/ppo_tuning/comparison_summary.json")
        summary_data = {
            'comparison_type': 'Comparison - Same Training Conditions',
            'training_parameters': {
                'total_timesteps': 200000,
                'eval_freq': 10000,
                'n_eval_episodes': 10,
                'final_eval_episodes': 20,
                'vectorized_envs': 4
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
        
        print(f"\nPPO COMPARISON COMPLETED!")
        print(f"   Main training results: models/ppo/ (unchanged)")
        print(f"   Hyperparameter results: models/ppo_tuning/")
        
    else:
        print(f"\nCOMPARISON FAILED")
        print("No configurations were successfully trained.")

if __name__ == "__main__":
    main()