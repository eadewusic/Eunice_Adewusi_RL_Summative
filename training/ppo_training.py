"""
PPO Training Script for Rwanda Traffic Junction Environment

This script implements Proximal Policy Optimization (PPO) using Stable Baselines3
for the traffic light optimization task.

PPO is an advanced policy gradient method that uses a clipped objective function
to prevent large policy updates, making it more stable than vanilla policy gradients.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Environment imports
from environment.traffic_junction_env import TrafficJunctionEnv

class PPOTrafficAgent:
    """
    PPO agent for traffic light optimization using Stable Baselines3
    """
    
    def __init__(self, 
                 learning_rate: float = 0.0003,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 verbose: int = 1):
        """
        Initialize PPO agent with hyperparameters
        
        Args:
            learning_rate: Learning rate for the optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs for each policy update
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Maximum value for gradient clipping
            verbose: Verbosity level
        """
        
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm
        }
        
        self.verbose = verbose
        self.model = None
        self.training_history = []
        
    def create_model(self, env) -> PPO:
        """
        Create PPO model with specified hyperparameters
        
        Args:
            env: Training environment
            
        Returns:
            PPO model instance
        """
        
        # Neural network architecture for PPO
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
            verbose=self.verbose,
            tensorboard_log="./tensorboard_logs/ppo/",
            device='auto'  # Use GPU if available
        )
        
        return self.model
    
    def train(self, 
              total_timesteps: int = 200000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              eval_env=None,
              save_path: str = "models/ppo/") -> Dict:
        """
        Train the PPO agent
        
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
        
        print(f"Starting PPO Training for {total_timesteps:,} timesteps")
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
        
        # Optional: Stop training when reward threshold is reached
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=200,  # Adjust based on environment
            verbose=1
        )
        
        # Train the model
        start_time = datetime.now()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, stop_callback],
                progress_bar=True
            )
            
            training_time = datetime.now() - start_time
            print(f"\nTraining completed in {training_time}")
            
            # Save final model
            final_model_path = os.path.join(save_path, "ppo_traffic_final.zip")
            self.model.save(final_model_path)
            print(f"Final model saved: {final_model_path}")
            
            # Save hyperparameters
            config_path = os.path.join(save_path, "ppo_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.hyperparameters, f, indent=2)
            
            return {
                'training_time': str(training_time),
                'total_timesteps': total_timesteps,
                'final_model_path': final_model_path,
                'config_path': config_path
            }
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            # Still save the current model
            interrupted_path = os.path.join(save_path, "ppo_traffic_interrupted.zip")
            self.model.save(interrupted_path)
            print(f"Interrupted model saved: {interrupted_path}")
            return None
    
    def evaluate(self, env, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the trained PPO agent
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes for evaluation
            
        Returns:
            Mean reward and standard deviation
        """
        
        if self.model is None:
            raise ValueError("No trained model available")
        
        print(f"Evaluating PPO agent over {n_episodes} episodes...")
        
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
        """Load a pre-trained PPO model"""
        self.model = PPO.load(model_path)
        print(f"PPO model loaded from: {model_path}")
    
    def predict(self, observation, deterministic: bool = True):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("No trained model available")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def get_action_probabilities(self, observation):
        """Get action probabilities for analysis"""
        if self.model is None:
            raise ValueError("No trained model available")
        
        # Get action probabilities from policy
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs
        
        return action_probs.cpu().numpy()

def create_vectorized_env(n_envs: int = 4, env_kwargs: dict = None):
    """
    Create vectorized environment for parallel training
    
    Args:
        n_envs: Number of parallel environments
        env_kwargs: Environment keyword arguments
        
    Returns:
        Vectorized environment
    """
    if env_kwargs is None:
        env_kwargs = {'render_mode': None}
    
    def make_env():
        def _init():
            env = TrafficJunctionEnv(**env_kwargs)
            env = Monitor(env)
            return env
        return _init
    
    # Create parallel environments
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    return env

def hyperparameter_tuning_experiment():
    """
    Run hyperparameter tuning experiment for PPO
    """
    print("PPO Hyperparameter Tuning Experiment")
    print("=" * 50)
    
    # Create environment for tuning
    env = create_vectorized_env(n_envs=2)  # Smaller for tuning
    
    # Hyperparameter combinations to test
    hyperparameter_configs = [
        # Configuration 1: Conservative
        {
            'name': 'conservative',
            'learning_rate': 0.0001,
            'n_steps': 1024,
            'batch_size': 32,
            'n_epochs': 4,
            'clip_range': 0.1,
            'ent_coef': 0.01
        },
        # Configuration 2: Standard
        {
            'name': 'standard',
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'clip_range': 0.2,
            'ent_coef': 0.0
        },
        # Configuration 3: Aggressive
        {
            'name': 'aggressive',
            'learning_rate': 0.001,
            'n_steps': 4096,
            'batch_size': 128,
            'n_epochs': 20,
            'clip_range': 0.3,
            'ent_coef': 0.001
        },
        # Configuration 4: High Entropy
        {
            'name': 'high_entropy',
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'clip_range': 0.2,
            'ent_coef': 0.1  # Higher entropy for more exploration
        }
    ]
    
    results = []
    
    for config in hyperparameter_configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Create agent with specific hyperparameters
        agent = PPOTrafficAgent(
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            verbose=0  # Reduced verbosity for tuning
        )
        
        # Create model
        agent.create_model(env)
        
        # Train for shorter duration for tuning
        training_stats = agent.train(
            total_timesteps=50000,
            eval_freq=10000,
            save_path=f"models/ppo_tuning/{config['name']}/"
        )
        
        if training_stats:
            # Evaluate performance
            eval_env = TrafficJunctionEnv(render_mode=None)
            eval_env = Monitor(eval_env)
            mean_reward, std_reward = agent.evaluate(eval_env, n_episodes=5)
            eval_env.close()
            
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

def advanced_ppo_analysis(agent: PPOTrafficAgent, env):
    """
    Perform advanced analysis of PPO agent behavior
    
    Args:
        agent: Trained PPO agent
        env: Environment for analysis
    """
    print("Advanced PPO Agent Analysis")
    print("=" * 40)
    
    # Test different traffic scenarios
    scenarios = [
        {'name': 'Rush Hour', 'time': 8.0},    # 8 AM
        {'name': 'Lunch Time', 'time': 12.5},  # 12:30 PM
        {'name': 'Evening Rush', 'time': 18.0}, # 6 PM
        {'name': 'Night Time', 'time': 23.0}   # 11 PM
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        print(f"\nAnalyzing {scenario['name']} scenario...")
        
        # Reset environment to specific time
        obs, info = env.reset()
        env.current_time = scenario['time']
        
        episode_actions = []
        action_probs_history = []
        rewards = []
        
        for step in range(100):  # Analyze 100 steps
            # Get action and probabilities
            action, _ = agent.predict(obs, deterministic=False)
            action_probs = agent.get_action_probabilities(obs)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_actions.append(action)
            action_probs_history.append(action_probs)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Analyze behavior
        action_distribution = np.bincount(episode_actions, minlength=9)
        avg_reward = np.mean(rewards)
        
        scenario_results.append({
            'scenario': scenario['name'],
            'action_distribution': action_distribution.tolist(),
            'avg_reward': avg_reward,
            'total_steps': len(episode_actions)
        })
        
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Most Frequent Action: {np.argmax(action_distribution)}")
        print(f"  Steps Completed: {len(episode_actions)}")
    
    return scenario_results

def main_training_experiment():
    """
    Main PPO training experiment with best hyperparameters
    """
    print("PPO Main Training Experiment")
    print("=" * 50)
    
    # Create vectorized training environment for parallel sampling
    train_env = create_vectorized_env(n_envs=4)
    
    # Create evaluation environment
    eval_env = TrafficJunctionEnv(render_mode=None)
    eval_env = Monitor(eval_env)
    
    # Create PPO agent with optimized hyperparameters
    agent = PPOTrafficAgent(
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Create model
    agent.create_model(train_env)
    
    # Train the agent
    training_stats = agent.train(
        total_timesteps=200000,
        eval_freq=10000,
        n_eval_episodes=10,
        eval_env=eval_env,
        save_path="models/ppo/"
    )
    
    if training_stats:
        # Final evaluation
        print("\nFINAL EVALUATION")
        print("=" * 30)
        
        mean_reward, std_reward = agent.evaluate(eval_env, n_episodes=20)
        
        # Advanced analysis
        scenario_analysis = advanced_ppo_analysis(agent, eval_env)
        
        # Save evaluation results
        evaluation_results = {
            'algorithm': 'PPO',
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'hyperparameters': agent.hyperparameters,
            'training_stats': training_stats,
            'scenario_analysis': scenario_analysis
        }
        
        results_path = "models/ppo/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved: {results_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return agent

if __name__ == "__main__":
    import torch
    
    print("Rwanda Traffic Flow Optimization - PPO Training")
    print("Assignment: Mission-Based Reinforcement Learning")
    print("Algorithm: Proximal Policy Optimization (Advanced Policy Method)")
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
    
    print("\nPPO training pipeline completed!")
    print("Key advantages of PPO:")
    print("- Stable training through clipped objective function")
    print("- Sample efficient due to multiple epochs per batch")
    print("- Actor-critic architecture combines value and policy learning")
    print("- Generalized Advantage Estimation (GAE) for better variance-bias trade-off")