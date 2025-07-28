"""
Model Evaluation and Comparison Script for Rwanda Traffic Junction

This script evaluates and compares all four trained RL algorithms:
- DQN (Deep Q-Network)
- REINFORCE (Policy Gradient)
- PPO (Proximal Policy Optimization)
- Actor-Critic

It provides comprehensive performance analysis
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_junction_env import TrafficJunctionEnv
from training.dqn_training import DQNTrafficAgent
from training.reinforce_training import REINFORCEAgent
from training.ppo_training import PPOTrafficAgent
from training.actor_critic_training import ActorCriticAgent

# Set matplotlib style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EvaluationSetup:
    """
    Setup and verification for comprehensive evaluation
    """
    
    def __init__(self):
        self.required_models = {
            'DQN': 'models/dqn/best_model.zip',
            'REINFORCE_Original': 'models/reinforce/reinforce_traffic_final.pth',
            'REINFORCE_Improved': 'models/reinforce/reinforce_best.pth',
            'Actor_Critic': 'models/actor_critic/ac_best.pth',
            'PPO': 'models/ppo/best_model.zip'
        }
        
        self.alternative_paths = {
            'DQN': [
                'models/dqn/dqn_traffic_final.zip',
                'models/dqn/best_model.zip'
            ],
            'PPO': [
                'models/ppo/ppo_traffic_final.zip',
                'models/ppo/best_model.zip'
            ],
            'REINFORCE_Original': [
                'models/reinforce/reinforce_traffic_final.pth',
                'models/reinforce/reinforce_final.pth'
            ],
            'REINFORCE_Improved': [
                'models/reinforce/reinforce_best.pth',
                'models/reinforce/reinforce_improved.pth'
            ],
            'Actor_Critic': [
                'models/actor_critic/ac_best.pth',
                'models/actor_critic/ac_traffic_final.pth'
            ]
        }
    
    def check_project_structure(self):
        """Check if the project has the expected structure"""
        
        print("Checking project structure...")
        
        required_dirs = [
            'environment',
            'training',
            'models',
            'models/dqn',
            'models/reinforce',
            'models/ppo',
            'models/actor_critic'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print("MISSING DIRECTORIES:")
            for dir_path in missing_dirs:
                print(f"  - {dir_path}")
            return False
        else:
            print("Project structure verified")
            return True
    
    def find_model_files(self):
        """Find and verify model files"""
        
        print("\nChecking for trained model files...")
        
        found_models = {}
        missing_models = []
        
        for model_name, primary_path in self.required_models.items():
            model_found = False
            actual_path = None
            
            # Check primary path
            if os.path.exists(primary_path):
                model_found = True
                actual_path = primary_path
            else:
                # Check alternative paths
                for alt_path in self.alternative_paths.get(model_name, []):
                    if os.path.exists(alt_path):
                        model_found = True
                        actual_path = alt_path
                        break
            
            if model_found:
                print(f"  Found {model_name}: {actual_path}")
                found_models[model_name] = actual_path
            else:
                print(f"  Missing {model_name}: NOT FOUND")
                missing_models.append(model_name)
        
        return found_models, missing_models
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        
        print("\nChecking Python dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
            'torch', 'stable_baselines3', 'gymnasium', 'pygame'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  {package}: OK")
            except ImportError:
                print(f"  {package}: MISSING")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\nMISSING PACKAGES: {missing_packages}")
            print("Install with: pip install numpy pandas matplotlib seaborn scipy torch stable-baselines3 gymnasium pygame")
            return False
        else:
            print("All dependencies satisfied")
            return True
    
    def run_setup(self):
        """Run complete setup and verification"""
        
        print("Rwanda Traffic Junction - Evaluation Setup")
        print("=" * 50)
        
        # Check project structure
        if not self.check_project_structure():
            print("\nSETUP FAILED: Missing required directories")
            return False, {}
        
        # Find model files
        found_models, missing_models = self.find_model_files()
        
        if missing_models:
            print(f"\nWARNING: Missing {len(missing_models)} model files")
            print("Models needing training:")
            for model in missing_models:
                print(f"  - {model}")
            
            print("\nOptions:")
            print("1. Train missing models first")
            print("2. Continue with available models only")
            
            choice = input("\nContinue with available models? (y/n): ").lower().strip()
            if choice != 'y':
                print("Setup cancelled. Please train missing models first.")
                return False, {}
        
        # Check dependencies
        if not self.check_dependencies():
            print("\nSETUP FAILED: Missing required packages")
            return False, {}
        
        # Setup directories
        self.setup_evaluation_directory()
        
        print(f"\nSETUP COMPLETE")
        print(f"Found {len(found_models)} trained models")
        print("Ready to run comprehensive evaluation!")
        
        return True, found_models

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for all RL algorithms
    """
    
    def __init__(self, results_dir: str = "evaluation", model_paths: Dict = None):
        """
        Initialize comprehensive evaluator
        
        Args:
            results_dir: Directory to save evaluation results
            model_paths: Dictionary of model paths (from setup)
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Algorithm configurations - use provided paths or defaults
        if model_paths:
            self.algorithms = {
                'Random': {'type': 'random', 'model_path': None},
                **{name: {'type': self._get_model_type(name), 'model_path': path} 
                   for name, path in model_paths.items()}
            }
        else:
            self.algorithms = {
                'Random': {'type': 'random', 'model_path': None},
                'DQN': {'type': 'sb3', 'model_path': 'models/dqn/best_model.zip'},
                'REINFORCE_Original': {'type': 'pytorch', 'model_path': 'models/reinforce/reinforce_traffic_final.pth'},
                'REINFORCE_Improved': {'type': 'pytorch', 'model_path': 'models/reinforce/reinforce_best.pth'},
                'Actor_Critic': {'type': 'pytorch', 'model_path': 'models/actor_critic/ac_best.pth'},
                'PPO': {'type': 'sb3', 'model_path': 'models/ppo/best_model.zip'}
            }
        
        # Evaluation scenarios
        self.scenarios = {
            'Morning_Rush': {'time': 8.0, 'description': 'Peak morning traffic (8 AM)'},
            'Lunch_Hour': {'time': 12.5, 'description': 'Lunch time traffic (12:30 PM)'},
            'Evening_Rush': {'time': 18.0, 'description': 'Peak evening traffic (6 PM)'},
            'Night_Time': {'time': 23.0, 'description': 'Low traffic period (11 PM)'},
            'Normal_Day': {'time': 14.0, 'description': 'Regular afternoon traffic (2 PM)'}
        }
        
        # Results storage
        self.results = {}
        self.scenario_results = {}
        
        print("Comprehensive Evaluator Initialized")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Algorithms to evaluate: {len(self.algorithms)}")
        print(f"Evaluation scenarios: {len(self.scenarios)}")
    
    def _get_model_type(self, algorithm_name: str) -> str:
        """Determine model type based on algorithm name"""
        if algorithm_name in ['DQN', 'PPO']:
            return 'sb3'
        elif 'REINFORCE' in algorithm_name or algorithm_name == 'Actor_Critic':
            return 'pytorch'
        else:
            return 'unknown'
    
    def load_algorithm(self, algorithm_name: str, env: TrafficJunctionEnv):
        """
        Load a trained algorithm
        
        Args:
            algorithm_name: Name of algorithm to load
            env: Environment instance
            
        Returns:
            Loaded algorithm or None if failed
        """
        config = self.algorithms[algorithm_name]
        
        try:
            if config['type'] == 'random':
                return None  # Random policy doesn't need loading
            
            elif config['type'] == 'sb3' and algorithm_name == 'DQN':
                agent = DQNTrafficAgent()
                if os.path.exists(config['model_path']):
                    agent.load_model(config['model_path'])
                    return agent
                else:
                    print(f"Model file not found: {config['model_path']}")
                    return None
            
            elif config['type'] == 'sb3' and algorithm_name == 'PPO':
                agent = PPOTrafficAgent()
                if os.path.exists(config['model_path']):
                    agent.load_model(config['model_path'])
                    return agent
                else:
                    print(f"Model file not found: {config['model_path']}")
                    return None
            
            elif config['type'] == 'pytorch' and 'REINFORCE' in algorithm_name:
                agent = REINFORCEAgent(
                    state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n
                )
                if os.path.exists(config['model_path']):
                    agent.load_model(config['model_path'])
                    return agent
                else:
                    print(f"Model file not found: {config['model_path']}")
                    return None
            
            elif config['type'] == 'pytorch' and algorithm_name == 'Actor_Critic':
                agent = ActorCriticAgent(
                    state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n
                )
                if os.path.exists(config['model_path']):
                    agent.load_model(config['model_path'])
                    return agent
                else:
                    print(f"Model file not found: {config['model_path']}")
                    return None
            
        except Exception as e:
            print(f"Failed to load {algorithm_name}: {e}")
            return None
    
    def evaluate_algorithm(self, algorithm_name: str, agent, env: TrafficJunctionEnv, 
                          n_episodes: int = 20) -> Dict:
        """
        Evaluate a single algorithm
        
        Args:
            algorithm_name: Name of algorithm
            agent: Loaded agent (or None for random)
            env: Environment instance
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results dictionary
        """
        print(f"Evaluating {algorithm_name}...")
        
        episode_rewards = []
        episode_lengths = []
        vehicles_processed = []
        final_queue_lengths = []
        
        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get action based on algorithm type
                if algorithm_name == 'Random':
                    action = env.action_space.sample()
                elif hasattr(agent, 'predict'):  # Stable Baselines3 agents
                    action, _ = agent.predict(state, deterministic=True)
                elif hasattr(agent, 'get_action'):  # PyTorch agents
                    if algorithm_name == 'Actor_Critic':
                        action, _, _ = agent.get_action(state, training=False)  # Actor-Critic returns 3 values
                    else:
                        action, _ = agent.get_action(state, training=False)  # REINFORCE returns 2 values
                else:
                    action = env.action_space.sample()  # Fallback
                
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated or episode_length >= 500:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            vehicles_processed.append(info.get('vehicles_processed', 0))
            final_queue_lengths.append(info.get('total_vehicles_waiting', 0))
        
        # Calculate statistics
        results = {
            'algorithm': algorithm_name,
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_vehicles_processed': np.mean(vehicles_processed),
            'mean_final_queue': np.mean(final_queue_lengths),
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'vehicles': vehicles_processed,
            'queues': final_queue_lengths
        }
        
        print(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f} steps")
        
        return results
    
    def evaluate_scenario(self, algorithm_name: str, agent, scenario_name: str, 
                         scenario_config: Dict, n_episodes: int = 10) -> Dict:
        """
        Evaluate algorithm in specific traffic scenario
        
        Args:
            algorithm_name: Name of algorithm
            agent: Loaded agent
            scenario_name: Name of scenario
            scenario_config: Scenario configuration
            n_episodes: Number of episodes
            
        Returns:
            Scenario evaluation results
        """
        env = TrafficJunctionEnv(render_mode=None)
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            state, info = env.reset()
            # Set scenario-specific time
            env.current_time = scenario_config['time']
            
            episode_reward = 0
            episode_length = 0
            
            while episode_length < 200:  # Shorter episodes for scenario testing
                # Get action
                if algorithm_name == 'Random':
                    action = env.action_space.sample()
                elif hasattr(agent, 'predict'):
                    action, _ = agent.predict(state, deterministic=True)
                elif hasattr(agent, 'get_action'):
                    if algorithm_name == 'Actor_Critic':
                        action, _, _ = agent.get_action(state, training=False)  # Actor-Critic returns 3 values
                    else:
                        action, _ = agent.get_action(state, training=False)  # REINFORCE returns 2 values
                else:
                    action = env.action_space.sample()
                
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        env.close()
        
        return {
            'scenario': scenario_name,
            'algorithm': algorithm_name,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'rewards': episode_rewards
        }
    
    def run_comprehensive_evaluation(self, n_episodes: int = 20, n_scenario_episodes: int = 10):
        """
        Run complete evaluation of all algorithms
        
        Args:
            n_episodes: Episodes for main evaluation
            n_scenario_episodes: Episodes per scenario
        """
        print("Starting Comprehensive Evaluation")
        print("=" * 60)
        
        env = TrafficJunctionEnv(render_mode=None)
        
        # Main evaluation
        print("\n1. MAIN ALGORITHM EVALUATION")
        print("-" * 40)
        
        for algorithm_name in self.algorithms.keys():
            agent = self.load_algorithm(algorithm_name, env)
            results = self.evaluate_algorithm(algorithm_name, agent, env, n_episodes)
            self.results[algorithm_name] = results
        
        # Scenario evaluation
        print(f"\n2. SCENARIO-BASED EVALUATION")
        print("-" * 40)
        
        for scenario_name, scenario_config in self.scenarios.items():
            print(f"\nEvaluating scenario: {scenario_config['description']}")
            self.scenario_results[scenario_name] = {}
            
            for algorithm_name in self.algorithms.keys():
                agent = self.load_algorithm(algorithm_name, env)
                scenario_result = self.evaluate_scenario(
                    algorithm_name, agent, scenario_name, scenario_config, n_scenario_episodes
                )
                self.scenario_results[scenario_name][algorithm_name] = scenario_result
        
        env.close()
        print("\nComprehensive evaluation completed!")
    
    def create_performance_comparison_chart(self):
        """Create comprehensive performance comparison chart"""
        
        # Prepare data
        algorithms = list(self.results.keys())
        mean_rewards = [self.results[alg]['mean_reward'] for alg in algorithms]
        std_rewards = [self.results[alg]['std_reward'] for alg in algorithms]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Algorithm Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mean Reward Comparison with Error Bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        bars = ax1.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5, color=colors, alpha=0.8)
        ax1.set_title('Mean Episode Rewards with Standard Deviation', fontweight='bold')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, mean_rewards, std_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 5,
                    f'{mean_val:.1f}+/-{std_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Episode Length Comparison
        mean_lengths = [self.results[alg]['mean_length'] for alg in algorithms]
        bars2 = ax2.bar(algorithms, mean_lengths, color=colors, alpha=0.8)
        ax2.set_title('Mean Episode Length', fontweight='bold')
        ax2.set_ylabel('Episode Length (steps)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, length in zip(bars2, mean_lengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{length:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Performance Distribution (Box Plot)
        reward_data = [self.results[alg]['rewards'] for alg in algorithms]
        box_plot = ax3.boxplot(reward_data, labels=algorithms, patch_artist=True)
        ax3.set_title('Reward Distribution Analysis', fontweight='bold')
        ax3.set_ylabel('Episode Rewards')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Color the box plots
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # 4. Efficiency Metrics (Vehicles Processed)
        vehicles_processed = [self.results[alg]['mean_vehicles_processed'] for alg in algorithms]
        bars4 = ax4.bar(algorithms, vehicles_processed, color=colors, alpha=0.8)
        ax4.set_title('Traffic Processing Efficiency', fontweight='bold')
        ax4.set_ylabel('Mean Vehicles Processed')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, vehicles in zip(bars4, vehicles_processed):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{vehicles:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.results_dir, 'performance_comparison_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance comparison chart saved: {chart_path}")
    
    def create_scenario_analysis_chart(self):
        """Create scenario-based performance analysis"""
        
        # Prepare data
        scenarios = list(self.scenarios.keys())
        algorithms = list(self.algorithms.keys())
        
        # Create performance matrix
        performance_matrix = np.zeros((len(algorithms), len(scenarios)))
        
        for i, algorithm in enumerate(algorithms):
            for j, scenario in enumerate(scenarios):
                if scenario in self.scenario_results and algorithm in self.scenario_results[scenario]:
                    performance_matrix[i, j] = self.scenario_results[scenario][algorithm]['mean_reward']
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Traffic Scenario Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance Heatmap
        im = ax1.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_xticklabels([s.replace('_', ' ') for s in scenarios], rotation=45)
        ax1.set_yticklabels(algorithms)
        ax1.set_title('Performance Heatmap by Scenario', fontweight='bold')
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(scenarios)):
                text = ax1.text(j, i, f'{performance_matrix[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Mean Reward', rotation=270, labelpad=15)
        
        # 2. Scenario Comparison Line Plot
        for i, algorithm in enumerate(algorithms):
            scenario_rewards = [performance_matrix[i, j] for j in range(len(scenarios))]
            ax2.plot(scenarios, scenario_rewards, marker='o', linewidth=2, 
                    label=algorithm, alpha=0.8)
        
        ax2.set_title('Performance Across Traffic Scenarios', fontweight='bold')
        ax2.set_ylabel('Mean Reward')
        ax2.set_xlabel('Traffic Scenario')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.results_dir, 'scenario_analysis_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Scenario analysis chart saved: {chart_path}")
    
    def create_statistical_analysis_chart(self):
        """Create detailed statistical analysis charts"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Performance Analysis', fontsize=16, fontweight='bold')
        
        algorithms = list(self.results.keys())
        
        # 1. Confidence Intervals
        means = [self.results[alg]['mean_reward'] for alg in algorithms]
        stds = [self.results[alg]['std_reward'] for alg in algorithms]
        n_episodes = [self.results[alg]['n_episodes'] for alg in algorithms]
        
        # Calculate 95% confidence intervals
        confidence_intervals = []
        for i, alg in enumerate(algorithms):
            sem = stds[i] / np.sqrt(n_episodes[i])
            ci = 1.96 * sem  # 95% confidence interval
            confidence_intervals.append(ci)
        
        x_pos = np.arange(len(algorithms))
        bars = ax1.bar(x_pos, means, yerr=confidence_intervals, capsize=5, alpha=0.8)
        ax1.set_title('Performance with 95% Confidence Intervals', fontweight='bold')
        ax1.set_ylabel('Mean Reward')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(algorithms, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Improvement over Random Baseline
        random_performance = self.results['Random']['mean_reward']
        improvements = [(self.results[alg]['mean_reward'] - random_performance) / 
                       abs(random_performance) * 100 for alg in algorithms]
        
        colors = ['red' if imp < 0 else 'green' for imp in improvements]
        bars2 = ax2.bar(algorithms, improvements, color=colors, alpha=0.7)
        ax2.set_title('Improvement over Random Baseline (%)', fontweight='bold')
        ax2.set_ylabel('Improvement (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, imp in zip(bars2, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (5 if height >= 0 else -10),
                    f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        # 3. Stability Analysis (Coefficient of Variation)
        cv_values = [stds[i] / abs(means[i]) * 100 if means[i] != 0 else 0 
                    for i in range(len(algorithms))]
        
        bars3 = ax3.bar(algorithms, cv_values, alpha=0.8)
        ax3.set_title('Performance Stability (Coefficient of Variation)', fontweight='bold')
        ax3.set_ylabel('Coefficient of Variation (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add interpretation line
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='High Variability (>20%)')
        ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderate Variability (>10%)')
        ax3.legend()
        
        # 4. Performance Rankings
        # Sort by mean reward (descending)
        sorted_results = sorted([(alg, self.results[alg]['mean_reward']) for alg in algorithms], 
                               key=lambda x: x[1], reverse=True)
        
        sorted_algs = [x[0] for x in sorted_results]
        sorted_rewards = [x[1] for x in sorted_results]
        rankings = range(1, len(sorted_algs) + 1)
        
        colors_ranking = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_algs)))
        bars4 = ax4.barh(sorted_algs, sorted_rewards, color=colors_ranking, alpha=0.8)
        ax4.set_title('Final Algorithm Rankings', fontweight='bold')
        ax4.set_xlabel('Mean Reward')
        ax4.grid(True, alpha=0.3)
        
        # Add ranking numbers
        for i, (bar, reward) in enumerate(zip(bars4, sorted_rewards)):
            width = bar.get_width()
            ax4.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2,
                    f'#{i+1}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.results_dir, 'statistical_analysis_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Statistical analysis chart saved: {chart_path}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        
        report_path = os.path.join(self.results_dir, 'comprehensive_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE ALGORITHM EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall Performance Summary
            f.write("1. OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Sort algorithms by performance
            sorted_results = sorted(self.results.items(), 
                                  key=lambda x: x[1]['mean_reward'], reverse=True)
            
            f.write(f"{'Rank':<4} {'Algorithm':<20} {'Mean Reward':<15} {'Std Dev':<12} {'Improvement':<12}\n")
            f.write("-" * 75 + "\n")
            
            random_baseline = self.results['Random']['mean_reward']
            
            for i, (alg_name, results) in enumerate(sorted_results):
                improvement = ((results['mean_reward'] - random_baseline) / 
                             abs(random_baseline) * 100)
                f.write(f"{i+1:<4} {alg_name:<20} {results['mean_reward']:<15.2f} "
                       f"{results['std_reward']:<12.2f} {improvement:<12.1f}%\n")
            
            # Statistical Analysis
            f.write(f"\n2. STATISTICAL ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Best performer
            best_alg = sorted_results[0][0]
            best_results = sorted_results[0][1]
            f.write(f"Best Performing Algorithm: {best_alg}\n")
            f.write(f"  Mean Reward: {best_results['mean_reward']:.2f} +/- {best_results['std_reward']:.2f}\n")
            f.write(f"  Mean Episode Length: {best_results['mean_length']:.1f} steps\n")
            f.write(f"  Vehicles Processed: {best_results['mean_vehicles_processed']:.1f}\n\n")
            
            # Stability analysis
            f.write("Algorithm Stability Rankings (by Coefficient of Variation):\n")
            stability_results = []
            for alg_name, results in self.results.items():
                cv = results['std_reward'] / abs(results['mean_reward']) * 100 if results['mean_reward'] != 0 else float('inf')
                stability_results.append((alg_name, cv))
            
            stability_results.sort(key=lambda x: x[1])
            
            for i, (alg_name, cv) in enumerate(stability_results):
                stability_level = "Excellent" if cv < 10 else "Good" if cv < 20 else "Moderate" if cv < 30 else "Poor"
                f.write(f"  {i+1}. {alg_name}: {cv:.1f}% ({stability_level})\n")
            
            # Scenario Performance
            f.write(f"\n3. SCENARIO PERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            for scenario_name, scenario_results in self.scenario_results.items():
                f.write(f"\n{scenario_name.replace('_', ' ')} Scenario:\n")
                scenario_sorted = sorted(scenario_results.items(), 
                                       key=lambda x: x[1]['mean_reward'], reverse=True)
                
                for i, (alg_name, result) in enumerate(scenario_sorted):
                    f.write(f"  {i+1}. {alg_name}: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}\n")
            
            # Deployment Recommendations
            f.write(f"\n4. DEPLOYMENT RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            # Top 3 algorithms
            top_3 = sorted_results[:3]
            
            f.write("Recommended Deployment Portfolio:\n\n")
            
            for i, (alg_name, results) in enumerate(top_3):
                improvement = ((results['mean_reward'] - random_baseline) / 
                             abs(random_baseline) * 100)
                
                if i == 0:
                    f.write(f"PRIMARY DEPLOYMENT: {alg_name}\n")
                elif i == 1:
                    f.write(f"SECONDARY OPTION: {alg_name}\n")
                else:
                    f.write(f"BACKUP SYSTEM: {alg_name}\n")
                
                f.write(f"  Performance: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}\n")
                f.write(f"  Improvement: {improvement:.1f}% over baseline\n")
                f.write(f"  Reliability: {results['std_reward']:.2f} standard deviation\n")
                f.write(f"  Efficiency: {results['mean_length']:.1f} average steps per episode\n\n")
            
            # Technical Summary
            f.write(f"5. TECHNICAL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Algorithms Evaluated: {len(self.results)}\n")
            f.write(f"Episodes per Algorithm: {self.results[best_alg]['n_episodes']}\n")
            f.write(f"Traffic Scenarios Tested: {len(self.scenarios)}\n")
            f.write(f"Best Overall Performance: {best_results['mean_reward']:.2f}\n")
            f.write(f"Baseline (Random) Performance: {random_baseline:.2f}\n")
            best_improvement = ((best_results['mean_reward'] - random_baseline) / 
                              abs(random_baseline) * 100)
            f.write(f"Maximum Improvement Achieved: {best_improvement:.1f}%\n")
        
        print(f"Comprehensive evaluation report saved: {report_path}")
    
    def save_results_to_csv(self):
        """Save detailed results to CSV files"""
        
        # Main results CSV
        main_results_data = []
        for alg_name, results in self.results.items():
            main_results_data.append({
                'Algorithm': alg_name,
                'Mean_Reward': results['mean_reward'],
                'Std_Reward': results['std_reward'],
                'Mean_Length': results['mean_length'], 
                'Std_Length': results['std_length'],
                'Mean_Vehicles_Processed': results['mean_vehicles_processed'],
                'Mean_Final_Queue': results['mean_final_queue'],
                'N_Episodes': results['n_episodes']
            })
        
        main_df = pd.DataFrame(main_results_data)
        main_csv_path = os.path.join(self.results_dir, 'main_evaluation_results.csv')
        main_df.to_csv(main_csv_path, index=False)
        
        # Scenario results CSV
        scenario_results_data = []
        for scenario_name, scenario_data in self.scenario_results.items():
            for alg_name, result in scenario_data.items():
                scenario_results_data.append({
                    'Scenario': scenario_name,
                    'Algorithm': alg_name,
                    'Mean_Reward': result['mean_reward'],
                    'Std_Reward': result['std_reward']
                })
        
        scenario_df = pd.DataFrame(scenario_results_data)
        scenario_csv_path = os.path.join(self.results_dir, 'scenario_evaluation_results.csv')
        scenario_df.to_csv(scenario_csv_path, index=False)
        
        print(f"Main results saved: {main_csv_path}")
        print(f"Scenario results saved: {scenario_csv_path}")
    
    def save_results_to_json(self):
        """Save complete results to JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg_name, results in self.results.items():
            json_results[alg_name] = {
                'mean_reward': float(results['mean_reward']),
                'std_reward': float(results['std_reward']),
                'mean_length': float(results['mean_length']),
                'std_length': float(results['std_length']),
                'mean_vehicles_processed': float(results['mean_vehicles_processed']),
                'mean_final_queue': float(results['mean_final_queue']),
                'n_episodes': int(results['n_episodes']),
                'rewards': [float(r) for r in results['rewards']],
                'lengths': [int(l) for l in results['lengths']]
            }
        
        # Convert scenario results
        json_scenario_results = {}
        for scenario_name, scenario_data in self.scenario_results.items():
            json_scenario_results[scenario_name] = {}
            for alg_name, result in scenario_data.items():
                json_scenario_results[scenario_name][alg_name] = {
                    'mean_reward': float(result['mean_reward']),
                    'std_reward': float(result['std_reward']),
                    'rewards': [float(r) for r in result['rewards']]
                }
        
        # Complete results dictionary
        complete_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'main_results': json_results,
            'scenario_results': json_scenario_results,
            'scenarios': self.scenarios
        }
        
        json_path = os.path.join(self.results_dir, 'complete_evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"Complete results saved: {json_path}")
    
    def run_complete_analysis(self, n_episodes: int = 20, n_scenario_episodes: int = 10):
        """
        Run the complete comprehensive evaluation and analysis
        
        Args:
            n_episodes: Episodes for main evaluation
            n_scenario_episodes: Episodes per scenario
        """
        print("STARTING COMPREHENSIVE EVALUATION AND ANALYSIS")
        print("=" * 70)
        
        # Run evaluation
        self.run_comprehensive_evaluation(n_episodes, n_scenario_episodes)
        
        # Generate all visualizations
        print("\nGenerating visualizations...")
        self.create_performance_comparison_chart()
        self.create_scenario_analysis_chart() 
        self.create_statistical_analysis_chart()
        
        # Generate reports and save data
        print("\nGenerating reports and saving data...")
        self.generate_performance_report()
        self.save_results_to_csv()
        self.save_results_to_json()
        
        print(f"\nCOMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.results_dir}")
        print(f"Generated files:")
        print(f"  - performance_comparison_chart.png")
        print(f"  - scenario_analysis_chart.png") 
        print(f"  - statistical_analysis_chart.png")
        print(f"  - comprehensive_evaluation_report.txt")
        print(f"  - main_evaluation_results.csv")
        print(f"  - scenario_evaluation_results.csv")
        print(f"  - complete_evaluation_results.json")


def main():
    """
    Main function to run comprehensive evaluation
    """
    print("Rwanda Traffic Junction - Comprehensive Algorithm Evaluation")
    print("Mission: Final analysis of all trained RL algorithms")
    print()
    
    # Configuration
    config = {
        'n_episodes': 25,           # Episodes per algorithm for main evaluation
        'n_scenario_episodes': 15,  # Episodes per scenario per algorithm
        'results_dir': 'evaluation' # Output directory
    }
    
    print("Evaluation Configuration:")
    print(f"  Main evaluation episodes per algorithm: {config['n_episodes']}")
    print(f"  Scenario episodes per algorithm: {config['n_scenario_episodes']}")
    print(f"  Results directory: {config['results_dir']}")
    print()
    
    # Run setup first
    print("PHASE 1: SETUP AND VERIFICATION")
    print("=" * 40)
    
    setup = EvaluationSetup()
    setup_success, found_models = setup.run_setup()
    
    if not setup_success:
        print("Setup failed. Cannot proceed with evaluation.")
        return 1
    
    # Ask to proceed
    print("\nPHASE 2: COMPREHENSIVE EVALUATION")
    print("=" * 40)
    
    response = input("Proceed with comprehensive evaluation? (y/n): ").lower().strip()
    if response != 'y':
        print("Evaluation cancelled.")
        return 0
    
    try:
        # Create evaluator with verified model paths
        evaluator = ComprehensiveEvaluator(
            results_dir=config['results_dir'],
            model_paths=found_models
        )
        
        # Run complete analysis
        evaluator.run_complete_analysis(
            n_episodes=config['n_episodes'],
            n_scenario_episodes=config['n_scenario_episodes']
        )
        
        print("\nComprehensive evaluation completed!")
        print(f"Results saved to: {config['results_dir']}/")
        
        # List generated files
        result_files = [
            'performance_comparison_chart.png',
            'scenario_analysis_chart.png', 
            'statistical_analysis_chart.png',
            'comprehensive_evaluation_report.txt',
            'main_evaluation_results.csv',
            'scenario_evaluation_results.csv',
            'complete_evaluation_results.json'
        ]
        
        print("\nGenerated Files:")
        for file in result_files:
            file_path = os.path.join(config['results_dir'], file)
            if os.path.exists(file_path):
                print(f" {file}")
            else:
                print(f"  MISSING: {file}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Evaluation failed with error: {e}")
        print("Please check that all model files exist and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)