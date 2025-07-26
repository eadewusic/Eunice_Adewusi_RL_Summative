"""
Rwanda Traffic Flow Optimization - Main Entry Point

This is the main script for the Mission-Based Reinforcement Learning assignment.
It provides a unified interface to run different experiments and demonstrations.

Environment: Traffic Junction Optimization for Rwanda
"""

import argparse
import sys
import os
from typing import Dict, Any

def run_dqn_training():
    """Run DQN training experiment"""
    print("Running DQN Training...")
    try:
        from training.dqn_training import main_training_experiment
        return main_training_experiment()
    except Exception as e:
        print(f"DQN training failed: {e}")
        return None

def run_reinforce_training():
    """Run REINFORCE training experiment"""
    print("Running REINFORCE Training...")
    try:
        from training.reinforce_training import main_training_experiment as reinforce_training
        return reinforce_training()
    except Exception as e:
        print(f"REINFORCE training failed: {e}")
        return None

def run_ppo_training():
    """Run PPO training experiment"""
    print("Running PPO Training...")
    try:
        from training.ppo_training import main_training_experiment as ppo_training
        return ppo_training()
    except Exception as e:
        print(f"PPO training failed: {e}")
        return None

def run_actor_critic_training():
    """Run Actor-Critic training experiment"""
    print("Running Actor-Critic Training...")
    try:
        from training.actor_critic_training import main_training_experiment as ac_training
        return ac_training()
    except Exception as e:
        print(f"Actor-Critic training failed: {e}")
        return None

def run_all_training():
    """Run training for all RL algorithms"""
    print("Running All RL Algorithm Training...")
    
    results = {}
    
    # DQN Training
    print("\n" + "="*60)
    print("1/4: Training DQN (Deep Q-Network)")
    print("="*60)
    try:
        from training.dqn_training import main_training_experiment
        dqn_agent = main_training_experiment()
        results['dqn'] = dqn_agent
        print("DQN training completed")
    except Exception as e:
        print(f"DQN training failed: {e}")
        results['dqn'] = None
    
    # REINFORCE Training
    print("\n" + "="*60)
    print("2/4: Training REINFORCE (Policy Gradient)")
    print("="*60)
    try:
        from training.reinforce_training import main_training_experiment as reinforce_training
        reinforce_agent = reinforce_training()
        results['reinforce'] = reinforce_agent
        print("REINFORCE training completed")
    except Exception as e:
        print(f"REINFORCE training failed: {e}")
        results['reinforce'] = None
    
    # PPO Training
    print("\n" + "="*60)
    print("3/4: Training PPO (Proximal Policy Optimization)")
    print("="*60)
    try:
        from training.ppo_training import main_training_experiment as ppo_training
        ppo_agent = ppo_training()
        results['ppo'] = ppo_agent
        print("PPO training completed")
    except Exception as e:
        print(f"PPO training failed: {e}")
        results['ppo'] = None
    
    # Actor-Critic Training
    print("\n" + "="*60)
    print("4/4: Training Actor-Critic")
    print("="*60)
    try:
        from training.actor_critic_training import main_training_experiment as ac_training
        ac_agent = ac_training()
        results['actor_critic'] = ac_agent
        print("Actor-Critic training completed")
    except Exception as e:
        print(f"Actor-Critic training failed: {e}")
        results['actor_critic'] = None
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    successful = sum(1 for agent in results.values() if agent is not None)
    print(f"Successfully trained: {successful}/4 algorithms")
    
    for algorithm, agent in results.items():
        status = "SUCCESS" if agent is not None else "FAILED"
        print(f"   {algorithm.upper()}: {status}")
    
    return results

def run_evaluation():
    """Run evaluation of all trained models"""
    print("Running Model Evaluation and Comparison...")
    
    try:
        from evaluation.compare_algorithms import main as run_comprehensive_evaluation
        run_comprehensive_evaluation()
        print("Comprehensive evaluation completed")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Please ensure models are trained and evaluation script is available")

def print_project_info():
    """Print project information and structure"""
    print("Rwanda Traffic Flow Optimization Project")
    print("=" * 60)
    print("Mission: Replace manual road wardens with intelligent RL agents")
    print("Problem: Traffic congestion in major African cities")
    print("Solution: Predictive traffic light optimization")
    print()
    print("Environment Details:")
    print("  - 4-way traffic intersection simulation")
    print("  - Hidden states: Free Flow, Building Congestion, Peak Hour Gridlock, etc.")
    print("  - Actions: Traffic light control (9 discrete actions)")
    print("  - Rewards: Based on traffic flow efficiency and waiting times")
    print("  - Vehicle types: Cars, buses, motorcycles, trucks, emergency vehicles")
    print()
    print("RL Algorithms Implemented:")
    print("  1. DQN (Deep Q-Network) - Value-based method")
    print("  2. REINFORCE - Policy gradient method")
    print("  3. PPO (Proximal Policy Optimization) - Advanced policy method")
    print("  4. Actor-Critic - Hybrid value-policy method")
    print()
    print("Project Structure:")
    print("  environment/     - Custom Gymnasium environment and visualization")
    print("  training/        - RL algorithm training scripts")
    print("  models/          - Saved trained models")
    print("  visualization/   - Visualization scripts")
    print("  main.py          - This entry point script")
    print()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Rwanda Traffic Flow Optimization - RL Assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --info                 # Show project information
  python main.py --train-dqn            # Train DQN agent only
  python main.py --train-reinforce      # Train REINFORCE agent only
  python main.py --train-ppo            # Train PPO agent only
  python main.py --train-ac             # Train Actor-Critic agent only
  python main.py --train-all            # Train all RL algorithms
  python main.py --evaluate             # Evaluate trained models
        """
    )
    
    parser.add_argument('--info', action='store_true',
                       help='Show project information and structure')
    parser.add_argument('--train-dqn', action='store_true',
                       help='Train DQN agent only')
    parser.add_argument('--train-reinforce', action='store_true',
                       help='Train REINFORCE agent only')
    parser.add_argument('--train-ppo', action='store_true',
                       help='Train PPO agent only')
    parser.add_argument('--train-ac', action='store_true',
                       help='Train Actor-Critic agent only')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all RL algorithms (DQN, REINFORCE, PPO, Actor-Critic)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate and compare all trained models')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        print_project_info()
        parser.print_help()
        return
    
    # Execute requested operations
    try:
        if args.info:
            print_project_info()
        
        if args.train_dqn:
            run_dqn_training()
        
        if args.train_reinforce:
            run_reinforce_training()
        
        if args.train_ppo:
            run_ppo_training()
            
        if args.train_ac:
            run_actor_critic_training()
        
        if args.train_all:
            run_all_training()
        
        if args.evaluate:
            run_evaluation()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

def setup_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'models/dqn',
        'models/reinforce', 
        'models/ppo',
        'models/actor_critic',
        'models/dqn_tuning',
        'models/reinforce_tuning',
        'models/ppo_tuning',
        'tensorboard_logs/dqn',
        'tensorboard_logs/reinforce',
        'tensorboard_logs/ppo',
        'tensorboard_logs/actor_critic',
        'results',
        'results/training_logs',  # For CSV training metrics
        'videos',
        'evaluation'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories created successfully")

if __name__ == "__main__":
    print("Rwanda Traffic Junction - RL Optimization System")
    print("Objective: Replace road wardens with intelligent agents")
    print()
    
    # Setup project structure
    setup_project_structure()
    
    # Run main function
    main()