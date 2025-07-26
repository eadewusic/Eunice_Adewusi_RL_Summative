"""
Random Action Demo for Rwanda Traffic Junction Environment

This script demonstrates the traffic junction environment with an agent
taking random actions. It creates both a live visualization and exports
a GIF for the assignment submission.

Requirements for assignment:
- Static file showing agent taking random actions
- GIF of agent in simulated environment
- No model training involved (just demonstration)
"""

import sys
import os
import random
import imageio
import numpy as np
from typing import List
import pygame

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_junction_env import TrafficJunctionEnv
from environment.traffic_rendering import TrafficVisualizer

class RandomAgent:
    """
    A simple agent that takes random actions for demonstration purposes
    """
    
    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size
        self.action_names = [
            "Extend NS Green",     # 0
            "Extend EW Green",     # 1  
            "Switch to NS",        # 2
            "Switch to EW",        # 3
            "Emergency Priority",  # 4
            "All Red",            # 5
            "Reset Timer",        # 6
            "Short Cycle",        # 7
            "Long Cycle"          # 8
        ]
    
    def get_action(self, observation: np.ndarray) -> int:
        """
        Get a random action
        
        Args:
            observation: Current environment state (unused for random agent)
            
        Returns:
            Random action index
        """
        return random.randint(0, self.action_space_size - 1)
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        if 0 <= action < len(self.action_names):
            return self.action_names[action]
        return f"Unknown Action {action}"

def capture_frame_from_surface(surface) -> np.ndarray:
    """
    Capture frame from pygame surface for GIF
    
    Args:
        surface: Pygame surface
        
    Returns:
        RGB array suitable for GIF
    """
    # Convert pygame surface to numpy array
    frame_array = pygame.surfarray.array3d(surface)
    # Transpose to get correct orientation (pygame uses (width, height, channels))
    frame_array = np.transpose(frame_array, (1, 0, 2))
    return frame_array

def run_random_demo(episodes: int = 3, max_steps_per_episode: int = 200, 
                   export_gif: bool = True, gif_filename: str = "traffic_random_demo.gif", 
                   show_window: bool = True):
    """
    Run the random action demonstration
    
    Args:
        episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        export_gif: Whether to export GIF
        gif_filename: Name of the GIF file
        show_window: Whether to show the pygame window
    """
    
    print("Rwanda Traffic Junction - Random Action Demo")
    print("=" * 60)
    print(f"Running {episodes} episodes with random actions...")
    print("This demonstrates the environment without any trained model.")
    print()
    
    # Create environment - always use human mode if showing window
    render_mode = "human" if show_window else None
    env = TrafficJunctionEnv(render_mode=render_mode)
    agent = RandomAgent(env.action_space.n)
    
    # Create visualizer
    visualizer = TrafficVisualizer(env)
    
    # Storage for GIF frames
    gif_frames: List[np.ndarray] = []
    
    # Create videos directory if exporting GIF
    if export_gif:
        os.makedirs("videos", exist_ok=True)
        gif_path = os.path.join("videos", gif_filename)
        print(f"📹 GIF will be saved to: {gif_path}")
    
    # Statistics tracking
    total_reward = 0
    total_steps = 0
    total_vehicles_processed = 0
    
    try:
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            
            # Reset environment
            observation, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            print(f"Starting time: {info['time_of_day']}")
            print(f"Initial state: {info['hidden_state']}")
            
            for step in range(max_steps_per_episode):
                # Agent takes random action
                action = agent.get_action(observation)
                action_name = agent.get_action_name(action)
                
                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Update statistics
                episode_reward += reward
                episode_steps += 1
                
                # Render environment - Draw all components
                if show_window:
                    # Clear screen and draw everything
                    visualizer.screen.fill(visualizer.colors['background'])
                    visualizer._draw_roads()
                    visualizer._draw_intersection()
                    visualizer._draw_traffic_lights()
                    visualizer._draw_vehicles()
                    visualizer._draw_queue_indicators()
                    visualizer._draw_performance_panel(action)
                    visualizer._draw_info_panel()
                    
                    # Update display
                    pygame.display.flip()
                    visualizer.clock.tick(8)  # 8 FPS for good viewing speed
                    
                    # Capture frame for GIF (every 3rd frame to reduce file size but keep smoothness)
                    if export_gif and step % 3 == 0:
                        frame = capture_frame_from_surface(visualizer.screen)
                        gif_frames.append(frame)
                
                # Print step information (every 20 steps to avoid spam)
                if step % 20 == 0:
                    print(f"  Step {step:3d}: {action_name:15s} | "
                          f"Reward: {reward:+6.2f} | "
                          f"Queues: {info['total_vehicles_waiting']:2d} | "
                          f"Light: {info['current_light']:12s} | "
                          f"State: {info['hidden_state']}")
                
                # Handle pygame events
                if show_window and not visualizer.handle_events():
                    print("User closed window. Exiting...")
                    return
                
                # Check if episode ended
                if terminated or truncated:
                    print(f"  Episode ended at step {step}")
                    if terminated:
                        print("  Reason: Environment terminated (gridlock)")
                    else:
                        print("  Reason: Maximum steps reached")
                    break
            
            # Episode summary
            total_reward += episode_reward
            total_steps += episode_steps
            total_vehicles_processed += info['vehicles_processed']
            
            print(f"Episode {episode + 1} Summary:")
            print(f"  Steps: {episode_steps}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Vehicles Processed: {info['vehicles_processed']}")
            print(f"  Final Queue Length: {info['total_vehicles_waiting']}")
            print(f"  Final Time: {info['time_of_day']}")
            print(f"  Final State: {info['hidden_state']}")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

    finally:
        # Overall statistics
        print("\n" + "=" * 60)
        print("OVERALL DEMONSTRATION STATISTICS")
        print("=" * 60)
        print(f"Total Episodes: {episodes}")
        print(f"Total Steps: {total_steps}")
        print(f"Average Steps per Episode: {total_steps / episodes:.1f}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Reward per Episode: {total_reward / episodes:.2f}")
        print(f"Total Vehicles Processed: {total_vehicles_processed}")
        print(f"Average Vehicles per Episode: {total_vehicles_processed / episodes:.1f}")
        
        # Export GIF if requested
        if export_gif and gif_frames:
            print(f"\nExporting GIF with {len(gif_frames)} frames...")
            try:
                # Ensure frames are valid and have consistent dimensions
                valid_frames = []
                for frame in gif_frames:
                    if frame.size > 0 and len(frame.shape) == 3:
                        valid_frames.append(frame)
                
                if valid_frames:
                    # Save to videos folder
                    gif_path = os.path.join("videos", gif_filename)
                    imageio.mimsave(gif_path, valid_frames, fps=8, loop=0, duration=0.125)
                    
                    print(f"GIF saved as: {gif_path}")
                    file_size = os.path.getsize(gif_path) / 1024 / 1024
                    print(f"   File size: ~{file_size:.1f} MB")
                    print(f"   Duration: ~{len(valid_frames) / 8:.1f} seconds")
                    print(f"   Frames: {len(valid_frames)}")
                else:
                    print("No valid frames captured for GIF")
            except Exception as e:
                print(f"Error saving GIF: {e}")
                import traceback
                traceback.print_exc()
        elif export_gif:
            print("No frames captured - ensure show_window=True for GIF export")
        
        # Clean up
        visualizer.close()
        env.close()

        print("\nRandom action demonstration completed!")

def analyze_random_performance():
    """
    Analyze the performance of random actions to establish baseline
    """
    print("\nRANDOM AGENT BASELINE ANALYSIS")
    print("=" * 50)
    
    env = TrafficJunctionEnv(render_mode=None)  # No rendering for analysis
    agent = RandomAgent(env.action_space.n)
    
    num_episodes = 10
    episode_rewards = []
    episode_vehicles_processed = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(500):  # Longer episodes for analysis
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_vehicles_processed.append(info['vehicles_processed'])
        episode_lengths.append(steps)
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_vehicles = np.mean(episode_vehicles_processed)
    avg_length = np.mean(episode_lengths)
    
    print(f"Episodes analyzed: {num_episodes}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average vehicles processed: {avg_vehicles:.1f}")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    
    env.close()
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_vehicles': avg_vehicles,
        'avg_length': avg_length
    }

def create_random_actions_gif():
    """
    Create a high-quality GIF specifically for assignment submission
    """
    print("Creating Assignment Submission GIF")
    print("=" * 50)
    
    # Create a focused demo for assignment
    run_random_demo(
        episodes=2,  # Shorter for assignment
        max_steps_per_episode=120,
        export_gif=True,
        gif_filename="traffic_random_demo.gif",
        show_window=True
    )
    
    print("Random Traffic Actions GIF created!")

if __name__ == "__main__":
    print("Rwanda Traffic Flow Optimization - Random Demo")
    print("Environment: Traffic light control to replace road wardens")
    print()
    
    # Option 1: Create assignment-ready GIF
    create_random_actions_gif()
    
    # Option 2: Full analysis
    print("\n" + "="*60)
    baseline_stats = analyze_random_performance()
    
    print(f"\nBaseline established for RL agent comparison:")
    print(f"   Random agent achieves {baseline_stats['avg_reward']:.2f} average reward")
    print(f"   This will be used to measure improvement from trained agents")