import cv2
import numpy as np
import pygame
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add environment path
sys.path.append(str(Path(__file__).parent.parent))

from environment.traffic_junction_env import TrafficJunctionEnv
from environment.traffic_rendering import TrafficVisualizer
from stable_baselines3 import PPO

class TrafficVideoDemo:
    """
    2D video demonstration using existing Pygame visualization
    """
    
    def __init__(self, model_path: str, output_path: str = "2d_traffic_demo.mp4"):
        self.model_path = model_path
        self.output_path = output_path
        self.width = 1200
        self.height = 800
        self.fps = 20
        
        # Initialize environment
        self.env = TrafficJunctionEnv(render_mode="human")
        
        # Load model
        try:
            self.model = PPO.load(model_path)
            print(f"Successfully loaded model from {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using random actions instead...")
            self.model = None
            self.model_loaded = False
        
        # Initialize visualizer
        self.visualizer = TrafficVisualizer(self.env, self.width, self.height)
        
        # Initialize demo statistics
        self.demo_stats = {
            "creation_timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "model_loaded": self.model_loaded,
            "output_video_path": output_path,
            "video_config": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps
            }
        }
        
    def pygame_to_cv2(self, pygame_surface):
        """Convert pygame surface to OpenCV format"""
        # Get pygame surface as numpy array
        w, h = pygame_surface.get_size()
        raw = pygame.image.tostring(pygame_surface, 'RGB')
        
        # Convert to numpy array
        image = np.frombuffer(raw, dtype=np.uint8)
        image = image.reshape((h, w, 3))
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def add_text_overlay(self, frame, episode, step, reward, action, total_reward):
        """Add text overlay to frame"""
        
        # Action names
        action_names = [
            "Extend NS Green", "Extend EW Green", "Switch to NS", "Switch to EW",
            "Emergency Priority", "All Red", "Reset Timer", "Short Cycle", "Extended Cycle"
        ]
        
        # Add black background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        texts = [
            f"RWANDA TRAFFIC RL AGENT DEMO",
            f"Episode: {episode}/3",
            f"Step: {step}",
            f"Current Reward: {reward:.2f}",
            f"Total Reward: {total_reward:.2f}",
            f"Action: {action_names[action] if 0 <= action < len(action_names) else 'Random'}",
            f"Light State: {self.env.current_light.name}",
            f"Timer: {self.env.light_timer}s"
        ]
        
        for i, text in enumerate(texts):
            y = 35 + i * 22
            cv2.putText(frame, text, (20, y), font, font_scale, color, thickness)
        
        return frame
    
    def save_summary_json(self, episode_data, total_frames):
        """Save demo results to JSON file"""
        
        # Calculate summary statistics
        episode_rewards = [ep["total_reward"] for ep in episode_data]
        
        summary = {
            **self.demo_stats,
            "episodes": {
                "count": len(episode_data),
                "data": episode_data,
                "statistics": {
                    "total_reward_sum": float(sum(episode_rewards)),
                    "average_reward": float(np.mean(episode_rewards)),
                    "min_reward": float(min(episode_rewards)),
                    "max_reward": float(max(episode_rewards)),
                    "reward_std": float(np.std(episode_rewards)),
                    "best_episode": int(np.argmax(episode_rewards)) + 1,
                    "worst_episode": int(np.argmin(episode_rewards)) + 1
                }
            },
            "video_stats": {
                "total_frames": int(total_frames),
                "duration_seconds": float(total_frames / self.fps),
                "file_size_mb": float(os.path.getsize(self.output_path) / (1024*1024)) if os.path.exists(self.output_path) else 0.0
            },
            "performance_evaluation": {
                "agent_type": "PPO_MODEL" if self.model_loaded else "RANDOM_ACTIONS",
                "performance_rating": self._get_performance_rating(np.mean(episode_rewards)),
                "completion_status": "SUCCESS"
            }
        }
        
        # Save JSON file
        json_path = self.output_path.replace('.mp4', '_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary JSON saved to: {json_path}")
        return json_path
    
    def save_readme(self, episode_data, total_frames, json_path):
        """Save README file with demo information"""
        
        episode_rewards = [ep["total_reward"] for ep in episode_data]
        avg_reward = float(np.mean(episode_rewards))
        best_episode_idx = int(np.argmax(episode_rewards)) 
        worst_episode_idx = int(np.argmin(episode_rewards))
        
        readme_content = f"""# Rwanda Traffic Junction RL Agent Demo

## Overview
This video demonstration showcases a Reinforcement Learning agent trained to control traffic lights at a busy junction in Rwanda. The agent uses PPO (Proximal Policy Optimization) to learn optimal traffic management strategies.

## Demo Information
- **Created**: {self.demo_stats['creation_timestamp']}
- **Agent Type**: {'PPO Model' if self.model_loaded else 'Random Actions'}
- **Model Path**: `{self.model_path}`
- **Video Output**: `{self.output_path}`
- **Summary Data**: `{os.path.basename(json_path)}`

## Video Specifications
- **Resolution**: {self.width}x{self.height} pixels
- **Frame Rate**: {self.fps} FPS
- **Duration**: {total_frames / self.fps:.1f} seconds
- **Total Frames**: {total_frames:,}
- **File Size**: {os.path.getsize(self.output_path) / (1024*1024):.1f} MB

## Episode Performance

| Episode | Steps | Total Reward | Performance |
|---------|-------|--------------|-------------|
"""
        
        for i, ep in enumerate(episode_data, 1):
            if ep["total_reward"] > -100:
                performance = "EXCELLENT"
            elif ep["total_reward"] > -300:
                performance = "GOOD"
            else:
                performance = "NEEDS WORK"
            readme_content += f"| {i} | {ep['steps']} | {ep['total_reward']:.2f} | {performance} |\n"
        
        readme_content += f"""
## Summary Statistics
- **Average Reward**: {avg_reward:.2f}
- **Best Episode**: Episode {best_episode_idx + 1} ({max(episode_rewards):.2f} reward)
- **Worst Episode**: Episode {worst_episode_idx + 1} ({min(episode_rewards):.2f} reward)
- **Performance Rating**: {self._get_performance_rating(avg_reward)}

## Actions Available
The agent can choose from 9 different actions:
1. **Extend NS Green** - Keep North-South traffic flowing longer
2. **Extend EW Green** - Keep East-West traffic flowing longer  
3. **Switch to NS** - Change to North-South green phase
4. **Switch to EW** - Change to East-West green phase
5. **Emergency Priority** - Activate emergency vehicle protocol
6. **All Red** - Stop all traffic temporarily
7. **Reset Timer** - Reset current phase timer
8. **Short Cycle** - Use shorter light cycle
9. **Extended Cycle** - Use longer light cycle

## Environment Details
- **State Space**: Vehicle positions, waiting times, light phases
- **Reward Function**: Based on traffic flow efficiency and waiting times
- **Traffic Pattern**: Realistic Rwanda urban traffic simulation

## How to Run
```bash
python traffic_video_demo.py
```

## Files Generated
- `{os.path.basename(self.output_path)}` - Main demonstration video
- `{os.path.basename(json_path)}` - Detailed statistics and metadata
- `README.md` - This documentation file

## Notes
{('SUCCESS: Model successfully loaded and used for intelligent traffic control' if self.model_loaded else 'WARNING: Model could not be loaded - using random actions for demonstration')}

---
*Generated automatically by Rwanda Traffic RL Demo System*
"""
        
        # Save README file
        readme_path = self.output_path.replace('.mp4', '_README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"README saved to: {readme_path}")
        return readme_path
    
    def _get_performance_rating(self, avg_reward):
        """Evaluate agent performance based on average reward"""
        if avg_reward > -100:
            return "EXCELLENT"
        elif avg_reward > -200:
            return "VERY_GOOD"
        elif avg_reward > -300:
            return "GOOD"
        elif avg_reward > -400:
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def create_demo_video(self):
        """Create the demo video"""
        
        print("Starting traffic simulation video creation...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        total_frames = 0
        episode_data = []
        
        # Run 3 episodes
        for episode in range(1, 4):
            print(f"\nRecording Episode {episode}/3...")
            
            obs, _ = self.env.reset()
            episode_reward = 0
            step = 0
            episode_actions = []
            episode_rewards = []
            
            while step < 60:  # 60 steps per episode
                
                # Get action from model or random
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Take environment step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Store episode data
                episode_actions.append(int(action))
                episode_rewards.append(float(reward))
                
                # Render the environment
                self.visualizer.render(action)
                
                # Get pygame surface and convert to OpenCV
                pygame_surface = self.visualizer.screen
                frame = self.pygame_to_cv2(pygame_surface)
                
                # Add information overlay
                frame = self.add_text_overlay(frame, episode, step, reward, action, episode_reward)
                
                # Write frame to video
                out.write(frame)
                total_frames += 1
                
                step += 1
                
                # Handle pygame events to prevent freezing
                if not self.visualizer.handle_events():
                    break
                
                if terminated or truncated:
                    break
            
            # Store episode data
            episode_info = {
                "episode_number": int(episode),
                "steps": int(step),
                "total_reward": float(episode_reward),
                "average_reward_per_step": float(episode_reward / step) if step > 0 else 0.0,
                "actions_taken": [int(a) for a in episode_actions],
                "step_rewards": [float(r) for r in episode_rewards],
                "completed": not (terminated or truncated)
            }
            episode_data.append(episode_info)
            
            print(f"Episode {episode} completed with reward: {episode_reward:.2f}")
        
        # Add summary frames
        print("Adding summary frames...")
        avg_reward = float(np.mean([ep["total_reward"] for ep in episode_data]))
        best_episode_idx = int(np.argmax([ep['total_reward'] for ep in episode_data]))
        
        for _ in range(60):  # 3 seconds of summary
            summary_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Add gradient background
            for y in range(self.height):
                intensity = int(30 + 20 * np.sin(y / 100))
                summary_frame[y, :] = [0, intensity, 0]
            
            # Add summary text
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            summary_texts = [
                "DEMONSTRATION COMPLETE!",
                f"Episodes: 3",
                f"Total Frames: {total_frames}",
                f"Average Reward: {avg_reward:.2f}",
                f"Best Episode: {best_episode_idx + 1}",
                f"Agent Performance: {self._get_performance_rating(avg_reward)}"
            ]
            
            for i, text in enumerate(summary_texts):
                text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
                x = (self.width - text_size[0]) // 2
                y = self.height // 2 - 100 + i * 50
                
                cv2.putText(summary_frame, text, (x, y), font, 1.2, (255, 255, 255), 2)
            
            out.write(summary_frame)
            total_frames += 1
        
        # Clean up video
        out.release()
        self.visualizer.close()
        
        # Save additional files
        json_path = self.save_summary_json(episode_data, total_frames)
        readme_path = self.save_readme(episode_data, total_frames, json_path)
        
        # Final summary
        final_avg_reward = float(np.mean([ep["total_reward"] for ep in episode_data]))
        print(f"\n" + "="*60)
        print(f"DEMO CREATION COMPLETE!")
        print(f"="*60)
        print(f"Video: {self.output_path}")
        print(f"JSON Summary: {json_path}")
        print(f"README: {readme_path}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames / self.fps:.1f} seconds")
        print(f"Average reward: {final_avg_reward:.2f}")
        print(f"Performance: {self._get_performance_rating(final_avg_reward)}")
        print(f"="*60)

def main():
    """Main function to create the demo"""
    
    # Path to your trained model
    model_path = "models/ppo_tuning/aggressive/ppo_aggressive_final.zip"
    
    # Create demo
    demo = TrafficVideoDemo(
        model_path=model_path,
        output_path="videos/rwanda_traffic_2d_demo.mp4"
    )
    
    try:
        demo.create_demo_video()
    except Exception as e:
        print(f"Error during video creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()