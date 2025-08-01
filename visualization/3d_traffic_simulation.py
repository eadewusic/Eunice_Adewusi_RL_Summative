import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import math
import sys

# Add the environment path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_junction_env import TrafficJunctionEnv, TrafficDirection, TrafficLightState

class EnhancedTrafficVideoCreator:
    """
    Enhanced 3D Traffic Video Creator with Comprehensive Documentation
    """
    
    def __init__(self, model_path: str, output_folder: str = "videos"):
        self.model_path = model_path
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"traffic_3d_demo_{self.timestamp}"
        
        # Video settings
        self.width = 1280
        self.height = 720
        self.fps = 20
        
        # Comprehensive tracking
        self.session_data = {
            "session_info": {
                "session_name": self.session_name,
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "video_settings": {
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps
                }
            },
            "episodes": [],
            "overall_performance": {},
            "technical_details": {},
            "environment_info": {}
        }
        
        # Initialize environment
        self.env = TrafficJunctionEnv(render_mode="human")
        self._capture_environment_info()
        
        # Load model
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            print(f"Successfully loaded model from {model_path}")
            self.session_data["technical_details"]["model_loaded"] = True
            self.session_data["technical_details"]["model_type"] = "PPO"
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using random actions instead...")
            self.model = None
            self.session_data["technical_details"]["model_loaded"] = False
            self.session_data["technical_details"]["fallback_mode"] = "random_actions"
        
        # Initialize 3D visualizer
        self._init_pybullet()
        
    def _capture_environment_info(self):
        """Capture detailed environment information"""
        self.session_data["environment_info"] = {
            "environment_type": "Rwanda Traffic Junction",
            "observation_space_size": self.env.observation_space.shape[0],
            "action_space_size": self.env.action_space.n,
            "action_descriptions": [
                "Extend North-South Green",
                "Extend East-West Green", 
                "Switch to North-South Green",
                "Switch to East-West Green",
                "Emergency Priority Override",
                "All Red Transition",
                "Reset Timer to Default",
                "Short Green Cycle",
                "Extended Green Cycle"
            ],
            "traffic_directions": ["North", "South", "East", "West"],
            "max_queue_length": self.env.max_queue_length,
            "light_timer_max": self.env.light_timer_max
        }
    
    def _init_pybullet(self):
        """Initialize PyBullet 3D environment"""
        print("Initializing PyBullet 3D visualization...")
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure rendering
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.setGravity(0, 0, -9.81)
        
        # Initialize tracking
        self.vehicle_objects = {}
        self.traffic_light_objects = {}
        
        # Camera settings
        self.camera_target = [0, 0, 0]
        self.camera_distance = 25
        self.camera_yaw = 45
        self.camera_pitch = -30
        
        # Create 3D environment
        self._create_3d_environment()
        
    def _create_3d_environment(self):
        """Create the complete 3D environment"""
        print("Building 3D traffic junction...")
        
        # Ground plane
        self.ground_id = p.loadURDF("plane.urdf", [0, 0, 0])
        p.changeVisualShape(self.ground_id, -1, rgbaColor=[0.2, 0.6, 0.2, 1])
        
        # Roads
        self._create_roads()
        self._create_traffic_lights()
        self._create_buildings()
        self._setup_camera()
        
        print("3D environment created successfully!")
    
    def _create_roads(self):
        """Create road infrastructure"""
        road_color = [0.3, 0.3, 0.3, 1]
        
        # Central intersection
        intersection_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[5, 5, 0.01], rgbaColor=[0.4, 0.4, 0.4, 1]
        )
        self.intersection_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=intersection_visual, basePosition=[0, 0, 0.01]
        )
        
        # North-South road
        ns_road_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[2.5, 15, 0.01], rgbaColor=road_color
        )
        self.ns_road = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=ns_road_visual, basePosition=[0, 0, 0.01]
        )
        
        # East-West road
        ew_road_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[15, 2.5, 0.01], rgbaColor=road_color
        )
        self.ew_road = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=ew_road_visual, basePosition=[0, 0, 0.01]
        )
    
    def _create_traffic_lights(self):
        """Create traffic light system"""
        light_positions = [(6, 6, 0), (-6, 6, 0), (-6, -6, 0), (6, -6, 0)]
        
        for i, pos in enumerate(light_positions):
            # Pole
            pole_visual = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.1, length=4, rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            pole_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=pole_visual, basePosition=[pos[0], pos[1], 2]
            )
            
            # Traffic light
            light_visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1]
            )
            light_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=light_visual, basePosition=[pos[0], pos[1], 4.5]
            )
            
            self.traffic_light_objects[i] = {"pole": pole_id, "light": light_id}
    
    def _create_buildings(self):
        """Create surrounding buildings"""
        building_positions = [(12, 12, 0), (-12, 12, 0), (-12, -12, 0), (12, -12, 0)]
        
        for i, pos in enumerate(building_positions):
            height = 8
            building_visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[3, 3, height/2], rgbaColor=[0.7, 0.5, 0.3, 1]
            )
            p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=building_visual, 
                basePosition=[pos[0], pos[1], height/2]
            )
    
    def _setup_camera(self):
        """Setup optimal camera view"""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target
        )
    
    def update_visualization(self, episode, step, reward, action, total_reward):
        """Update all visual elements"""
        self._update_traffic_lights()
        self._update_vehicles()
        self._add_hud_display(episode, step, reward, action, total_reward)
        p.stepSimulation()
    
    def _update_traffic_lights(self):
        """Update traffic light colors"""
        if self.env.current_light == TrafficLightState.NORTH_SOUTH_GREEN:
            color = [0, 1, 0, 1]  # Green
        elif self.env.current_light == TrafficLightState.EAST_WEST_GREEN:
            color = [0, 1, 0, 1]  # Green
        else:
            color = [1, 0, 0, 1]  # Red
        
        for light_set in self.traffic_light_objects.values():
            p.changeVisualShape(light_set['light'], -1, rgbaColor=color)
    
    def _update_vehicles(self):
        """Update vehicle positions"""
        # Clear existing vehicles
        for vehicle_id in list(self.vehicle_objects.values()):
            try:
                p.removeBody(vehicle_id)
            except:
                pass
        self.vehicle_objects.clear()
        
        # Vehicle placement functions
        positions = {
            TrafficDirection.NORTH: lambda i: (0, 8 + i * 2, 0.5),
            TrafficDirection.SOUTH: lambda i: (0, -8 - i * 2, 0.5),
            TrafficDirection.EAST: lambda i: (8 + i * 2, 0, 0.5),
            TrafficDirection.WEST: lambda i: (-8 - i * 2, 0, 0.5)
        }
        
        colors = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0.5, 0.5, 0.5, 1], [1, 1, 1, 1]]
        
        # Create vehicles
        for direction in TrafficDirection:
            queue_length = min(self.env.vehicle_queues[direction], 6)
            
            for i in range(queue_length):
                pos = positions[direction](i)
                color = colors[i % len(colors)]
                
                vehicle_visual = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.8, 0.4, 0.3], rgbaColor=color
                )
                
                try:
                    vehicle_id = p.createMultiBody(
                        baseMass=0, baseVisualShapeIndex=vehicle_visual, basePosition=pos
                    )
                    self.vehicle_objects[f"{direction.name}_{i}"] = vehicle_id
                    
                    # Emergency vehicle highlighting
                    if self.env.emergency_vehicles[direction] and i == 0:
                        p.changeVisualShape(vehicle_id, -1, rgbaColor=[1, 0, 0, 1])
                except:
                    pass
    
    def _add_hud_display(self, episode, step, reward, action, total_reward):
        """Add comprehensive HUD display with left/right layout"""
        p.removeAllUserDebugItems()
        
        action_names = [
            "Extend NS Green", "Extend EW Green", "Switch to NS", "Switch to EW",
            "Emergency Priority", "All Red", "Reset Timer", "Short Cycle", "Extended Cycle"
        ]
        
        # LEFT SIDE - Main performance info
        left_info_lines = [
            f"TRAFFIC RL AGENT",
            f"Episode: {episode}/3",
            f"Step: {step}",
            f"Current Reward: {reward:.2f}",
            f"Episode Total: {total_reward:.2f}",
            f"Action: {action_names[action] if 0 <= action < len(action_names) else 'Unknown'}",
            f"Light State: {self.env.current_light.name}",
            f"Timer Remaining: {self.env.light_timer}s"
        ]
        
        # RIGHT SIDE - Traffic queues info
        right_info_lines = [
            "TRAFFIC QUEUES:",
            f"North: {self.env.vehicle_queues[TrafficDirection.NORTH]} vehicles",
            f"South: {self.env.vehicle_queues[TrafficDirection.SOUTH]} vehicles", 
            f"East: {self.env.vehicle_queues[TrafficDirection.EAST]} vehicles",
            f"West: {self.env.vehicle_queues[TrafficDirection.WEST]} vehicles",
            "",
            f"Total Waiting: {sum(self.env.vehicle_queues.values())} vehicles",
            f"Processed: {self.env.vehicles_processed} vehicles"
        ]
        
        # Display LEFT SIDE text
        for i, line in enumerate(left_info_lines):
            p.addUserDebugText(
                line, textPosition=[-15, 15, 19.7 - i * 1.2],  # Left side (negative x)
                textColorRGB=[1, 1, 1], textSize=1.0
            )

        # Display RIGHT SIDE text
        for i, line in enumerate(right_info_lines):
            p.addUserDebugText(
                line, textPosition=[8, 18, 17 - i * 0.8],   # Right side (positive x)
                textColorRGB=[1, 1, 1], textSize=1.0
            )
    
    def capture_frame(self, episode, step, reward, action, total_reward):
        """Capture a single frame"""
        self.update_visualization(episode, step, reward, action, total_reward)
        
        # Get camera matrices
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target, distance=self.camera_distance,
            yaw=self.camera_yaw, pitch=self.camera_pitch, roll=0, upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.width / self.height, nearVal=0.1, farVal=100.0
        )
        
        # Capture image
        try:
            _, _, color_img, _, _ = p.getCameraImage(
                width=self.width, height=self.height,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix
            )
            
            color_array = np.array(color_img, dtype=np.uint8)
            color_array = color_array.reshape((self.height, self.width, 4))
            return color_array[:, :, :3]  # Remove alpha
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def create_comprehensive_demo(self):
        """Create complete demonstration with full documentation"""
        
        print("Starting comprehensive 3D traffic RL demon...")
        
        # Setup video output
        video_path = self.output_folder / f"{self.session_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
        
        total_frames = 0
        episode_data = []
        
        # Record 3 episodes
        for episode in range(1, 4):
            print(f"\nRecording Episode {episode}/3...")
            
            obs, _ = self.env.reset()
            episode_reward = 0
            step = 0
            
            episode_info = {
                "episode_number": episode,
                "steps": [],
                "total_reward": 0,
                "final_metrics": {}
            }
            
            while step < 50:  # 50 steps per episode
                # Get action
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                # Environment step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Record step data
                step_data = {
                    "step": int(step),
                    "action": int(action),
                    "reward": float(reward),
                    "cumulative_reward": float(episode_reward),
                    "light_state": str(self.env.current_light.name),
                    "light_timer": int(self.env.light_timer),
                    "queue_lengths": {
                        "north": int(self.env.vehicle_queues[TrafficDirection.NORTH]),
                        "south": int(self.env.vehicle_queues[TrafficDirection.SOUTH]),
                        "east": int(self.env.vehicle_queues[TrafficDirection.EAST]),
                        "west": int(self.env.vehicle_queues[TrafficDirection.WEST])
                    },
                    "total_waiting": int(sum(self.env.vehicle_queues.values())),
                    "vehicles_processed": int(self.env.vehicles_processed)
                }
                episode_info["steps"].append(step_data)
                
                # Capture and save frame
                frame = self.capture_frame(episode, step, reward, action, episode_reward)
                if frame is not None:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                    total_frames += 1
                
                step += 1
                time.sleep(0.03)  # Slight delay for smooth playback
                
                if terminated or truncated:
                    break
            
            # Complete episode data
            episode_info["total_reward"] = float(episode_reward)
            episode_info["final_metrics"] = {
                "total_steps": int(step),
                "avg_reward_per_step": float(episode_reward / step) if step > 0 else 0.0,
                "final_queue_total": int(sum(self.env.vehicle_queues.values())),
                "vehicles_processed": int(self.env.vehicles_processed)
            }
            
            episode_data.append(episode_info)
            print(f"Episode {episode} completed: {episode_reward:.2f} reward")
        
        out.release()
        
        # Calculate overall performance
        episode_rewards = [ep["total_reward"] for ep in episode_data]
        self.session_data["episodes"] = episode_data
        self.session_data["overall_performance"] = {
            "total_episodes": 3,
            "episode_rewards": [float(r) for r in episode_rewards],
            "average_reward": float(np.mean(episode_rewards)),
            "best_episode": int(np.argmax(episode_rewards) + 1),
            "worst_episode": int(np.argmin(episode_rewards) + 1),
            "reward_std": float(np.std(episode_rewards)),
            "total_frames": int(total_frames),
            "video_duration_seconds": float(total_frames / self.fps),
            "performance_rating": self._get_performance_rating(np.mean(episode_rewards))
        }
        
        # Save all documentation
        self._save_documentation(video_path)
        
        print(f"\nComprehensive demonstration complete!")
        print(f"All files saved to: {self.output_folder}")
        return video_path
    
    def _get_performance_rating(self, avg_reward):
        """Determine performance rating"""
        if avg_reward > -100:
            return "EXCELLENT"
        elif avg_reward > -150:
            return "VERY_GOOD"
        elif avg_reward > -200:
            return "GOOD"
        elif avg_reward > -250:
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _save_documentation(self, video_path):
        """Save comprehensive documentation files"""
        
        # Convert numpy types to JSON-serializable types
        json_safe_data = self._convert_numpy_types(self.session_data)
        
        # 1. Save JSON summary
        json_path = self.output_folder / f"{self.session_name}_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_data, f, indent=2, ensure_ascii=False)
        
        # 2. Save detailed README
        readme_path = self.output_folder / f"{self.session_name}_README.md"
        self._create_detailed_readme(readme_path, video_path)
        
        # 3. Save quick stats
        stats_path = self.output_folder / f"{self.session_name}_quick_stats.txt"
        self._create_quick_stats(stats_path)
        
        print(f"Documentation saved:")
        print(f"   • Video: {video_path}")
        print(f"   • Summary JSON: {json_path}")
        print(f"   • Detailed README: {readme_path}")
        print(f"   • Quick Stats: {stats_path}")
    
    def _create_detailed_readme(self, readme_path, video_path):
        """Create comprehensive README file"""
        
        perf = self.session_data["overall_performance"]
        env_info = self.session_data["environment_info"]
        
        readme_content = f"""# Rwanda Traffic Junction 3D RL Agent Demo

## Video Information
- **File**: `{video_path.name}`
- **Duration**: {perf['video_duration_seconds']:.1f} seconds ({perf['total_frames']} frames)
- **Resolution**: {self.width}x{self.height} @ {self.fps} FPS
- **Created**: {self.session_data['session_info']['timestamp']}

## AI Agent Details
- **Algorithm**: {self.session_data['technical_details'].get('model_type', 'Random')}
- **Model Path**: `{self.model_path}`
- **Model Loaded**: {'Yes' if self.session_data['technical_details']['model_loaded'] else '❌ No (using random actions)'}

## Traffic Environment
- **Location**: Rwanda Traffic Junction (Kigali-inspired)
- **Observation Space**: {env_info['observation_space_size']} dimensions
- **Action Space**: {env_info['action_space_size']} discrete actions
- **Traffic Directions**: {', '.join(env_info['traffic_directions'])}

## Available Actions
The RL agent can choose from {env_info['action_space_size']} different traffic control actions:

"""
        
        for i, action_desc in enumerate(env_info['action_descriptions']):
            readme_content += f"{i}. **{action_desc}**\n"
        
        readme_content += f"""

## Performance Summary

### Overall Results
- **Episodes Completed**: {perf['total_episodes']}
- **Average Reward**: {perf['average_reward']:.2f}
- **Performance Rating**: **{perf['performance_rating']}**
- **Best Episode**: Episode {perf['best_episode']} ({max(perf['episode_rewards']):.2f} reward)
- **Worst Episode**: Episode {perf['worst_episode']} ({min(perf['episode_rewards']):.2f} reward)
- **Consistency (Std Dev)**: {perf['reward_std']:.2f}

### Episode Breakdown
"""
        
        for i, episode in enumerate(self.session_data['episodes'], 1):
            final_metrics = episode['final_metrics']
            readme_content += f"""
#### Episode {i}
- **Total Reward**: {episode['total_reward']:.2f}
- **Steps Completed**: {final_metrics['total_steps']}
- **Average Reward/Step**: {final_metrics['avg_reward_per_step']:.3f}
- **Final Queue Length**: {final_metrics['final_queue_total']} vehicles
- **Vehicles Processed**: {final_metrics['vehicles_processed']}
"""
        
        readme_content += f"""

## What You'll See in the Video

### Visual Elements
1. **3D Traffic Junction**: Realistic intersection with roads, buildings, and traffic lights
2. **Dynamic Vehicles**: Colored vehicles representing traffic queues from each direction
3. **Traffic Light System**: Real-time traffic light changes based on agent decisions
4. **Performance HUD**: Live metrics including:
   - Current episode and step
   - Real-time reward tracking
   - Current action being taken
   - Traffic light state and timer
   - Queue lengths for each direction
   - Total vehicles waiting and processed

### Episode Structure
The video shows **3 complete episodes** of the RL agent controlling traffic:

- **Episode 1**: {self.session_data['episodes'][0]['total_reward']:.2f} reward - Agent learning traffic patterns
- **Episode 2**: {self.session_data['episodes'][1]['total_reward']:.2f} reward - Adaptation to different traffic scenarios  
- **Episode 3**: {self.session_data['episodes'][2]['total_reward']:.2f} reward - Final performance demonstration

## How the Agent Works

### Reward System
The agent receives rewards/penalties based on:
- **Positive**: Keeping traffic flowing smoothly, appropriate light timing
- **Negative**: Long queues, unnecessary light switches, gridlock conditions

### State Information
The agent observes:
- Vehicle queue lengths in all directions
- Current traffic light state and remaining time
- Time of day (affects traffic patterns)
- Emergency vehicle presence
- Historical traffic flow metrics

### Decision Making
Each step, the agent:
1. Observes current traffic state
2. Selects optimal action using trained policy
3. Receives reward based on traffic flow improvement
4. Updates traffic light system accordingly

## Rwanda Context
This simulation represents traffic management challenges in Kigali, Rwanda:
- **Rush Hour Patterns**: Morning (7-9 AM) and evening (5-7 PM) traffic surges
- **Vehicle Mix**: Cars, buses (important public transport), motorcycles (very popular)
- **Emergency Priority**: Ambulances and emergency vehicles get immediate priority
- **Real-world Constraints**: Minimum light timing, pedestrian considerations

## Performance Interpretation

### Reward Scale Understanding
- **Above -100**: Excellent traffic management, minimal delays
- **-100 to -150**: Very good performance, occasional minor delays
- **-150 to -200**: Good performance, manageable traffic flow
- **-200 to -250**: Fair performance, some congestion issues
- **Below -250**: Needs improvement, significant delays

### Agent Performance: **{perf['performance_rating']}**
Average reward of **{perf['average_reward']:.2f}** indicates {self._get_performance_description(perf['performance_rating'])}.

## Technical Details
- **Simulation Engine**: PyBullet (3D physics simulation)
- **RL Framework**: Stable-Baselines3 (PPO algorithm)
- **Environment**: Custom Gymnasium environment
- **Visualization**: Real-time 3D rendering with performance metrics
- **Video Encoding**: OpenCV with MP4V codec

## Additional Files
- `{self.session_name}_summary.json`: Complete session data in JSON format
- `{self.session_name}_quick_stats.txt`: Quick performance overview
- `{video_path.name}`: The main demonstration video

---
*Generated automatically by Rwanda Traffic Junction RL Demonstration System*
*Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _get_performance_description(self, rating):
        """Get description for performance rating"""
        descriptions = {
            "EXCELLENT": "outstanding traffic management with minimal delays and optimal flow",
            "VERY_GOOD": "very effective traffic control with only minor occasional delays", 
            "GOOD": "solid traffic management with manageable congestion levels",
            "FAIR": "acceptable performance but with room for improvement in reducing delays",
            "NEEDS_IMPROVEMENT": "significant optimization needed to reduce traffic congestion"
        }
        return descriptions.get(rating, "performance analysis")
    
    def _create_quick_stats(self, stats_path):
        """Create quick stats file"""
        perf = self.session_data["overall_performance"]
        
        stats_content = f"""RWANDA TRAFFIC RL AGENT - QUICK STATS
======================================

SESSION: {self.session_name}
TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PERFORMANCE SUMMARY:
- Average Reward: {perf['average_reward']:.2f}
- Performance Rating: {perf['performance_rating']}
- Best Episode: {perf['best_episode']} ({max(perf['episode_rewards']):.2f})
- Episodes: {perf['episode_rewards']}
- Video Duration: {perf['video_duration_seconds']:.1f}s
- Total Frames: {perf['total_frames']}

MODEL INFO:
- Type: {self.session_data['technical_details'].get('model_type', 'Random')}
- Loaded: {'Yes' if self.session_data['technical_details']['model_loaded'] else 'No'}
- Path: {self.model_path}

TECHNICAL:
- Resolution: {self.width}x{self.height}
- FPS: {self.fps}
- Engine: PyBullet 3D
"""
        
        with open(stats_path, 'w') as f:
            f.write(stats_content)
    
    def close(self):
        """Clean up resources"""
        print("Cleaning up PyBullet...")
        p.disconnect()

def main():
    """Main execution function"""
    
    # Configuration
    model_path = "models/ppo_tuning/aggressive/ppo_aggressive_final.zip"
    output_folder = "videos"
    
    print("Initializing Enhanced Traffic RL Video Creator...")
    
    # Create comprehensive demonstration
    creator = EnhancedTrafficVideoCreator(model_path, output_folder)
    
    try:
        video_path = creator.create_comprehensive_demo()
        print(f"\nSUCCESS! Complete demonstration package created!")
        print(f"Check the '{output_folder}' folder for all files.")
        
    except Exception as e:
        print(f"Error during creation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        creator.close()

if __name__ == "__main__":
    main()