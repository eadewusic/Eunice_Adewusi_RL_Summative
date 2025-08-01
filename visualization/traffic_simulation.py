import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
from typing import Dict, List, Tuple
import math
import sys
import os

# Add the environment path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_junction_env import TrafficJunctionEnv, TrafficDirection, TrafficLightState

class PyBullet3DTrafficVisualizer:
    """
    Fixed Advanced 3D visualization for traffic junction using PyBullet
    """
    
    def __init__(self, env: TrafficJunctionEnv, width: int = 1280, height: int = 720):
        self.env = env
        self.width = width
        self.height = height
        
        # Initialize all attributes first
        self.vehicle_objects = {}
        self.traffic_light_objects = {}
        self.last_vehicle_id = 0
        self.performance_data = []
        
        # Initialize PyBullet
        print("Initializing PyBullet...")
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure camera and rendering
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Keep GUI for debugging
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        # Camera setup attributes
        self.camera_target = [0, 0, 0]
        self.camera_distance = 25
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.current_camera = 0
        
        # Create the environment
        print("Creating 3D environment...")
        self._create_3d_environment()
        
    def _create_3d_environment(self):
        """Create the 3D traffic junction environment"""
        
        # Load ground plane
        print("Loading ground plane...")
        self.ground_id = p.loadURDF("plane.urdf", [0, 0, 0])
        p.changeVisualShape(self.ground_id, -1, rgbaColor=[0.2, 0.6, 0.2, 1])
        
        # Create roads
        print("Creating roads...")
        self._create_roads()
        
        # Create traffic lights
        print("Creating traffic lights...")
        self._create_traffic_lights()
        
        # Create buildings
        print("Creating buildings...")
        self._create_surrounding_buildings()
        
        # Set up camera
        print("Setting up camera...")
        self._setup_camera()
        
        print("3D environment created successfully!")
    
    def _create_roads(self):
        """Create road structures"""
        road_color = [0.3, 0.3, 0.3, 1]
        
        # Main intersection
        intersection_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[5, 5, 0.01], 
            rgbaColor=[0.4, 0.4, 0.4, 1]
        )
        self.intersection_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=intersection_visual,
            basePosition=[0, 0, 0.01]
        )
        
        # North-South road
        ns_road_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[2.5, 15, 0.01], 
            rgbaColor=road_color
        )
        self.ns_road = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=ns_road_visual,
            basePosition=[0, 0, 0.01]
        )
        
        # East-West road  
        ew_road_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[15, 2.5, 0.01], 
            rgbaColor=road_color
        )
        self.ew_road = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=ew_road_visual,
            basePosition=[0, 0, 0.01]
        )
    
    def _create_traffic_lights(self):
        """Create simplified traffic light structures"""
        
        # Traffic light positions (corners of intersection)
        light_positions = [
            (6, 6, 0),    # NE
            (-6, 6, 0),   # NW  
            (-6, -6, 0),  # SW
            (6, -6, 0)    # SE
        ]
        
        for i, pos in enumerate(light_positions):
            # Create pole
            pole_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.1,
                length=4,
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            pole_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=pole_visual,
                basePosition=[pos[0], pos[1], 2]
            )
            
            # Create single traffic light (simplified)
            light_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.2,
                rgbaColor=[1, 0, 0, 1]  # Start with red
            )
            light_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=light_visual,
                basePosition=[pos[0], pos[1], 4.5]
            )
            
            # Store light objects
            self.traffic_light_objects[i] = {
                'pole': pole_id,
                'light': light_id
            }
    
    def _create_surrounding_buildings(self):
        """Create simplified buildings around the intersection"""
        building_positions = [
            (12, 12, 0), (-12, 12, 0), (-12, -12, 0), (12, -12, 0)
        ]
        
        self.buildings = []
        for i, pos in enumerate(building_positions):
            # Simple building
            height = 8
            building_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[3, 3, height/2],
                rgbaColor=[0.7, 0.5, 0.3, 1]
            )
            building_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=building_visual,
                basePosition=[pos[0], pos[1], height/2]
            )
            self.buildings.append(building_id)
    
    def _setup_camera(self):
        """Setup camera view"""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target
        )
    
    def update_traffic_lights(self):
        """Update traffic light colors based on environment state"""
        current_light = self.env.current_light
        
        # Determine light color
        if current_light == TrafficLightState.NORTH_SOUTH_GREEN:
            light_color = [0, 1, 0, 1]  # Green
        elif current_light == TrafficLightState.EAST_WEST_GREEN:
            light_color = [0, 1, 0, 1]  # Green
        else:  # ALL_RED
            light_color = [1, 0, 0, 1]  # Red
        
        # Update all traffic lights
        for light_set in self.traffic_light_objects.values():
            light_id = light_set['light']
            p.changeVisualShape(light_id, -1, rgbaColor=light_color)
    
    def update_vehicles(self):
        """Update 3D vehicle representations"""
        
        # Remove existing vehicles
        for vehicle_id in list(self.vehicle_objects.values()):
            try:
                p.removeBody(vehicle_id)
            except:
                pass
        self.vehicle_objects.clear()
        
        # Vehicle positions for each direction
        vehicle_positions = {
            TrafficDirection.NORTH: lambda i: (0, 8 + i * 2, 0.5),
            TrafficDirection.SOUTH: lambda i: (0, -8 - i * 2, 0.5),
            TrafficDirection.EAST: lambda i: (8 + i * 2, 0, 0.5),
            TrafficDirection.WEST: lambda i: (-8 - i * 2, 0, 0.5)
        }
        
        # Vehicle colors
        colors = [
            [1, 0, 0, 1],     # Red
            [0, 0, 1, 1],     # Blue  
            [1, 1, 0, 1],     # Yellow
            [0.5, 0.5, 0.5, 1], # Gray
            [1, 1, 1, 1],     # White
        ]
        
        # Create vehicles based on queue lengths
        for direction in TrafficDirection:
            queue_length = min(self.env.vehicle_queues[direction], 6)  # Limit to 6 vehicles
            
            for i in range(queue_length):
                pos = vehicle_positions[direction](i)
                color = colors[i % len(colors)]
                
                # Create simple box vehicle
                vehicle_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.8, 0.4, 0.3],
                    rgbaColor=color
                )
                
                try:
                    vehicle_id = p.createMultiBody(
                        baseMass=0,  # Static for simplicity
                        baseVisualShapeIndex=vehicle_visual,
                        basePosition=pos
                    )
                    
                    self.vehicle_objects[f"{direction.name}_{i}"] = vehicle_id
                    
                    # Emergency vehicle special coloring
                    if self.env.emergency_vehicles[direction] and i == 0:
                        p.changeVisualShape(vehicle_id, -1, rgbaColor=[1, 0, 0, 1])
                        
                except Exception as e:
                    print(f"Error creating vehicle: {e}")
    
    def add_debug_info(self, episode: int, step: int, reward: float, 
                      action: int, total_reward: float):
        """Add debug information as text"""
        
        # Clear previous debug items
        p.removeAllUserDebugItems()
        
        # Action names
        action_names = [
            "Extend NS Green", "Extend EW Green", "Switch to NS", "Switch to EW",
            "Emergency Priority", "All Red", "Reset Timer", "Short Cycle", "Extended Cycle"
        ]
        
        # Create info text
        info_lines = [
            f"Episode: {episode}/3",
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"Total: {total_reward:.2f}",
            f"Action: {action_names[action] if 0 <= action < len(action_names) else 'Unknown'}",
            f"Light: {self.env.current_light.name}",
            f"Timer: {self.env.light_timer}s",
            "",
            "Queue Lengths:",
            f"North: {self.env.vehicle_queues[TrafficDirection.NORTH]}",
            f"South: {self.env.vehicle_queues[TrafficDirection.SOUTH]}",
            f"East: {self.env.vehicle_queues[TrafficDirection.EAST]}",
            f"West: {self.env.vehicle_queues[TrafficDirection.WEST]}"
        ]
        
        # Add debug text
        for i, line in enumerate(info_lines):
            p.addUserDebugText(
                line,
                textPosition=[10, 12, 10 - i * 0.5],
                textColorRGB=[1, 1, 1],
                textSize=1.0
            )
    
    def render_and_save_frame(self, frame_num: int, episode: int, step: int, 
                             reward: float, action: int, total_reward: float) -> np.ndarray:
        """Render a single frame"""
        
        # Update visualization
        self.update_traffic_lights()
        self.update_vehicles()
        self.add_debug_info(episode, step, reward, action, total_reward)
        
        # Step simulation
        p.stepSimulation()
        
        # Get camera image
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Capture image
        try:
            _, _, color_img, _, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            
            # Convert to numpy array
            color_array = np.array(color_img, dtype=np.uint8)
            color_array = color_array.reshape((self.height, self.width, 4))
            color_array = color_array[:, :, :3]  # Remove alpha channel
            
            return color_array
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # Return black frame as fallback
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def create_demo_video(self, model_path: str, output_path: str = "traffic_3d_demo.mp4"):
        """Create demonstration video"""
        
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running with random actions...")
            model = None
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (self.width, self.height))
        
        frames_written = 0
        total_episode_rewards = []
        
        print("Starting 3D traffic demonstration...")
        
        # Record 3 episodes
        for episode in range(1, 4):
            print(f"\nRecording Episode {episode}/3...")
            
            obs, _ = self.env.reset()
            episode_reward = 0
            step = 0
            
            while step < 50:  # 50 steps per episode
                # Get action
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()  # Random action
                
                # Take step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Render and save frame
                frame = self.render_and_save_frame(
                    frames_written, episode, step, reward, action, episode_reward
                )
                
                if frame is not None:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                    frames_written += 1
                
                step += 1
                
                # Small delay for visualization
                time.sleep(0.05)
                
                if terminated or truncated:
                    break
            
            total_episode_rewards.append(episode_reward)
            print(f"Episode {episode} reward: {episode_reward:.2f}")
        
        out.release()
        print(f"\n3D video saved to: {output_path}")
        print(f"Total frames written: {frames_written}")
        print(f"Episode rewards: {total_episode_rewards}")
        print(f"Average reward: {np.mean(total_episode_rewards):.2f}")
    
    def close(self):
        """Clean up PyBullet"""
        print("Closing PyBullet...")
        p.disconnect()

def create_advanced_3d_demo():
    """Create the 3D demonstration"""
    
    print("Initializing traffic environment...")
    env = TrafficJunctionEnv()
    
    print("Creating 3D visualizer...")
    visualizer = PyBullet3DTrafficVisualizer(env, width=1280, height=720)
    
    # Path to your trained model
    model_path = "models/ppo_tuning/aggressive/ppo_aggressive_final.zip"
    
    try:
        visualizer.create_demo_video(
            model_path=model_path,
            output_path="rwanda_traffic_3d_demo.mp4"
        )
    except Exception as e:
        print(f"Error during video creation: {e}")
    finally:
        visualizer.close()
    
    print("3D demonstration complete!")

if __name__ == "__main__":
    create_advanced_3d_demo()