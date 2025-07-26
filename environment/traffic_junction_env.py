import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time

class TrafficDirection(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class TrafficLightState(Enum):
    NORTH_SOUTH_GREEN = 0
    EAST_WEST_GREEN = 1
    ALL_RED = 2

class HiddenTrafficState(Enum):
    FREE_FLOW = 0
    BUILDING_CONGESTION = 1
    PEAK_HOUR_GRIDLOCK = 2
    INCIDENT_RELATED_DELAYS = 3
    RECOVERY_PHASE = 4
    OFF_PEAK_FLOW = 5

class TrafficJunctionEnv(gym.Env):
    """
    Rwanda Traffic Junction Environment for RL Agent Training
    
    Mission: Replace manual road wardens by optimizing traffic light control
    through intelligent agent decision making.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment parameters
        self.grid_size = 5  # 5x5 grid per road direction
        self.max_vehicles_per_cell = 5
        self.max_queue_length = 20
        self.light_timer_max = 60  # Maximum light duration in seconds
        self.emergency_override_time = 10
        
        # Action space: 9 discrete actions for traffic light control
        self.action_space = spaces.Discrete(9)
        
        # Observation space: Multi-component state representation
        # - 4 road grids (North, South, East, West): 4 * 5 * 5 = 100
        # - Queue lengths: 4 values
        # - Current light state: 3 values (one-hot)
        # - Light timer remaining: 1 value (normalized)
        # - Time of day: 1 value (normalized)
        # - Emergency vehicle present: 4 values (per direction)
        # Total: 100 + 4 + 3 + 1 + 1 + 4 = 113
        obs_space_size = (4 * self.grid_size * self.grid_size) + 4 + 3 + 1 + 1 + 4
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_space_size,), dtype=np.float32
        )
        
        # Initialize environment state
        self.render_mode = render_mode
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the traffic junction environment"""
        super().reset(seed=seed)
        
        # Initialize road grids (vehicles per cell)
        self.road_grids = {
            TrafficDirection.NORTH: np.zeros((self.grid_size, self.grid_size)),
            TrafficDirection.SOUTH: np.zeros((self.grid_size, self.grid_size)),
            TrafficDirection.EAST: np.zeros((self.grid_size, self.grid_size)),
            TrafficDirection.WEST: np.zeros((self.grid_size, self.grid_size))
        }
        
        # Initialize vehicle queues
        self.vehicle_queues = {
            TrafficDirection.NORTH: 0,
            TrafficDirection.SOUTH: 0,
            TrafficDirection.EAST: 0,
            TrafficDirection.WEST: 0
        }
        
        # Traffic light state
        self.current_light = TrafficLightState.NORTH_SOUTH_GREEN
        self.light_timer = 30  # Start with 30 seconds
        
        # Time simulation
        self.current_time = 8.0  # Start at 8 AM
        self.step_count = 0
        
        # Emergency vehicles
        self.emergency_vehicles = {
            TrafficDirection.NORTH: False,
            TrafficDirection.SOUTH: False,
            TrafficDirection.EAST: False,
            TrafficDirection.WEST: False
        }
        
        # Performance tracking
        self.total_waiting_time = 0
        self.vehicles_processed = 0
        self.total_reward = 0
        
        # Hidden state (for analysis)
        self.hidden_state = HiddenTrafficState.FREE_FLOW
        
        # Generate initial traffic
        self._generate_traffic()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int):
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update traffic dynamics
        self._update_traffic()
        
        # Update time (each step = 5 seconds of simulation time)
        self.current_time += 5/3600  # 5 seconds in hours
        if self.current_time >= 24:
            self.current_time -= 24
        
        # Update light timer
        self.light_timer = max(0, self.light_timer - 5)
        
        # Calculate total reward for this step
        step_reward = reward + self._calculate_flow_reward()
        self.total_reward += step_reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= 1000  # Max episode length
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, step_reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return immediate reward"""
        reward = 0
        
        if action == 0:  # Extend Green Time North-South
            if self.current_light == TrafficLightState.NORTH_SOUTH_GREEN:
                self.light_timer = min(self.light_timer_max, self.light_timer + 10)
                reward = 2  # Small positive reward for extending appropriate direction
            else:
                reward = -5  # Penalty for inappropriate action
                
        elif action == 1:  # Extend Green Time East-West
            if self.current_light == TrafficLightState.EAST_WEST_GREEN:
                self.light_timer = min(self.light_timer_max, self.light_timer + 10)
                reward = 2
            else:
                reward = -5
                
        elif action == 2:  # Switch to North-South Green
            if self.current_light != TrafficLightState.NORTH_SOUTH_GREEN:
                self.current_light = TrafficLightState.NORTH_SOUTH_GREEN
                self.light_timer = 30
                reward = self._calculate_switch_reward(TrafficDirection.NORTH, TrafficDirection.SOUTH)
            else:
                reward = -2  # Penalty for unnecessary switch
                
        elif action == 3:  # Switch to East-West Green
            if self.current_light != TrafficLightState.EAST_WEST_GREEN:
                self.current_light = TrafficLightState.EAST_WEST_GREEN
                self.light_timer = 30
                reward = self._calculate_switch_reward(TrafficDirection.EAST, TrafficDirection.WEST)
            else:
                reward = -2
                
        elif action == 4:  # Prioritize Emergency Lane
            emergency_present = any(self.emergency_vehicles.values())
            if emergency_present:
                self._handle_emergency_override()
                reward = 20  # High reward for emergency handling
            else:
                reward = -10  # Penalty for false emergency activation
                
        elif action == 5:  # All Red (Transition)
            self.current_light = TrafficLightState.ALL_RED
            self.light_timer = 5  # Short all-red phase
            reward = -1  # Small penalty for stopping all traffic
            
        elif action == 6:  # Reset Timer
            self.light_timer = 30  # Reset to standard timing
            reward = 0
            
        elif action == 7:  # Short Green Cycle
            self.light_timer = 15
            reward = 1
            
        elif action == 8:  # Extended Green Cycle
            self.light_timer = 60
            reward = -3 if self._is_direction_clear() else 3
        
        return reward
    
    def _update_traffic(self):
        """Update traffic flow and vehicle positions"""
        # Process vehicles through intersection based on current light
        if self.current_light == TrafficLightState.NORTH_SOUTH_GREEN:
            self._process_vehicles([TrafficDirection.NORTH, TrafficDirection.SOUTH])
        elif self.current_light == TrafficLightState.EAST_WEST_GREEN:
            self._process_vehicles([TrafficDirection.EAST, TrafficDirection.WEST])
        
        # Generate new traffic based on time patterns
        self._generate_traffic()
        
        # Update hidden traffic state
        self._update_hidden_state()
    
    def _process_vehicles(self, green_directions: List[TrafficDirection]):
        """Process vehicles for green light directions"""
        for direction in green_directions:
            if self.vehicle_queues[direction] > 0:
                # Each green light step processes 1-3 vehicles (realistic throughput)
                vehicles_to_process = min(
                    random.randint(1, 3),
                    self.vehicle_queues[direction]
                )
                self.vehicle_queues[direction] -= vehicles_to_process
                self.vehicles_processed += vehicles_to_process
    
    def _generate_traffic(self):
        """Generate new vehicles based on time-of-day patterns"""
        # Rush hour patterns for Rwanda
        hour = int(self.current_time)
        
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            arrival_rate = 0.8
        elif 12 <= hour <= 14:  # Lunch hour
            arrival_rate = 0.6
        elif 22 <= hour or hour <= 6:  # Night time
            arrival_rate = 0.2
        else:  # Regular hours
            arrival_rate = 0.4
        
        # Generate vehicles for each direction
        for direction in TrafficDirection:
            if random.random() < arrival_rate:
                self.vehicle_queues[direction] += random.randint(1, 2)
                self.vehicle_queues[direction] = min(
                    self.vehicle_queues[direction], 
                    self.max_queue_length
                )
        
        # Occasionally generate emergency vehicles
        if random.random() < 0.05:  # 5% chance per step
            emergency_direction = random.choice(list(TrafficDirection))
            self.emergency_vehicles[emergency_direction] = True
    
    def _calculate_flow_reward(self) -> float:
        """Calculate reward based on traffic flow efficiency"""
        total_queue = sum(self.vehicle_queues.values())
        
        # Reward for keeping queues short
        if total_queue == 0:
            return 10  # Excellent flow
        elif total_queue <= 5:
            return 5   # Good flow
        elif total_queue <= 10:
            return 0   # Acceptable flow
        elif total_queue <= 15:
            return -5  # Building congestion
        else:
            return -10 # Heavy congestion
    
    def _calculate_switch_reward(self, dir1: TrafficDirection, dir2: TrafficDirection) -> float:
        """Calculate reward for switching light direction"""
        target_queue = self.vehicle_queues[dir1] + self.vehicle_queues[dir2]
        other_dirs = [d for d in TrafficDirection if d not in [dir1, dir2]]
        other_queue = sum(self.vehicle_queues[d] for d in other_dirs)
        
        if target_queue > other_queue:
            return 5  # Good switch
        else:
            return -3  # Poor switch
    
    def _handle_emergency_override(self):
        """Handle emergency vehicle priority"""
        for direction, has_emergency in self.emergency_vehicles.items():
            if has_emergency:
                if direction in [TrafficDirection.NORTH, TrafficDirection.SOUTH]:
                    self.current_light = TrafficLightState.NORTH_SOUTH_GREEN
                else:
                    self.current_light = TrafficLightState.EAST_WEST_GREEN
                
                self.light_timer = self.emergency_override_time
                self.emergency_vehicles[direction] = False  # Clear emergency
                break
    
    def _is_direction_clear(self) -> bool:
        """Check if current green direction has low traffic"""
        if self.current_light == TrafficLightState.NORTH_SOUTH_GREEN:
            return (self.vehicle_queues[TrafficDirection.NORTH] + 
                   self.vehicle_queues[TrafficDirection.SOUTH]) <= 2
        elif self.current_light == TrafficLightState.EAST_WEST_GREEN:
            return (self.vehicle_queues[TrafficDirection.EAST] + 
                   self.vehicle_queues[TrafficDirection.WEST]) <= 2
        return True
    
    def _update_hidden_state(self):
        """Update the hidden traffic state for analysis"""
        total_queue = sum(self.vehicle_queues.values())
        hour = int(self.current_time)
        
        if any(self.emergency_vehicles.values()):
            self.hidden_state = HiddenTrafficState.INCIDENT_RELATED_DELAYS
        elif total_queue >= 15:
            self.hidden_state = HiddenTrafficState.PEAK_HOUR_GRIDLOCK
        elif total_queue >= 8:
            self.hidden_state = HiddenTrafficState.BUILDING_CONGESTION
        elif 22 <= hour or hour <= 6:
            self.hidden_state = HiddenTrafficState.OFF_PEAK_FLOW
        elif total_queue <= 3:
            self.hidden_state = HiddenTrafficState.FREE_FLOW
        else:
            self.hidden_state = HiddenTrafficState.RECOVERY_PHASE
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        obs = []
        
        # Road grid states (normalized)
        for direction in TrafficDirection:
            grid = self.road_grids[direction].flatten()
            obs.extend(grid / self.max_vehicles_per_cell)
        
        # Queue lengths (normalized)
        for direction in TrafficDirection:
            obs.append(self.vehicle_queues[direction] / self.max_queue_length)
        
        # Traffic light state (one-hot encoding)
        light_state = [0, 0, 0]
        light_state[self.current_light.value] = 1
        obs.extend(light_state)
        
        # Light timer (normalized)
        obs.append(self.light_timer / self.light_timer_max)
        
        # Time of day (normalized)
        obs.append(self.current_time / 24.0)
        
        # Emergency vehicles (binary)
        for direction in TrafficDirection:
            obs.append(1.0 if self.emergency_vehicles[direction] else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        return {
            'total_vehicles_waiting': sum(self.vehicle_queues.values()),
            'vehicles_processed': self.vehicles_processed,
            'current_light': self.current_light.name,
            'light_timer': self.light_timer,
            'hidden_state': self.hidden_state.name,
            'emergency_active': any(self.emergency_vehicles.values()),
            'time_of_day': f"{int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}",
            'step_count': self.step_count
        }
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated"""
        # Terminate if gridlock occurs for too long
        total_queue = sum(self.vehicle_queues.values())
        return total_queue >= self.max_queue_length * 4
    
    def render(self):
        """Render the environment (placeholder for now)"""
        if self.render_mode == "human":
            print(f"\n=== Traffic Junction Status ===")
            print(f"Time: {int(self.current_time):02d}:{int((self.current_time % 1) * 60):02d}")
            print(f"Light: {self.current_light.name} ({self.light_timer}s remaining)")
            print(f"Hidden State: {self.hidden_state.name}")
            print("\nQueues:")
            for direction in TrafficDirection:
                emergency = " [EMERGENCY]" if self.emergency_vehicles[direction] else ""
                print(f"  {direction.name}: {self.vehicle_queues[direction]} vehicles{emergency}")
            print(f"\nTotal Processed: {self.vehicles_processed}")
            print(f"Total Reward: {self.total_reward:.2f}")
    
    def close(self):
        """Clean up environment resources"""
        pass