import pygame
import math
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np

from .traffic_junction_env import TrafficJunctionEnv, TrafficDirection, TrafficLightState, HiddenTrafficState
from .vehicle import Vehicle, VehicleGenerator, VehicleType

class TrafficVisualizer:
    """
    Pygame-based visualization for the traffic junction environment
    """
    
    def __init__(self, env: TrafficJunctionEnv, width: int = 1000, height: int = 800):
        """
        Initialize the traffic visualizer
        
        Args:
            env: The traffic junction environment
            width: Window width
            height: Window height
        """
        self.env = env
        self.width = width
        self.height = height
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Rwanda Traffic Junction - RL Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        # Colors
        self.colors = {
            'background': (34, 139, 34),      # Forest green (Rwanda's hills)
            'road': (64, 64, 64),             # Dark gray
            'lane_marking': (255, 255, 255),  # White
            'intersection': (96, 96, 96),     # Light gray
            'grass': (34, 139, 34),           # Green
            'building': (139, 69, 19),        # Brown
            'text': (255, 255, 255),          # White
            'red_light': (255, 0, 0),         # Red
            'green_light': (0, 255, 0),       # Green
            'yellow_light': (255, 255, 0),    # Yellow
            'light_off': (64, 64, 64),        # Dark gray
        }
        
        # Layout calculations
        self.intersection_size = 200
        self.road_width = 120
        self.lane_width = 30
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Vehicle management
        self.vehicle_generator = VehicleGenerator()
        self.active_vehicles = {
            TrafficDirection.NORTH: [],
            TrafficDirection.SOUTH: [],
            TrafficDirection.EAST: [],
            TrafficDirection.WEST: []
        }
        
        # Performance tracking for visualization
        self.performance_history = []
        self.max_history_length = 100
        
        # Animation state
        self.vehicle_id_counter = 0
        
    def render(self, action_taken: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Render the current state of the traffic junction
        
        Args:
            action_taken: The action taken by the agent (for display)
            
        Returns:
            RGB array if render_mode is 'rgb_array', else None
        """
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw environment components
        self._draw_roads()
        self._draw_intersection()
        self._draw_traffic_lights()
        self._draw_vehicles()
        self._draw_queue_indicators()
        self._draw_performance_panel(action_taken)
        self._draw_info_panel()
        
        # Update display
        if self.env.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
            
        elif self.env.render_mode == "rgb_array":
            # Return RGB array for recording
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
        
        return None
    
    def _draw_roads(self):
        """Draw the road network"""
        # Vertical road (North-South)
        road_rect_vertical = pygame.Rect(
            self.center_x - self.road_width // 2,
            0,
            self.road_width,
            self.height
        )
        pygame.draw.rect(self.screen, self.colors['road'], road_rect_vertical)
        
        # Horizontal road (East-West) 
        road_rect_horizontal = pygame.Rect(
            0,
            self.center_y - self.road_width // 2,
            self.width,
            self.road_width
        )
        pygame.draw.rect(self.screen, self.colors['road'], road_rect_horizontal)
        
        # Lane markings - Vertical road
        for i in range(1, 4):  # 3 lane dividers for 4 lanes
            x = self.center_x - self.road_width // 2 + i * (self.road_width // 4)
            self._draw_dashed_line(
                (x, 0), 
                (x, self.center_y - self.intersection_size // 2),
                self.colors['lane_marking']
            )
            self._draw_dashed_line(
                (x, self.center_y + self.intersection_size // 2), 
                (x, self.height),
                self.colors['lane_marking']
            )
        
        # Lane markings - Horizontal road
        for i in range(1, 4):
            y = self.center_y - self.road_width // 2 + i * (self.road_width // 4)
            self._draw_dashed_line(
                (0, y), 
                (self.center_x - self.intersection_size // 2, y),
                self.colors['lane_marking']
            )
            self._draw_dashed_line(
                (self.center_x + self.intersection_size // 2, y), 
                (self.width, y),
                self.colors['lane_marking']
            )
    
    def _draw_intersection(self):
        """Draw the central intersection"""
        intersection_rect = pygame.Rect(
            self.center_x - self.intersection_size // 2,
            self.center_y - self.intersection_size // 2,
            self.intersection_size,
            self.intersection_size
        )
        pygame.draw.rect(self.screen, self.colors['intersection'], intersection_rect)
        
        # Draw intersection border
        pygame.draw.rect(self.screen, self.colors['lane_marking'], intersection_rect, 2)
    
    def _draw_traffic_lights(self):
        """Draw traffic lights at each corner of the intersection"""
        light_positions = [
            (self.center_x - 120, self.center_y - 120),  # Northwest
            (self.center_x + 120, self.center_y - 120),  # Northeast  
            (self.center_x + 120, self.center_y + 120),  # Southeast
            (self.center_x - 120, self.center_y + 120),  # Southwest
        ]
        
        for pos in light_positions:
            self._draw_traffic_light_pole(pos)
    
    def _draw_traffic_light_pole(self, position: Tuple[int, int]):
        """Draw a single traffic light pole"""
        x, y = position
        
        # Pole
        pygame.draw.rect(self.screen, (100, 100, 100), (x-3, y, 6, 80))
        
        # Light housing
        light_rect = pygame.Rect(x-15, y-20, 30, 60)
        pygame.draw.rect(self.screen, (50, 50, 50), light_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), light_rect, 2)
        
        # Light states based on current traffic light
        if self.env.current_light == TrafficLightState.NORTH_SOUTH_GREEN:
            # North-South is green, East-West is red
            if position[0] < self.center_x:  # West side
                if position[1] < self.center_y:  # Northwest - Red
                    self._draw_light_circle((x, y-10), self.colors['red_light'])
                    self._draw_light_circle((x, y+5), self.colors['light_off'])
                else:  # Southwest - Red
                    self._draw_light_circle((x, y-10), self.colors['red_light'])
                    self._draw_light_circle((x, y+5), self.colors['light_off'])
            else:  # East side
                if position[1] < self.center_y:  # Northeast - Red
                    self._draw_light_circle((x, y-10), self.colors['red_light'])
                    self._draw_light_circle((x, y+5), self.colors['light_off'])
                else:  # Southeast - Red
                    self._draw_light_circle((x, y-10), self.colors['red_light'])
                    self._draw_light_circle((x, y+5), self.colors['light_off'])
                    
        elif self.env.current_light == TrafficLightState.EAST_WEST_GREEN:
            # East-West is green, North-South is red
            if position[1] < self.center_y:  # North side
                self._draw_light_circle((x, y-10), self.colors['red_light'])
                self._draw_light_circle((x, y+5), self.colors['light_off'])
            else:  # South side
                self._draw_light_circle((x, y-10), self.colors['red_light'])
                self._draw_light_circle((x, y+5), self.colors['light_off'])
                
        else:  # ALL_RED
            self._draw_light_circle((x, y-10), self.colors['red_light'])
            self._draw_light_circle((x, y+5), self.colors['light_off'])
    
    def _draw_light_circle(self, position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a single light circle"""
        pygame.draw.circle(self.screen, color, position, 8)
        pygame.draw.circle(self.screen, (200, 200, 200), position, 8, 2)
    
    def _draw_vehicles(self):
        """Draw all vehicles in the scene"""
        # Update vehicle positions based on environment state
        self._update_vehicle_positions()
        
        # Draw vehicles for each direction
        for direction, vehicles in self.active_vehicles.items():
            for i, vehicle in enumerate(vehicles):
                self._draw_vehicle(vehicle, direction, i)
    
    def _update_vehicle_positions(self):
        """Update vehicle positions based on current environment state"""
        # Clear existing vehicles
        for direction in TrafficDirection:
            self.active_vehicles[direction] = []
        
        # Generate vehicles based on queue lengths
        for direction in TrafficDirection:
            queue_length = self.env.vehicle_queues[direction]
            
            # Create vehicles for visualization
            for i in range(min(queue_length, 10)):  # Limit to 10 visible vehicles per direction
                vehicle = self.vehicle_generator.generate_vehicle(
                    self.env.current_time, 
                    direction.value
                )
                self.active_vehicles[direction].append(vehicle)
    
    def _draw_vehicle(self, vehicle: Vehicle, direction: TrafficDirection, queue_position: int):
        """Draw a single vehicle"""
        # Calculate position based on direction and queue position
        spacing = 35  # Space between vehicles
        
        if direction == TrafficDirection.NORTH:
            x = self.center_x - 15
            y = self.center_y - self.intersection_size // 2 - 50 - (queue_position * spacing)
            rotation = 0
        elif direction == TrafficDirection.SOUTH:
            x = self.center_x + 15
            y = self.center_y + self.intersection_size // 2 + 50 + (queue_position * spacing)
            rotation = 180
        elif direction == TrafficDirection.EAST:
            x = self.center_x + self.intersection_size // 2 + 50 + (queue_position * spacing)
            y = self.center_y - 15
            rotation = 270
        else:  # WEST
            x = self.center_x - self.intersection_size // 2 - 50 - (queue_position * spacing)
            y = self.center_y + 15
            rotation = 90
        
        # Don't draw vehicles outside screen bounds
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return
            
        # Draw vehicle
        vehicle.draw(self.screen, x, y, rotation)
        
        # Draw emergency flashers if emergency vehicle
        if vehicle.type == VehicleType.EMERGENCY:
            flash_color = self.colors['red_light'] if (pygame.time.get_ticks() // 200) % 2 else self.colors['yellow_light']
            pygame.draw.circle(self.screen, flash_color, (x-8, y-8), 4)
            pygame.draw.circle(self.screen, flash_color, (x+8, y-8), 4)
    
    def _draw_queue_indicators(self):
        """Draw queue length indicators"""
        indicator_positions = {
            TrafficDirection.NORTH: (self.center_x - 60, 50),
            TrafficDirection.SOUTH: (self.center_x + 60, self.height - 50),
            TrafficDirection.EAST: (self.width - 100, self.center_y - 30),
            TrafficDirection.WEST: (100, self.center_y + 30)
        }
        
        for direction, pos in indicator_positions.items():
            queue_length = self.env.vehicle_queues[direction]
            
            # Draw queue length bar
            bar_width = 60
            bar_height = 10
            max_queue = 20
            
            # Background bar
            bg_rect = pygame.Rect(pos[0] - bar_width//2, pos[1], bar_width, bar_height)
            pygame.draw.rect(self.screen, (100, 100, 100), bg_rect)
            
            # Queue level bar
            if queue_length > 0:
                fill_width = min(bar_width, (queue_length / max_queue) * bar_width)
                color = self._get_queue_color(queue_length, max_queue)
                fill_rect = pygame.Rect(pos[0] - bar_width//2, pos[1], fill_width, bar_height)
                pygame.draw.rect(self.screen, color, fill_rect)
            
            # Queue length text
            text = self.font.render(f"{direction.name}: {queue_length}", True, self.colors['text'])
            text_rect = text.get_rect()
            text_rect.center = (pos[0], pos[1] - 15)
            self.screen.blit(text, text_rect)
    
    def _get_queue_color(self, queue_length: int, max_queue: int) -> Tuple[int, int, int]:
        """Get color based on queue length severity"""
        ratio = queue_length / max_queue
        if ratio < 0.3:
            return (0, 255, 0)    # Green - good
        elif ratio < 0.6:
            return (255, 255, 0)  # Yellow - moderate
        elif ratio < 0.8:
            return (255, 165, 0)  # Orange - concerning
        else:
            return (255, 0, 0)    # Red - critical
    
    def _draw_performance_panel(self, action_taken: Optional[int] = None):
        """Draw performance metrics panel"""
        panel_rect = pygame.Rect(20, 20, 300, 200)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.screen, self.colors['text'], panel_rect, 2)
        
        y_offset = 35
        line_height = 22
        
        # Title
        title = self.large_font.render("Performance Metrics", True, self.colors['text'])
        self.screen.blit(title, (30, y_offset))
        y_offset += 35
        
        # Current metrics
        metrics = [
            f"Time: {int(self.env.current_time):02d}:{int((self.env.current_time % 1) * 60):02d}",
            f"Light: {self.env.current_light.name}",
            f"Timer: {self.env.light_timer}s",
            f"Total Waiting: {sum(self.env.vehicle_queues.values())}",
            f"Processed: {self.env.vehicles_processed}",
            f"Reward: {self.env.total_reward:.1f}",
            f"Hidden State: {self.env.hidden_state.name}"
        ]
        
        if action_taken is not None:
            action_names = [
                "Extend NS Green", "Extend EW Green", "Switch to NS", "Switch to EW",
                "Emergency Priority", "All Red", "Reset Timer", "Short Cycle", "Long Cycle"
            ]
            if 0 <= action_taken < len(action_names):
                metrics.append(f"Action: {action_names[action_taken]}")
        
        for metric in metrics:
            text = self.font.render(metric, True, self.colors['text'])
            self.screen.blit(text, (30, y_offset))
            y_offset += line_height
    
    def _draw_info_panel(self):
        """Draw additional information panel"""
        panel_rect = pygame.Rect(self.width - 320, 20, 300, 150)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.screen, self.colors['text'], panel_rect, 2)
        
        y_offset = 35
        line_height = 20
        
        # Title
        title = self.large_font.render("Traffic Status", True, self.colors['text'])
        self.screen.blit(title, (self.width - 310, y_offset))
        y_offset += 30
        
        # Emergency status
        emergency_active = any(self.env.emergency_vehicles.values())
        emergency_text = "EMERGENCY ACTIVE" if emergency_active else "Normal Operation"
        emergency_color = self.colors['red_light'] if emergency_active else self.colors['green_light']
        text = self.font.render(emergency_text, True, emergency_color)
        self.screen.blit(text, (self.width - 310, y_offset))
        y_offset += line_height + 10
        
        # Traffic pattern info
        hour = int(self.env.current_time)
        if 7 <= hour <= 9:
            pattern = "Morning Rush"
        elif 17 <= hour <= 19:
            pattern = "Evening Rush"
        elif 12 <= hour <= 14:
            pattern = "Lunch Hour"
        else:
            pattern = "Normal Traffic"
        
        pattern_text = self.font.render(f"Pattern: {pattern}", True, self.colors['text'])
        self.screen.blit(pattern_text, (self.width - 310, y_offset))
        y_offset += line_height
        
        # Efficiency indicator
        total_vehicles = sum(self.env.vehicle_queues.values())
        if total_vehicles == 0:
            efficiency = "Excellent"
            eff_color = self.colors['green_light']
        elif total_vehicles <= 5:
            efficiency = "Good"
            eff_color = self.colors['green_light']
        elif total_vehicles <= 10:
            efficiency = "Fair"
            eff_color = self.colors['yellow_light']
        else:
            efficiency = "Poor"
            eff_color = self.colors['red_light']
        
        eff_text = self.font.render(f"Flow: {efficiency}", True, eff_color)
        self.screen.blit(eff_text, (self.width - 310, y_offset))
    
    def _draw_dashed_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a dashed line"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate line length and direction
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length == 0:
            return
        
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        # Draw dashed segments
        dash_length = 10
        gap_length = 5
        current_length = 0
        
        while current_length < length:
            # Start of dash
            start_x = x1 + current_length * dx
            start_y = y1 + current_length * dy
            
            # End of dash
            end_length = min(current_length + dash_length, length)
            end_x = x1 + end_length * dx
            end_y = y1 + end_length * dy
            
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)
            
            current_length += dash_length + gap_length
    
    def handle_events(self):
        """Handle pygame events"""
        events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            events.append(event)
        return True
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()