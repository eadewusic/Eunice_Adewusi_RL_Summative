import random
from enum import Enum
from typing import Tuple
import pygame

class VehicleType(Enum):
    CAR = "car"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    EMERGENCY = "emergency"

class Vehicle:
    """
    Represents a vehicle in the traffic simulation
    """
    
    def __init__(self, vehicle_type: VehicleType, direction: int, spawn_time: float):
        self.type = vehicle_type
        self.direction = direction  # 0=North, 1=South, 2=East, 3=West
        self.spawn_time = spawn_time
        self.waiting_time = 0
        self.position = [0, 0]  # Grid position
        self.speed = self._get_base_speed()
        self.size = self._get_vehicle_size()
        self.color = self._get_vehicle_color()
        self.crossing_time = self._get_crossing_time()
        self.priority = self._get_priority()
        
    def _get_base_speed(self) -> float:
        """Get base speed based on vehicle type"""
        speed_map = {
            VehicleType.CAR: 1.0,
            VehicleType.BUS: 0.7,
            VehicleType.MOTORCYCLE: 1.5,
            VehicleType.TRUCK: 0.5,
            VehicleType.EMERGENCY: 1.2
        }
        return speed_map[self.type]
    
    def _get_vehicle_size(self) -> Tuple[int, int]:
        """Get vehicle size for rendering (width, height)"""
        size_map = {
            VehicleType.CAR: (20, 15),
            VehicleType.BUS: (25, 40),
            VehicleType.MOTORCYCLE: (12, 8),
            VehicleType.TRUCK: (22, 45),
            VehicleType.EMERGENCY: (20, 18)
        }
        return size_map[self.type]
    
    def _get_vehicle_color(self) -> Tuple[int, int, int]:
        """Get vehicle color for rendering (RGB)"""
        if self.type == VehicleType.CAR:
            colors = [
                (255, 0, 0),    # Red
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (128, 128, 128), # Gray
                (255, 255, 255)  # White
            ]
            return random.choice(colors)
        elif self.type == VehicleType.BUS:
            return (0, 128, 255)  # Light blue
        elif self.type == VehicleType.MOTORCYCLE:
            return (255, 165, 0)  # Orange
        elif self.type == VehicleType.TRUCK:
            return (139, 69, 19)  # Brown
        elif self.type == VehicleType.EMERGENCY:
            return (255, 0, 0)   # Red
    
    def _get_crossing_time(self) -> int:
        """Get time needed to cross intersection (in simulation steps)"""
        crossing_times = {
            VehicleType.CAR: 4,
            VehicleType.BUS: 8,
            VehicleType.MOTORCYCLE: 2,
            VehicleType.TRUCK: 10,
            VehicleType.EMERGENCY: 3
        }
        return crossing_times[self.type]
    
    def _get_priority(self) -> int:
        """Get vehicle priority (higher = more important)"""
        priority_map = {
            VehicleType.CAR: 1,
            VehicleType.BUS: 3,  # Higher priority for public transport
            VehicleType.MOTORCYCLE: 1,
            VehicleType.TRUCK: 2,
            VehicleType.EMERGENCY: 10  # Highest priority
        }
        return priority_map[self.type]
    
    def update_waiting_time(self, current_time: float):
        """Update how long this vehicle has been waiting"""
        self.waiting_time = current_time - self.spawn_time
    
    def is_emergency(self) -> bool:
        """Check if this is an emergency vehicle"""
        return self.type == VehicleType.EMERGENCY
    
    def get_fuel_consumption_rate(self) -> float:
        """Get fuel consumption rate while idling (liters per minute)"""
        consumption_map = {
            VehicleType.CAR: 0.5,
            VehicleType.BUS: 2.0,
            VehicleType.MOTORCYCLE: 0.2,
            VehicleType.TRUCK: 1.5,
            VehicleType.EMERGENCY: 0.7
        }
        return consumption_map[self.type]
    
    def get_passenger_count(self) -> int:
        """Estimate passenger count for economic impact calculation"""
        passenger_map = {
            VehicleType.CAR: random.randint(1, 4),
            VehicleType.BUS: random.randint(15, 50),
            VehicleType.MOTORCYCLE: random.randint(1, 2),
            VehicleType.TRUCK: 1,  # Just driver
            VehicleType.EMERGENCY: random.randint(2, 4)
        }
        return passenger_map[self.type]
    
    def draw(self, screen, x: int, y: int, rotation: int = 0):
        """Draw the vehicle on the pygame screen"""
        width, height = self.size
        
        # Create vehicle rectangle
        vehicle_rect = pygame.Rect(0, 0, width, height)
        vehicle_surface = pygame.Surface((width, height))
        vehicle_surface.fill(self.color)
        
        # Add special markings for specific vehicle types
        if self.type == VehicleType.EMERGENCY:
            # Add emergency flashers
            pygame.draw.circle(vehicle_surface, (255, 255, 0), (5, 5), 3)
            pygame.draw.circle(vehicle_surface, (255, 255, 0), (width-5, 5), 3)
        elif self.type == VehicleType.BUS:
            # Add windows
            for i in range(3):
                window_y = 5 + i * 10
                pygame.draw.rect(vehicle_surface, (200, 200, 255), 
                               (3, window_y, width-6, 8))
        
        # Rotate if needed
        if rotation != 0:
            vehicle_surface = pygame.transform.rotate(vehicle_surface, rotation)
            vehicle_rect = vehicle_surface.get_rect()
        
        # Center the vehicle at the given position
        vehicle_rect.center = (x, y)
        screen.blit(vehicle_surface, vehicle_rect)
    
    def __repr__(self):
        return f"Vehicle({self.type.value}, dir={self.direction}, wait={self.waiting_time:.1f}s)"

class VehicleGenerator:
    """
    Generates vehicles based on realistic traffic patterns
    """
    
    def __init__(self):
        # Vehicle type probabilities for Rwanda traffic
        self.vehicle_probabilities = {
            VehicleType.CAR: 0.65,          # 65% cars
            VehicleType.BUS: 0.15,          # 15% buses (popular public transport)
            VehicleType.MOTORCYCLE: 0.18,   # 18% motorcycles (common in Rwanda)
            VehicleType.TRUCK: 0.015,       # 1.5% trucks
            VehicleType.EMERGENCY: 0.005    # 0.5% emergency vehicles
        }
        
        # Rush hour directional preferences
        self.rush_hour_patterns = {
            'morning': {  # 7-9 AM: Residential to business areas
                0: 0.4,  # North (from residential)
                1: 0.1,  # South (to residential)
                2: 0.3,  # East (from suburbs)
                3: 0.2   # West (mixed)
            },
            'evening': {  # 5-7 PM: Business to residential areas
                0: 0.1,  # North (to residential)
                1: 0.4,  # South (from business)
                2: 0.2,  # East (mixed)
                3: 0.3   # West (to suburbs)
            },
            'normal': {   # Regular hours
                0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25
            }
        }
    
    def generate_vehicle(self, current_time: float, direction: int = None) -> Vehicle:
        """Generate a new vehicle based on current time and traffic patterns"""
        
        # Determine vehicle type
        rand = random.random()
        cumulative_prob = 0
        vehicle_type = VehicleType.CAR  # Default
        
        for v_type, prob in self.vehicle_probabilities.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                vehicle_type = v_type
                break
        
        # Determine direction if not specified
        if direction is None:
            hour = int(current_time) % 24
            
            if 7 <= hour <= 9:
                pattern = self.rush_hour_patterns['morning']
            elif 17 <= hour <= 19:
                pattern = self.rush_hour_patterns['evening']
            else:
                pattern = self.rush_hour_patterns['normal']
            
            # Select direction based on pattern
            rand = random.random()
            cumulative_prob = 0
            direction = 0  # Default North
            
            for dir_num, prob in pattern.items():
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    direction = dir_num
                    break
        
        return Vehicle(vehicle_type, direction, current_time)
    
    def get_arrival_rate(self, current_time: float) -> float:
        """Get vehicle arrival rate based on time of day"""
        hour = int(current_time) % 24
        minute = int((current_time % 1) * 60)
        
        # Base rates per hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_rate = 0.8
        elif 12 <= hour <= 14:  # Lunch hour
            base_rate = 0.6
        elif 10 <= hour <= 16:  # Business hours
            base_rate = 0.5
        elif 20 <= hour <= 22:  # Evening
            base_rate = 0.4
        elif 6 <= hour <= 7 or 19 <= hour <= 20:  # Transition periods
            base_rate = 0.3
        else:  # Late night/early morning
            base_rate = 0.1
        
        # Add some randomness
        variation = random.uniform(0.8, 1.2)
        return base_rate * variation
    
    def should_generate_emergency(self, current_time: float) -> bool:
        """Determine if an emergency vehicle should be generated"""
        # Higher probability during rush hours
        hour = int(current_time) % 24
        
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            emergency_rate = 0.02  # 2% chance during rush hour
        else:
            emergency_rate = 0.005  # 0.5% chance normally
        
        return random.random() < emergency_rate