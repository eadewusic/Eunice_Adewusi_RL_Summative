"""
Traffic Patterns for Rwanda Traffic Junction Environment

This module contains realistic traffic patterns based on Rwandan urban traffic
characteristics, including rush hour patterns, special events, and seasonal variations.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta

class DayType(Enum):
    WEEKDAY = "weekday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"
    HOLIDAY = "holiday"

class WeatherCondition(Enum):
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"

class SpecialEvent(Enum):
    NONE = "none"
    MARKET_DAY = "market_day"
    SCHOOL_EVENT = "school_event"
    SPORTS_EVENT = "sports_event"
    GOVERNMENT_MEETING = "government_meeting"
    RELIGIOUS_SERVICE = "religious_service"

class RwandaTrafficPatterns:
    """
    Realistic traffic patterns for Rwanda based on local traffic characteristics
    """
    
    def __init__(self):
        """Initialize Rwanda traffic patterns"""
        
        # Base traffic intensity by hour (0-23) for weekdays
        self.weekday_base_pattern = {
            0: 0.05,   1: 0.03,   2: 0.02,   3: 0.02,   4: 0.03,   5: 0.08,
            6: 0.25,   7: 0.85,   8: 0.95,   9: 0.65,   10: 0.45,  11: 0.55,
            12: 0.75,  13: 0.70,  14: 0.50,  15: 0.60,  16: 0.70,  17: 0.90,
            18: 0.95,  19: 0.80,  20: 0.55,  21: 0.35,  22: 0.20,  23: 0.10
        }
        
        # Weekend patterns (Saturday)
        self.saturday_base_pattern = {
            0: 0.02,   1: 0.01,   2: 0.01,   3: 0.01,   4: 0.01,   5: 0.02,
            6: 0.05,   7: 0.15,   8: 0.35,   9: 0.55,   10: 0.70,  11: 0.75,
            12: 0.80,  13: 0.85,  14: 0.90,  15: 0.85,  16: 0.75,  17: 0.60,
            18: 0.45,  19: 0.35,  20: 0.25,  21: 0.15,  22: 0.10,  23: 0.05
        }
        
        # Sunday patterns (quieter, with church traffic)
        self.sunday_base_pattern = {
            0: 0.01,   1: 0.01,   2: 0.01,   3: 0.01,   4: 0.01,   5: 0.02,
            6: 0.05,   7: 0.20,   8: 0.45,   9: 0.60,   10: 0.35,  11: 0.20,
            12: 0.40,  13: 0.50,  14: 0.55,  15: 0.45,  16: 0.40,  17: 0.35,
            18: 0.25,  19: 0.20,  20: 0.15,  21: 0.10,  22: 0.05,  23: 0.02
        }
        
        # Directional flow patterns (percentage going each direction)
        # Based on typical residential-to-business flow in Kigali
        self.directional_patterns = {
            'morning_rush': {  # 7-9 AM: Home to work
                'north': 0.35,   # From residential areas
                'south': 0.15,   # To residential areas
                'east': 0.30,    # From suburbs
                'west': 0.20     # Mixed flow
            },
            'lunch_hour': {    # 12-2 PM: Mixed movement
                'north': 0.25, 'south': 0.25, 'east': 0.25, 'west': 0.25
            },
            'evening_rush': {  # 5-7 PM: Work to home
                'north': 0.15,   # To residential areas
                'south': 0.35,   # From business areas
                'east': 0.20,    # Mixed flow
                'west': 0.30     # To suburbs
            },
            'normal': {        # Other times
                'north': 0.25, 'south': 0.25, 'east': 0.25, 'west': 0.25
            }
        }
        
        # Vehicle type probabilities (Rwanda-specific)
        self.vehicle_distribution = {
            'weekday': {
                'car': 0.60,
                'bus': 0.18,        # High public transport usage
                'motorcycle': 0.20,  # Very popular in Rwanda
                'truck': 0.015,
                'emergency': 0.005
            },
            'weekend': {
                'car': 0.70,
                'bus': 0.12,
                'motorcycle': 0.16,
                'truck': 0.01,
                'emergency': 0.01
            }
        }
        
        # Weather impact multipliers
        self.weather_multipliers = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.LIGHT_RAIN: 0.85,     # Slight reduction
            WeatherCondition.HEAVY_RAIN: 0.60,     # Significant reduction
            WeatherCondition.FOG: 0.75             # Moderate reduction
        }
        
        # Special event multipliers
        self.event_multipliers = {
            SpecialEvent.NONE: 1.0,
            SpecialEvent.MARKET_DAY: 1.5,          # Thursday/Saturday markets
            SpecialEvent.SCHOOL_EVENT: 1.3,        # School events increase traffic
            SpecialEvent.SPORTS_EVENT: 2.0,        # Major events significantly increase
            SpecialEvent.GOVERNMENT_MEETING: 1.2,  # Official events
            SpecialEvent.RELIGIOUS_SERVICE: 1.4    # Sunday services, special events
        }
    
    def get_traffic_intensity(self, 
                             hour: int, 
                             day_type: DayType = DayType.WEEKDAY,
                             weather: WeatherCondition = WeatherCondition.CLEAR,
                             special_event: SpecialEvent = SpecialEvent.NONE) -> float:
        """
        Get traffic intensity for specific conditions
        
        Args:
            hour: Hour of day (0-23)
            day_type: Type of day
            weather: Weather condition
            special_event: Any special events
            
        Returns:
            Traffic intensity (0.0 to 1.0+)
        """
        
        # Get base pattern
        if day_type == DayType.WEEKDAY:
            base_intensity = self.weekday_base_pattern.get(hour, 0.1)
        elif day_type == DayType.SATURDAY:
            base_intensity = self.saturday_base_pattern.get(hour, 0.1)
        elif day_type == DayType.SUNDAY:
            base_intensity = self.sunday_base_pattern.get(hour, 0.1)
        else:  # Holiday
            base_intensity = self.sunday_base_pattern.get(hour, 0.1) * 0.7
        
        # Apply weather modifier
        weather_modifier = self.weather_multipliers[weather]
        
        # Apply event modifier
        event_modifier = self.event_multipliers[special_event]
        
        # Calculate final intensity
        final_intensity = base_intensity * weather_modifier * event_modifier
        
        # Add some random variation (±15%)
        variation = random.uniform(0.85, 1.15)
        final_intensity *= variation
        
        return max(0.0, min(2.0, final_intensity))  # Clamp between 0 and 2
    
    def get_directional_flow(self, hour: int, day_type: DayType = DayType.WEEKDAY) -> Dict[str, float]:
        """
        Get directional traffic flow distribution
        
        Args:
            hour: Hour of day (0-23)
            day_type: Type of day
            
        Returns:
            Dictionary with flow percentages for each direction
        """
        
        if day_type in [DayType.SATURDAY, DayType.SUNDAY]:
            # Weekend traffic is more distributed
            return self.directional_patterns['normal']
        
        # Weekday patterns
        if 7 <= hour <= 9:
            return self.directional_patterns['morning_rush']
        elif 12 <= hour <= 14:
            return self.directional_patterns['lunch_hour']
        elif 17 <= hour <= 19:
            return self.directional_patterns['evening_rush']
        else:
            return self.directional_patterns['normal']
    
    def get_vehicle_distribution(self, day_type: DayType = DayType.WEEKDAY) -> Dict[str, float]:
        """
        Get vehicle type distribution
        
        Args:
            day_type: Type of day
            
        Returns:
            Dictionary with probabilities for each vehicle type
        """
        
        if day_type == DayType.WEEKDAY:
            return self.vehicle_distribution['weekday'].copy()
        else:
            return self.vehicle_distribution['weekend'].copy()
    
    def get_emergency_probability(self, 
                                hour: int, 
                                weather: WeatherCondition = WeatherCondition.CLEAR,
                                traffic_intensity: float = 0.5) -> float:
        """
        Get probability of emergency vehicle appearance
        
        Args:
            hour: Hour of day
            weather: Weather condition
            traffic_intensity: Current traffic intensity
            
        Returns:
            Probability of emergency vehicle (per step)
        """
        
        # Base emergency rate (per 1000 steps)
        base_rate = 0.001
        
        # Higher probability during rush hours
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            time_multiplier = 2.0
        elif 22 <= hour or hour <= 6:
            time_multiplier = 0.5  # Lower at night
        else:
            time_multiplier = 1.0
        
        # Weather increases emergency probability
        weather_multiplier = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.LIGHT_RAIN: 1.3,
            WeatherCondition.HEAVY_RAIN: 2.0,
            WeatherCondition.FOG: 1.5
        }[weather]
        
        # Higher traffic increases emergency probability
        traffic_multiplier = 1.0 + traffic_intensity
        
        return base_rate * time_multiplier * weather_multiplier * traffic_multiplier
    
    def generate_daily_schedule(self, day_type: DayType = DayType.WEEKDAY) -> List[Dict]:
        """
        Generate a full day's traffic schedule
        
        Args:
            day_type: Type of day
            
        Returns:
            List of hourly traffic parameters
        """
        
        schedule = []
        
        # Simulate weather for the day
        weather_prob = random.random()
        if weather_prob < 0.7:
            weather = WeatherCondition.CLEAR
        elif weather_prob < 0.85:
            weather = WeatherCondition.LIGHT_RAIN
        elif weather_prob < 0.95:
            weather = WeatherCondition.FOG
        else:
            weather = WeatherCondition.HEAVY_RAIN
        
        # Determine special events
        special_event = SpecialEvent.NONE
        if day_type == DayType.SATURDAY and random.random() < 0.3:
            special_event = SpecialEvent.MARKET_DAY
        elif day_type == DayType.SUNDAY and random.random() < 0.4:
            special_event = SpecialEvent.RELIGIOUS_SERVICE
        elif random.random() < 0.05:  # 5% chance of other events
            special_event = random.choice([
                SpecialEvent.SCHOOL_EVENT,
                SpecialEvent.SPORTS_EVENT,
                SpecialEvent.GOVERNMENT_MEETING
            ])
        
        for hour in range(24):
            intensity = self.get_traffic_intensity(hour, day_type, weather, special_event)
            directions = self.get_directional_flow(hour, day_type)
            vehicles = self.get_vehicle_distribution(day_type)
            emergency_prob = self.get_emergency_probability(hour, weather, intensity)
            
            schedule.append({
                'hour': hour,
                'intensity': intensity,
                'directions': directions,
                'vehicles': vehicles,
                'emergency_prob': emergency_prob,
                'weather': weather.value,
                'special_event': special_event.value
            })
        
        return schedule
    
    def get_rush_hour_challenges(self) -> Dict[str, any]:
        """
        Get specific challenges during rush hours
        
        Returns:
            Dictionary of rush hour challenge parameters
        """
        return {
            'increased_accidents': 1.5,        # 50% more accident probability
            'impatient_drivers': 1.3,          # 30% more aggressive behavior
            'bus_priority_requests': 2.0,      # 100% more bus priority needs
            'pedestrian_crossings': 1.4,       # 40% more pedestrian activity
            'parking_conflicts': 1.6,          # 60% more parking-related delays
            'fuel_shortage_impact': 1.2        # 20% impact from fuel availability
        }
    
    def simulate_real_world_disruptions(self, current_intensity: float) -> Dict[str, any]:
        """
        Simulate real-world traffic disruptions
        
        Args:
            current_intensity: Current traffic intensity
            
        Returns:
            Dictionary of disruption effects
        """
        disruptions = {
            'power_outage': False,
            'road_construction': False,
            'accident': False,
            'fuel_shortage': False,
            'protest_march': False
        }
        
        # Power outage probability (higher during peak hours)
        if current_intensity > 0.7 and random.random() < 0.02:
            disruptions['power_outage'] = True
        
        # Road construction (more likely on weekends)
        if random.random() < 0.01:
            disruptions['road_construction'] = True
        
        # Accidents (probability increases with traffic intensity)
        accident_prob = 0.005 * (1 + current_intensity)
        if random.random() < accident_prob:
            disruptions['accident'] = True
        
        # Fuel shortage effects (periodic in Rwanda)
        if random.random() < 0.005:
            disruptions['fuel_shortage'] = True
        
        # Protest or march (rare but impactful)
        if random.random() < 0.001:
            disruptions['protest_march'] = True
        
        return disruptions

# Utility functions for easy access
def get_current_traffic_pattern(hour: float, day_type: str = "weekday") -> Dict:
    """
    Get current traffic pattern for easy integration
    
    Args:
        hour: Current hour (can be float for minutes)
        day_type: Type of day ("weekday", "saturday", "sunday")
        
    Returns:
        Current traffic parameters
    """
    patterns = RwandaTrafficPatterns()
    
    day_enum = {
        "weekday": DayType.WEEKDAY,
        "saturday": DayType.SATURDAY,
        "sunday": DayType.SUNDAY
    }.get(day_type, DayType.WEEKDAY)
    
    hour_int = int(hour)
    
    return {
        'intensity': patterns.get_traffic_intensity(hour_int, day_enum),
        'directions': patterns.get_directional_flow(hour_int, day_enum),
        'vehicles': patterns.get_vehicle_distribution(day_enum),
        'emergency_prob': patterns.get_emergency_probability(hour_int)
    }

def is_rush_hour(hour: float) -> bool:
    """Check if current time is rush hour"""
    return (7 <= hour <= 9) or (17 <= hour <= 19)

def get_traffic_scenario_name(hour: float) -> str:
    """Get descriptive name for current traffic scenario"""
    if 6 <= hour <= 9:
        return "Morning Rush"
    elif 12 <= hour <= 14:
        return "Lunch Hour"
    elif 17 <= hour <= 19:
        return "Evening Rush"
    elif 20 <= hour <= 23:
        return "Evening Traffic"
    elif 0 <= hour <= 6:
        return "Night Time"
    else:
        return "Normal Traffic"