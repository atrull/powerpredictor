"""
Vehicle specifications for power calculations
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VehicleSpecs:
    """Vehicle specifications for power calculations"""
    weight_kg: float
    occupant_weight_kg: float
    final_drive: float
    gear_ratio: float
    tire_width: int  # mm
    tire_sidewall: int  # %
    tire_diameter: int  # inches
    engine_displacement: float  # liters
    cylinders: int
    
    @property
    def total_weight_kg(self) -> float:
        """Total weight in kg"""
        return self.weight_kg + self.occupant_weight_kg
    
    @property
    def tire_circumference_m(self) -> float:
        """Tire circumference in meters"""
        # Calculate tire diameter in mm
        sidewall_height = (self.tire_width * self.tire_sidewall / 100)
        tire_diameter_mm = (self.tire_diameter * 25.4) + (2 * sidewall_height)
        # Convert to circumference in meters
        return (tire_diameter_mm * np.pi) / 1000