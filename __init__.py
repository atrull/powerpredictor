"""
Power Predictor - ECU Log Analysis Tool

A modular tool for analyzing ECU log files to calculate and display power and torque curves
similar to a dynamometer.
"""

from .analyzer import PowerAnalyzer
from .constants import AnalysisConstants  
from .vehicle_specs import VehicleSpecs
from .data_loader import DataLoader
from .data_processing import DataProcessor
from .power_calculator import PowerCalculator
from .run_detector import RunDetector
from .plotting import Plotter

__version__ = "2.0.0"
__author__ = "Power Predictor Team"

# Main exports for easy importing
__all__ = [
    'PowerAnalyzer',
    'AnalysisConstants',
    'VehicleSpecs',
    'DataLoader', 
    'DataProcessor',
    'PowerCalculator',
    'RunDetector',
    'Plotter'
]