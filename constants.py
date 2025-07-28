"""
Constants and analysis parameters for power analysis
"""

import numpy as np


class AnalysisConstants:
    """Constants used throughout the power analysis"""
    
    # Physics constants
    HP_TORQUE_CROSSOVER_RPM = 5252
    WATTS_TO_HP = 745.7
    NM_TO_LBFT = 0.737562
    RPM_TO_RAD_PER_SEC = 2 * np.pi / 60
    GRAVITY_MS2 = 9.81
    AIR_DENSITY_KG_M3 = 1.225
    
    # Analysis defaults
    DEFAULT_THROTTLE_THRESHOLD = 96
    DEFAULT_MIN_RPM = 1500
    DEFAULT_MIN_DURATION = 1.0
    DEFAULT_MIN_RPM_RANGE = 2500
    DEFAULT_SMOOTHING_FACTOR = 2.5
    DEFAULT_MAX_GAP = 5
    
    # Data quality thresholds
    HIGH_FREQUENCY_THRESHOLD = 200  # Hz
    STAIR_STEP_RATIO_THRESHOLD = 0.3
    MAX_RPM_CHANGE_PER_SEC = 2000
    
    # Validation limits
    MAX_REALISTIC_POWER = 1000  # HP
    MAX_REALISTIC_TORQUE = 1000  # lb-ft