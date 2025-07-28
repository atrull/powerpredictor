"""
Power run detection logic
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from constants import AnalysisConstants

# Lazy imports for heavy dependencies
def _import_pandas():
    import pandas as pd
    return pd


class RunDetector:
    """Handles detection of WOT power runs in ECU data"""
    
    def __init__(self, max_gap: int = 5):
        self.max_gap = max_gap
    
    def find_power_runs(self, data: 'pd.DataFrame', min_duration: float = 1.0, min_rpm_range: float = 2500, 
                       throttle_threshold: float = 96) -> List[Dict]:
        """
        Find periods where throttle is at WOT and RPM increases steadily
        
        Args:
            data: DataFrame containing ECU log data
            min_duration: Minimum duration in seconds for a valid run
            min_rpm_range: Minimum RPM range for a valid run
            throttle_threshold: Minimum throttle percentage for WOT detection
            
        Returns:
            List of power run dictionaries with start/end indices
        """
        if data is None:
            raise ValueError("No data provided.")
            
        # Use main TPS column, fallback to TPS 2 if needed
        tps_col = 'TPS (Main)' if 'TPS (Main)' in data.columns else 'TPS 2(Main)'
        
        if tps_col not in data.columns or 'Engine Speed' not in data.columns:
            raise ValueError("Required columns (TPS, Engine Speed) not found in data")
        
        # Enhanced WOT detection with hysteresis to handle brief throttle dips
        # Use a rolling window to smooth throttle readings and detect sustained high throttle
        throttle_window = min(5, len(data) // 40)
        if throttle_window < 3:
            throttle_window = 3
            
        # Rolling average throttle position
        smoothed_tps = data[tps_col].rolling(window=throttle_window, center=True, min_periods=1).mean()
        
        # Primary WOT condition: sustained high throttle
        primary_wot_mask = smoothed_tps >= throttle_threshold
        
        # Secondary WOT condition: allow slightly lower throttle (90%) if RPM is increasing rapidly
        # This catches the end of power runs where throttle may dip but power is still being made
        secondary_wot_mask = data[tps_col] >= (throttle_threshold - 6)  # 90% for 96% threshold
        
        # Progressive WOT condition: allow lower throttle at beginning if acceleration is strong
        # This catches the start of power runs where throttle is building up
        progressive_wot_mask = data[tps_col] >= 80
        
        # Enhanced sustained acceleration detection
        # Calculate time-based RPM acceleration (RPM/sec) using a rolling window
        time_diffs = data['Section Time'].diff()
        rpm_diffs = data['Engine Speed'].diff()
        
        # Calculate instantaneous acceleration (RPM/sec)
        instantaneous_accel = rpm_diffs / time_diffs
        
        # Use a rolling window to smooth acceleration and detect sustained increases
        window_size = min(10, len(data) // 20)  # Adaptive window size
        if window_size < 3:
            window_size = 3
            
        # Rolling average acceleration over the window
        sustained_accel = instantaneous_accel.rolling(window=window_size, center=True, min_periods=2).mean()
        
        # Detect different types of valid acceleration periods
        # Strong acceleration: definitive power pull
        strong_acceleration = sustained_accel > 100  # Strong acceleration threshold
        
        # Moderate acceleration: steady power delivery
        moderate_acceleration = sustained_accel > 25  # Moderate acceleration threshold
        
        # Maintain RPM: high RPM with minimal deceleration (end of power band)
        maintain_rpm = (sustained_accel > -100) & (data['Engine Speed'] > 4000)
        
        # Combine acceleration conditions
        acceleration_condition = strong_acceleration | (moderate_acceleration & secondary_wot_mask) | (maintain_rpm & secondary_wot_mask)
        
        # Combine WOT and acceleration conditions
        # Primary: Full WOT with any valid acceleration
        # Secondary: High throttle (90%+) with strong acceleration or high RPM maintenance  
        # Progressive: Lower throttle (80%+) but only with very strong acceleration (early power run)
        wot_condition = (primary_wot_mask | 
                        (secondary_wot_mask & (strong_acceleration | maintain_rpm)) |
                        (progressive_wot_mask & strong_acceleration & (data['Engine Speed'] < 3000)))
        
        # Final valid conditions
        valid_conditions = wot_condition & acceleration_condition & (data['Engine Speed'] > AnalysisConstants.DEFAULT_MIN_RPM)
        
        # Find continuous runs with gap tolerance to merge close runs
        runs = []
        in_run = False
        start_idx = None
        gap_count = 0
        max_gap = max(self.max_gap, 15)  # Increase minimum gap tolerance for better run detection
        
        for i, valid in enumerate(valid_conditions):
            if valid and not in_run:
                start_idx = i
                in_run = True
                gap_count = 0
            elif valid and in_run:
                if gap_count > 0:
                    # We had a gap but it's been bridged
                    print(f"Gap bridged: {gap_count} invalid samples at index {i-gap_count}-{i-1} (RPM: {data.iloc[i-gap_count]['Engine Speed']:.0f}-{data.iloc[i-1]['Engine Speed']:.0f})")
                gap_count = 0  # Reset gap counter when we have valid data
            elif not valid and in_run:
                gap_count += 1
                if gap_count == 1:
                    # First invalid sample in potential gap
                    print(f"Gap detected at index {i} (RPM: {data.iloc[i]['Engine Speed']:.0f}, TPS: {data.iloc[i][tps_col]:.1f}%)")
                if gap_count > max_gap:
                    # End the run only after max_gap consecutive invalid samples
                    end_idx = i - gap_count - 1
                    print(f"Gap exceeded max_gap ({max_gap}): ending run with {gap_count} consecutive invalid samples at index {i-gap_count+1}-{i}")
                    
                    # Check if run meets minimum criteria
                    duration = data.iloc[end_idx]['Section Time'] - data.iloc[start_idx]['Section Time']
                    rpm_range = data.iloc[end_idx]['Engine Speed'] - data.iloc[start_idx]['Engine Speed']
                    
                    if duration >= min_duration and rpm_range >= min_rpm_range:
                        runs.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'duration': duration,
                            'rpm_range': rpm_range,
                            'start_rpm': data.iloc[start_idx]['Engine Speed'],
                            'end_rpm': data.iloc[end_idx]['Engine Speed']
                        })
                    
                    in_run = False
                    gap_count = 0
        
        # Handle case where run continues to end of data
        if in_run and start_idx is not None:
            end_idx = len(data) - 1 - gap_count
            duration = data.iloc[end_idx]['Section Time'] - data.iloc[start_idx]['Section Time']
            rpm_range = data.iloc[end_idx]['Engine Speed'] - data.iloc[start_idx]['Engine Speed']
            
            if duration >= min_duration and rpm_range >= min_rpm_range:
                runs.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration': duration,
                    'rpm_range': rpm_range,
                    'start_rpm': data.iloc[start_idx]['Engine Speed'],
                    'end_rpm': data.iloc[end_idx]['Engine Speed']
                })
        
        # Debug information about detection criteria
        if len(runs) > 0:
            print(f"Found {len(runs)} valid power runs with enhanced detection:")
            for i, run in enumerate(runs):
                print(f"  Run {i+1}: {run['start_rpm']:.0f}-{run['end_rpm']:.0f} RPM, {run['duration']:.1f}s")
        else:
            # Provide diagnostic information when no runs found
            wot_samples = np.sum(primary_wot_mask)
            accel_samples = np.sum(acceleration_condition)
            rpm_samples = np.sum(data['Engine Speed'] > AnalysisConstants.DEFAULT_MIN_RPM)
            combined_samples = np.sum(valid_conditions)
            
            print(f"No power runs found. Diagnostic info:")
            print(f"  Samples with TPS ≥ {throttle_threshold}%: {wot_samples}")
            print(f"  Samples with good acceleration: {accel_samples}")
            print(f"  Samples with RPM > {AnalysisConstants.DEFAULT_MIN_RPM}: {rpm_samples}")
            print(f"  Samples meeting all conditions: {combined_samples}")
            print(f"  Try lowering --throttle-threshold or --min-rpm-range")
        
        return runs