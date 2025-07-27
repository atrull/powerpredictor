#!/usr/bin/env python3
"""
Power and Torque Analysis Tool for ECU Log Data
Analyzes CSV logs to calculate power and torque curves like a dynamometer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import warnings
import sys
warnings.filterwarnings('ignore')

# Lazy imports for heavy dependencies
def _import_pandas():
    import pandas as pd
    return pd

def _import_matplotlib():
    import matplotlib.pyplot as plt
    return plt

def _import_scipy():
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    return signal, gaussian_filter1d, savgol_filter

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

class PowerAnalyzer:
    """Main class for analyzing ECU log data and calculating power/torque"""
    
    def __init__(self, vehicle_specs: VehicleSpecs, 
                 drivetrain_efficiency: float = 0.85,
                 rolling_resistance: float = 0.015,
                 drag_coefficient: float = 0.35,
                 frontal_area: float = 2.5,
                 smoothing_factor: float = 2.5,
                 apply_hp_torque_correction: bool = True,
                 filter_rpm_data: bool = True,
                 max_gap: int = 5,
                 downsample_hz: float = None):
        self.vehicle_specs = vehicle_specs
        self.drivetrain_efficiency = drivetrain_efficiency
        self.rolling_resistance = rolling_resistance
        self.drag_coefficient = drag_coefficient
        self.frontal_area = frontal_area
        self.smoothing_factor = smoothing_factor
        self.apply_hp_torque_correction = apply_hp_torque_correction
        self.filter_rpm_data = filter_rpm_data
        self.max_gap = max_gap
        self.downsample_hz = downsample_hz
        self.data = None
        self.power_runs = []
        
    def load_data(self, csv_path: str) -> None:
        """Load CSV data and clean column names"""
        try:
            # Read CSV - header is in row 2 (0-indexed row 1), data starts from row 4
            pd = _import_pandas()
            self.data = pd.read_csv(csv_path, header=1, skiprows=[2])  # Skip units row
            
            # Clean column names
            self.data.columns = [str(col).strip('"').strip() for col in self.data.columns]
            
            # Convert key columns to numeric, handling errors
            numeric_columns = [
                'Section Time', 'Engine Speed', 'TPS (Main)', 'TPS 2(Main)',
                'Driven Wheel Speed', 'Acceleration', 'MAP', 'BAP', 'IAT',
                'Lambda 1', 'Lambda Avg'
            ]
            
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Assess data quality and provide recommendations
            quality_metrics = self._assess_data_quality()
            
            # Detect optimal frequency if auto-downsampling not specified
            if self.downsample_hz is None:
                recommended_freq = self._detect_optimal_frequency()
                if recommended_freq and quality_metrics.get('quality_level') == 'needs_cleanup':
                    print(f"Auto-enabling downsampling to {recommended_freq}Hz due to data quality")
                    self.downsample_hz = recommended_freq
            
            # Apply generic parameter cleanup before any other processing
            self._cleanup_critical_parameters()
            
            # Apply downsampling FIRST if requested - this prevents RPM filtering from being overwritten
            if self.downsample_hz is not None:
                original_count = len(self.data)
                self.data = self._downsample_data(self.downsample_hz)
                print(f"Downsampled from {original_count} to {len(self.data)} data points at {self.downsample_hz}Hz")
            
            # Filter RPM data AFTER downsampling to handle ECU reporting issues on final dataset
            if self.filter_rpm_data:
                self._filter_rpm_data()
                    
            print(f"Loaded {len(self.data)} data points from {csv_path}")
            
        except Exception as e:
            raise ValueError(f"Error loading CSV data: {e}")
    
    def _filter_rpm_data(self) -> None:
        """
        Advanced RPM data filtering for ECUs with poor synchronization
        Uses multi-stage approach: outlier detection, trend validation, and adaptive smoothing
        """
        if 'Engine Speed' not in self.data.columns or 'Section Time' not in self.data.columns:
            return
        
        pd = _import_pandas()
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        
        rpm_col = 'Engine Speed'
        time_col = 'Section Time'
        tps_col = 'TPS (Main)' if 'TPS (Main)' in self.data.columns else 'TPS 2(Main)'
        
        # Make a copy to work with
        filtered_data = self.data.copy()
        
        rpm_values = filtered_data[rpm_col].values.copy().astype(float)
        time_values = filtered_data[time_col].values
        tps_values = filtered_data[tps_col].values if tps_col in filtered_data.columns else np.full(len(rpm_values), 0)
        
        # Create WOT mask for stair-step detection
        wot_mask = tps_values > 95
        
        # Calculate time deltas for physical constraint checking
        time_deltas = np.diff(time_values)
        
        problems_found = 0
        outliers_found = 0
        trend_violations = 0
        
        # Stage 1: Remove exact duplicates and clearly erroneous values
        for i in range(1, len(rpm_values)):
            if pd.notna(rpm_values[i]) and pd.notna(rpm_values[i-1]):
                # Remove exact duplicates at different timestamps
                if (rpm_values[i] == rpm_values[i-1] and 
                    time_values[i] != time_values[i-1]):
                    rpm_values[i] = np.nan
                    problems_found += 1
                # Remove negative RPM values (clearly impossible)
                elif rpm_values[i] < 0:
                    rpm_values[i] = np.nan
                    problems_found += 1
        
        # Stage 1.5: Enhanced stair-step detection for RPM plateaus during WOT
        stair_step_fixes = 0
        for i in range(2, len(rpm_values) - 2):
            if (wot_mask[i] and pd.notna(rpm_values[i-2:i+3]).all()):
                # Check for flat segments during WOT acceleration
                window = rpm_values[i-2:i+3]
                time_window = time_values[i-2:i+3]
                
                # Calculate expected RPM trend
                time_span = time_window[-1] - time_window[0]
                if time_span > 0.05:  # At least 50ms span
                    # Check if middle values are flat while we should be accelerating
                    if (abs(window[1] - window[2]) < 1 and abs(window[2] - window[3]) < 1 and
                        window[4] > window[0] + 10):  # Overall increasing trend
                        
                        # Apply micro-smoothing to flat segment using linear interpolation
                        rpm_values[i-1:i+2] = np.interp(
                            time_window[1:4], 
                            [time_window[0], time_window[4]], 
                            [window[0], window[4]]
                        )
                        stair_step_fixes += 1
        
        if stair_step_fixes > 0:
            print(f"Enhanced RPM stair-step fixes applied: {stair_step_fixes} plateau segments smoothed")
        
        # Stage 2: Enhanced outlier detection with median filtering
        window_size = 5
        median_filtered = signal.medfilt(rpm_values, kernel_size=window_size)
        
        # Find outliers that deviate significantly from median-filtered signal
        for i in range(len(rpm_values)):
            if pd.notna(rpm_values[i]) and pd.notna(median_filtered[i]):
                deviation = abs(rpm_values[i] - median_filtered[i])
                # More aggressive outlier detection for large deviations
                if deviation > 150:  # RPM significantly different from local median
                    rpm_values[i] = np.nan
                    outliers_found += 1
        
        # Stage 3: Trend-based validation for WOT sections  
        max_rpm_per_sec = 2000  # Slightly more generous limit
        
        # Apply trend validation in WOT sections
        for i in range(1, len(rpm_values) - 1):
            if (wot_mask[i] and pd.notna(rpm_values[i-1]) and 
                pd.notna(rpm_values[i]) and pd.notna(rpm_values[i+1])):
                
                dt_prev = time_deltas[i-1] if i-1 < len(time_deltas) else 0.02
                dt_next = time_deltas[i] if i < len(time_deltas) else 0.02
                
                # During WOT, RPM should generally increase or stay stable
                if dt_prev > 0:
                    accel_rate = (rpm_values[i] - rpm_values[i-1]) / dt_prev
                    
                    # Flag massive drops during WOT as trend violations
                    if accel_rate < -800:  # More than 800 RPM/sec drop during WOT
                        rpm_values[i] = np.nan
                        trend_violations += 1
                    # Also flag impossible accelerations
                    elif abs(accel_rate) > max_rpm_per_sec:
                        rpm_values[i] = np.nan
                        trend_violations += 1
        
        # Stage 4: Find and process WOT sections with enhanced smoothing
        wot_sections = []
        in_wot = False
        start_idx = None
        
        for i, is_wot in enumerate(wot_mask):
            if is_wot and not in_wot:
                start_idx = i
                in_wot = True
            elif not is_wot and in_wot:
                if start_idx is not None and i - start_idx > 10:
                    wot_sections.append((start_idx, i))
                in_wot = False
                start_idx = None
        
        # Handle case where WOT continues to end
        if in_wot and start_idx is not None:
            wot_sections.append((start_idx, len(wot_mask)))
        
        # Process each WOT section with adaptive smoothing
        for start, end in wot_sections:
            section_rpm = rpm_values[start:end].copy()
            section_time = time_values[start:end]
            section_mask = ~np.isnan(section_rpm)
            
            if np.sum(section_mask) > 5:  # Need at least 5 valid points
                # First, interpolate any NaN values
                if np.any(~section_mask):
                    section_rpm = np.interp(
                        section_time,
                        section_time[section_mask],
                        section_rpm[section_mask]
                    )
                
                # Apply adaptive smoothing based on section length
                if len(section_rpm) >= 15:
                    # For longer sections, use Savitzky-Golay filter
                    try:
                        window_length = min(11, len(section_rpm) // 2)
                        if window_length % 2 == 0:
                            window_length -= 1
                        if window_length >= 5:
                            section_rpm = savgol_filter(section_rpm, window_length, 3)
                    except:
                        # Fallback to Gaussian smoothing
                        sigma = len(section_rpm) / 20.0
                        section_rpm = gaussian_filter1d(section_rpm, sigma=sigma)
                
                elif len(section_rpm) >= 7:
                    # For medium sections, use Gaussian smoothing
                    sigma = len(section_rpm) / 15.0
                    section_rpm = gaussian_filter1d(section_rpm, sigma=sigma)
                
                # Apply monotonicity constraint for WOT sections
                # RPM should generally increase during acceleration
                section_rpm = self._apply_monotonicity_constraint(section_rpm, section_time)
                
                # Update the main array
                rpm_values[start:end] = section_rpm
        
        # Stage 5: Final interpolation for any remaining NaN values
        rpm_mask = ~np.isnan(rpm_values)
        if np.sum(rpm_mask) > 0:
            rpm_values = np.interp(
                time_values,
                time_values[rpm_mask],
                rpm_values[rpm_mask]
            )
        
        # Update the filtered data
        filtered_data[rpm_col] = rpm_values
        
        total_issues = problems_found + outliers_found + trend_violations
        if total_issues > 0:
            print(f"Advanced RPM filtering: {problems_found} bad values, {outliers_found} outliers, "
                  f"{trend_violations} trend violations, {len(wot_sections)} WOT sections processed")
        
        # Update the main data
        self.data = filtered_data
    
    def _enforce_rpm_monotonicity(self, rpm_values: np.ndarray, min_increment: float = 0.1) -> Tuple[np.ndarray, int]:
        """
        Ensure RPM values are strictly increasing for proper power calculations
        
        Args:
            rpm_values: Array of RPM values to fix
            min_increment: Minimum increment to add when fixing reversals
            
        Returns:
            Tuple of (corrected_rpm_values, number_of_fixes_applied)
        """
        corrected = rpm_values.copy()
        fixes = 0
        
        for i in range(1, len(corrected)):
            if corrected[i] <= corrected[i-1]:
                corrected[i] = corrected[i-1] + min_increment
                fixes += 1
        
        return corrected, fixes
    
    def _apply_generic_parameter_cleanup(self, data, column_name: str, time_column: str = 'Section Time') -> np.ndarray:
        """
        Generic forward-averaging cleanup for stair-step ECU data
        
        Addresses the "stairs vs angle" problem by detecting flat segments
        and interpolating smooth transitions between stable values.
        
        Args:
            data: DataFrame containing the data
            column_name: Name of the parameter column to clean
            time_column: Name of the time column
            
        Returns:
            Cleaned parameter values as numpy array
        """
        if column_name not in data.columns:
            return None
            
        pd = _import_pandas()
        values = data[column_name].values.copy().astype(float)
        time_values = data[time_column].values
        
        # Remove NaN values for processing
        valid_mask = ~pd.isna(values)
        if np.sum(valid_mask) < 3:
            return values
        
        # Detect stair-step pattern: consecutive identical values with time progression
        change_points = [0]  # Start with first point
        tolerance = 0.001  # Small tolerance for floating point comparison
        
        for i in range(1, len(values)):
            if valid_mask[i] and valid_mask[i-1]:
                if abs(values[i] - values[i-1]) > tolerance:
                    change_points.append(i)
        
        change_points.append(len(values) - 1)  # End with last point
        
        if len(change_points) < 3:  # Need at least start, middle, end
            return values
        
        # Apply forward moving average between change points
        smoothed_values = values.copy()
        
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            
            if end_idx - start_idx > 2:  # Only smooth segments with multiple flat points
                segment_time = time_values[start_idx:end_idx+1]
                start_val = values[start_idx]
                end_val = values[end_idx]
                
                # Create smooth transition using linear interpolation
                # This preserves the trend while smoothing the stair-step pattern
                if len(segment_time) > 1 and segment_time[-1] != segment_time[0]:
                    smoothed_segment = np.interp(segment_time, 
                                               [segment_time[0], segment_time[-1]], 
                                               [start_val, end_val])
                    smoothed_values[start_idx:end_idx+1] = smoothed_segment
        
        return smoothed_values

    def _assess_data_quality(self):
        """Assess data quality and recommend processing approach"""
        quality_metrics = {}
        
        # Check for stair-step patterns in key parameters
        if 'Engine Speed' in self.data.columns:
            rpm_changes = np.diff(self.data['Engine Speed'].dropna())
            zero_changes = np.sum(np.abs(rpm_changes) < 0.001)
            quality_metrics['rpm_stair_step_ratio'] = zero_changes / len(rpm_changes) if len(rpm_changes) > 0 else 0
        
        # Detect logging frequency
        time_diffs = np.diff(self.data['Section Time'].values)
        quality_metrics['avg_sample_rate'] = 1.0 / np.mean(time_diffs) if len(time_diffs) > 0 else 0
        quality_metrics['sample_rate_std'] = np.std(time_diffs)
        
        # Count total parameters
        quality_metrics['total_parameters'] = len(self.data.columns)
        
        # Assess overall data quality
        if quality_metrics['rpm_stair_step_ratio'] > 0.3:
            quality_metrics['quality_level'] = 'needs_cleanup'
            print(f"High stair-step pattern detected ({quality_metrics['rpm_stair_step_ratio']:.1%}) - enhanced smoothing recommended")
        elif quality_metrics['rpm_stair_step_ratio'] > 0.1:
            quality_metrics['quality_level'] = 'moderate_cleanup'
            print(f"Moderate stair-step pattern detected ({quality_metrics['rpm_stair_step_ratio']:.1%}) - light cleanup recommended")
        else:
            quality_metrics['quality_level'] = 'good'
            print(f"Good data quality detected ({quality_metrics['rpm_stair_step_ratio']:.1%} flat segments)")
        
        print(f"Data characteristics: {quality_metrics['avg_sample_rate']:.1f}Hz sampling, {quality_metrics['total_parameters']} parameters")
        
        return quality_metrics

    def _detect_optimal_frequency(self):
        """Detect optimal processing frequency based on data characteristics"""
        time_diffs = np.diff(self.data['Section Time'].values)
        avg_freq = 1.0 / np.mean(time_diffs)
        
        # Determine optimal frequency based on current frequency and data quality
        if avg_freq > 200:  # High frequency like k20_pull.csv
            recommended_freq = 50  # Downsample significantly
            print(f"High frequency data detected ({avg_freq:.1f}Hz) - recommending {recommended_freq}Hz for analysis")
        elif avg_freq > 100:
            recommended_freq = 25  # Moderate downsampling
            print(f"Medium frequency data detected ({avg_freq:.1f}Hz) - recommending {recommended_freq}Hz for analysis")
        elif avg_freq > 75:
            recommended_freq = None  # Light downsampling or none
            print(f"Good frequency data detected ({avg_freq:.1f}Hz) - no downsampling needed")
        else:
            recommended_freq = None  # No downsampling needed
            print(f"Standard frequency data detected ({avg_freq:.1f}Hz) - optimal for analysis")
        
        return recommended_freq

    def _cleanup_critical_parameters(self):
        """Apply generic cleanup to critical ECU parameters that commonly show stair-step patterns"""
        critical_params = [
            'Engine Speed', 'TPS (Main)', 'TPS 2(Main)', 'MAP', 
            'Lambda 1', 'Lambda Avg', 'IAT', 'BAP'
        ]
        
        cleaned_count = 0
        for param in critical_params:
            if param in self.data.columns:
                original_values = self.data[param].values.copy()
                cleaned_values = self._apply_generic_parameter_cleanup(self.data, param)
                
                if cleaned_values is not None:
                    # Calculate improvement metric
                    original_changes = np.diff(original_values[~np.isnan(original_values)])
                    cleaned_changes = np.diff(cleaned_values[~np.isnan(cleaned_values)])
                    
                    if len(original_changes) > 0 and len(cleaned_changes) > 0:
                        original_zeros = np.sum(np.abs(original_changes) < 0.001)
                        cleaned_zeros = np.sum(np.abs(cleaned_changes) < 0.001)
                        
                        if original_zeros > cleaned_zeros:
                            self.data[param] = cleaned_values
                            cleaned_count += 1
                            print(f"Applied stair-step cleanup to {param}: reduced flat segments from {original_zeros} to {cleaned_zeros}")
        
        if cleaned_count > 0:
            print(f"Generic cleanup applied to {cleaned_count} parameters")

    def _apply_monotonicity_constraint(self, rpm_values: np.ndarray, time_values: np.ndarray) -> np.ndarray:
        """
        Apply monotonicity constraint to RPM values during acceleration
        Ensures RPM generally increases during WOT pulls while allowing for minor variations
        """
        if len(rpm_values) < 3:
            return rpm_values
        
        corrected_rpm = rpm_values.copy()
        
        # Calculate the overall trend
        time_span = time_values[-1] - time_values[0]
        overall_rpm_rate = (rpm_values[-1] - rpm_values[0]) / time_span if time_span > 0 else 0
        
        # Only apply constraint if we're clearly in an acceleration phase
        if overall_rpm_rate > 100:  # More than 100 RPM/sec average increase
            # Smooth out backwards steps while preserving natural variation
            for i in range(1, len(corrected_rpm)):
                # Calculate expected RPM based on trend
                dt = time_values[i] - time_values[i-1] if i > 0 else 0.02
                expected_increase = overall_rpm_rate * dt * 0.5  # Allow 50% of average rate
                
                # If RPM decreases significantly, adjust it
                if corrected_rpm[i] < corrected_rpm[i-1] - expected_increase:
                    # Use a weighted average between current value and trend-based value
                    trend_value = corrected_rpm[i-1] + expected_increase * 0.5
                    corrected_rpm[i] = 0.7 * corrected_rpm[i] + 0.3 * trend_value
        
        return corrected_rpm
    
    def _smooth_rpm_steps(self, power_hp: np.ndarray, torque_lbft: np.ndarray, rpm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Targeted smoothing to eliminate small RPM step artifacts and end-of-run oscillations
        while preserving the overall power curve characteristics
        """
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        
        # Detect RPM steps - look for regular increments
        rpm_diffs = np.diff(rpm)
        median_step = np.median(rpm_diffs[rpm_diffs > 0])
        
        # Check for end-of-run oscillations (impossible zigzag patterns)
        power_changes = np.abs(np.diff(power_hp))
        end_third = len(power_hp) // 3
        end_section_changes = power_changes[-end_third:] if end_third > 5 else power_changes
        
        # Detect if end section has excessive oscillations
        oscillation_threshold = np.std(power_changes) * 2
        end_oscillations = np.sum(end_section_changes > oscillation_threshold)
        has_end_oscillations = end_oscillations > len(end_section_changes) * 0.3
        
        if has_end_oscillations:
            print(f"Detected end-of-run oscillations ({end_oscillations} spikes in final third) - applying progressive smoothing")
        
        # Apply different strategies based on what we detected
        if 50 < median_step < 500:  # Regular RPM steps
            print(f"Detected RPM steps of ~{median_step:.0f} RPM - applying targeted smoothing")
            window_size = max(3, min(7, int(len(power_hp) / 20)))
            if window_size % 2 == 0:
                window_size += 1
            
            try:
                smoothed_power = savgol_filter(power_hp, window_size, min(2, window_size-1))
                smoothed_torque = savgol_filter(torque_lbft, window_size, min(2, window_size-1))
                print(f"Applied light SavGol smoothing: window={window_size}")
            except (ValueError, np.linalg.LinAlgError):
                sigma = 1.0
                smoothed_power = gaussian_filter1d(power_hp, sigma=sigma)
                smoothed_torque = gaussian_filter1d(torque_lbft, sigma=sigma)
                print(f"Applied light Gaussian smoothing: σ={sigma}")
                
        elif has_end_oscillations:
            # Progressive smoothing - light in middle, heavier at ends
            print("Applying progressive smoothing to eliminate end oscillations")
            smoothed_power = power_hp.copy()
            smoothed_torque = torque_lbft.copy()
            
            # Apply heavier smoothing to the problematic end section
            if end_third > 5:
                end_window = min(9, end_third // 2)
                if end_window % 2 == 0:
                    end_window += 1
                
                try:
                    smoothed_power[-end_third:] = savgol_filter(power_hp[-end_third:], end_window, 2)
                    smoothed_torque[-end_third:] = savgol_filter(torque_lbft[-end_third:], end_window, 2)
                    print(f"Applied progressive smoothing: end section window={end_window}")
                except (ValueError, np.linalg.LinAlgError):
                    sigma = 2.0
                    smoothed_power[-end_third:] = gaussian_filter1d(power_hp[-end_third:], sigma=sigma)
                    smoothed_torque[-end_third:] = gaussian_filter1d(torque_lbft[-end_third:], sigma=sigma)
                    print(f"Applied progressive Gaussian smoothing: end σ={sigma}")
        else:
            # No major issues detected, minimal smoothing
            print(f"No major artifacts detected (median RPM step: {median_step:.0f}) - minimal smoothing")
            sigma = 0.5
            smoothed_power = gaussian_filter1d(power_hp, sigma=sigma)
            smoothed_torque = gaussian_filter1d(torque_lbft, sigma=sigma)
        
        return smoothed_power, smoothed_torque
    
    def _detect_and_smooth_rpm_step_artifacts(self, power_hp: np.ndarray, torque_lbft: np.ndarray, 
                                             rpm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and smooth step artifacts caused by ECU RPM 'step and jump' patterns.
        Uses angle analysis to detect sudden changes in power/torque slope that don't match
        the overall trend, then smooths backwards across the step.
        """
        if len(power_hp) < 10:
            return power_hp, torque_lbft
            
        power_smoothed = power_hp.copy()
        torque_smoothed = torque_lbft.copy()
        
        # Calculate the local slope (angle) for power and torque curves
        rpm_diffs = np.diff(rpm)
        power_diffs = np.diff(power_hp)
        torque_diffs = np.diff(torque_lbft)
        
        # Calculate slopes (power/rpm and torque/rpm) 
        power_slopes = np.divide(power_diffs, rpm_diffs, out=np.zeros_like(power_diffs), where=rpm_diffs!=0)
        torque_slopes = np.divide(torque_diffs, rpm_diffs, out=np.zeros_like(torque_diffs), where=rpm_diffs!=0)
        
        # Use rolling window to establish the expected trend
        window_size = max(5, len(power_hp) // 20)
        if window_size > 15:
            window_size = 15
            
        # Calculate expected slopes using rolling median (more robust to outliers)
        expected_power_slopes = np.zeros_like(power_slopes)
        expected_torque_slopes = np.zeros_like(torque_slopes)
        
        for i in range(len(power_slopes)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(power_slopes), i + window_size // 2 + 1)
            
            window_power_slopes = power_slopes[start_idx:end_idx]
            window_torque_slopes = torque_slopes[start_idx:end_idx]
            
            # Use median to get expected trend, ignoring extreme outliers
            expected_power_slopes[i] = np.median(window_power_slopes)
            expected_torque_slopes[i] = np.median(window_torque_slopes)
        
        # Detect step artifacts: slopes that deviate significantly from expected trend
        power_slope_deviations = np.abs(power_slopes - expected_power_slopes)
        torque_slope_deviations = np.abs(torque_slopes - expected_torque_slopes)
        
        # Calculate dynamic thresholds based on overall slope variation
        power_slope_threshold = np.std(expected_power_slopes) * 1.5  # 1.5 sigma threshold (very sensitive)
        torque_slope_threshold = np.std(expected_torque_slopes) * 1.5
        
        # Ensure very sensitive thresholds to catch ECU step artifacts
        power_slope_threshold = max(power_slope_threshold, 0.15)  # Minimum 0.15 HP/RPM deviation (more sensitive)
        torque_slope_threshold = max(torque_slope_threshold, 0.1)  # Minimum 0.1 lb-ft/RPM deviation (more sensitive)
        
        # Find step artifacts
        power_artifacts = power_slope_deviations > power_slope_threshold
        torque_artifacts = torque_slope_deviations > torque_slope_threshold
        
        # Combine artifacts (if either power or torque shows artifact, treat as step)
        step_artifacts = power_artifacts | torque_artifacts
        
        # Debug: Check for specific known artifacts around 3836 and 4252 RPM (disabled for now)
        if False:  # Temporarily disable verbose debug output
            debug_rpms = [3836, 4252]
            for debug_rpm in debug_rpms:
                closest_idx = np.argmin(np.abs(rpm - debug_rpm))
                if closest_idx < len(power_slopes):
                    slope_idx = closest_idx if closest_idx < len(power_slopes) else len(power_slopes) - 1
                    power_dev = power_slope_deviations[slope_idx] if slope_idx < len(power_slope_deviations) else 0
                    torque_dev = torque_slope_deviations[slope_idx] if slope_idx < len(torque_slope_deviations) else 0
                    
                    print(f"Debug artifact check at {debug_rpm} RPM: power_dev={power_dev:.3f}, torque_dev={torque_dev:.3f}")
        
        if np.any(step_artifacts):
            artifact_indices = np.where(step_artifacts)[0]
            print(f"Detected {len(artifact_indices)} step artifacts at RPM positions: {rpm[artifact_indices + 1].astype(int)}")
            print(f"  Power slope threshold: {power_slope_threshold:.3f}, Torque slope threshold: {torque_slope_threshold:.3f}")
            
            # Process each artifact
            for artifact_idx in artifact_indices:
                # Get the actual data point index (artifact_idx is slope index, so +1 for data point)
                step_point = artifact_idx + 1
                
                # Define smoothing window around the step
                smooth_start = max(0, step_point - 3)
                smooth_end = min(len(power_hp), step_point + 4)
                
                if smooth_end - smooth_start >= 4:  # Need at least 4 points to smooth
                    # Get the trend before and after the step
                    before_start = max(0, smooth_start - 5)
                    after_end = min(len(power_hp), smooth_end + 5)
                    
                    # Calculate trend using points before and after the artifact region
                    trend_points_power = np.concatenate([
                        power_hp[before_start:smooth_start],
                        power_hp[smooth_end:after_end]
                    ])
                    trend_points_torque = np.concatenate([
                        torque_lbft[before_start:smooth_start], 
                        torque_lbft[smooth_end:after_end]
                    ])
                    trend_points_rpm = np.concatenate([
                        rpm[before_start:smooth_start],
                        rpm[smooth_end:after_end]
                    ])
                    
                    if len(trend_points_rpm) >= 4:
                        # Fit linear trend through the surrounding points
                        try:
                            power_trend_coeff = np.polyfit(trend_points_rpm, trend_points_power, 1)
                            torque_trend_coeff = np.polyfit(trend_points_rpm, trend_points_torque, 1)
                            
                            # Calculate expected values based on trend
                            smooth_rpm = rpm[smooth_start:smooth_end]
                            expected_power = np.polyval(power_trend_coeff, smooth_rpm)
                            expected_torque = np.polyval(torque_trend_coeff, smooth_rpm)
                            
                            # Blend with original values using weights that emphasize the trend
                            # Give more weight to trend at the step point, less weight at the edges
                            weights = np.ones(len(smooth_rpm))
                            step_relative_pos = step_point - smooth_start
                            if 0 <= step_relative_pos < len(weights):
                                weights[step_relative_pos] = 0.8  # Heavy smoothing at step point
                                # Moderate smoothing for adjacent points
                                if step_relative_pos > 0:
                                    weights[step_relative_pos - 1] = 0.6
                                if step_relative_pos < len(weights) - 1:
                                    weights[step_relative_pos + 1] = 0.6
                            
                            # Apply blended smoothing
                            original_power = power_smoothed[smooth_start:smooth_end]
                            original_torque = torque_smoothed[smooth_start:smooth_end]
                            
                            power_smoothed[smooth_start:smooth_end] = (
                                weights * expected_power + (1 - weights) * original_power
                            )
                            torque_smoothed[smooth_start:smooth_end] = (
                                weights * expected_torque + (1 - weights) * original_torque
                            )
                            
                            # print(f"  Smoothed artifact at {rpm[step_point]:.0f} RPM")  # Disabled for debugging
                            
                        except np.linalg.LinAlgError:
                            # Fallback to simple interpolation if polyfit fails
                            if smooth_start > 0 and smooth_end < len(power_hp):
                                smooth_rpm = rpm[smooth_start:smooth_end]
                                start_power, end_power = power_hp[smooth_start-1], power_hp[smooth_end]
                                start_torque, end_torque = torque_lbft[smooth_start-1], torque_lbft[smooth_end]
                                start_rpm, end_rpm = rpm[smooth_start-1], rpm[smooth_end]
                                
                                interp_power = np.interp(smooth_rpm, [start_rpm, end_rpm], [start_power, end_power])
                                interp_torque = np.interp(smooth_rpm, [start_rpm, end_rpm], [start_torque, end_torque])
                                
                                # Blend interpolated with original (70% interpolated, 30% original)
                                power_smoothed[smooth_start:smooth_end] = (
                                    0.7 * interp_power + 0.3 * power_smoothed[smooth_start:smooth_end]
                                )
                                torque_smoothed[smooth_start:smooth_end] = (
                                    0.7 * interp_torque + 0.3 * torque_smoothed[smooth_start:smooth_end]
                                )
        
        return power_smoothed, torque_smoothed

    def _apply_physics_aware_smoothing(self, power_hp: np.ndarray, torque_lbft: np.ndarray, 
                                     rpm_smoothed: np.ndarray, rpm_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply physics-aware smoothing that eliminates impossible sharp angles in power curves
        while respecting the fundamental relationship between RPM and power output
        """
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        
        # Check if this is downsampled high-frequency data
        was_downsampled = hasattr(self, 'downsample_hz') and self.downsample_hz is not None
        
        if was_downsampled:
            # For downsampled high-frequency data, apply VERY aggressive smoothing to eliminate artifacts
            # These datasets have severe "dot artifacts" in torque that create power kinks
            
            # Apply extremely heavy torque smoothing first to eliminate "dots smashed together"
            sigma_torque_ultra = max(5.0, len(torque_lbft) / 15.0)  # Much more aggressive
            torque_ultra_smoothed = gaussian_filter1d(torque_lbft, sigma=sigma_torque_ultra)
            
            # Then apply standard aggressive smoothing to power
            sigma_aggressive = max(3.0, len(power_hp) / 30.0)
            power_heavily_smoothed = gaussian_filter1d(power_hp, sigma=sigma_aggressive)
            
            print(f"Physics-aware smoothing: ultra-aggressive torque σ={sigma_torque_ultra:.2f}, power σ={sigma_aggressive:.2f} for downsampled data")
            
            # Apply monotonicity constraint: power shouldn't increase dramatically during RPM plateaus
            rpm_changes = np.diff(rpm_smoothed)
            power_changes = np.diff(power_heavily_smoothed)
            
            # Find sections where RPM is nearly flat but power jumps
            rpm_plateau_mask = np.abs(rpm_changes) < 10  # RPM change < 10
            power_jump_mask = power_changes > 5  # Power increase > 5 HP
            violations = rpm_plateau_mask & power_jump_mask
            
            if np.sum(violations) > 0:
                print(f"Applying monotonicity fixes to {np.sum(violations)} power plateau violations")
                
                # Smooth out violations by interpolating between non-violating points
                corrected_power = power_heavily_smoothed.copy()
                for i, is_violation in enumerate(violations):
                    if is_violation:
                        # Use trend-based interpolation instead of sharp jump
                        if i > 0 and i < len(corrected_power) - 2:
                            trend = (corrected_power[i+2] - corrected_power[i-1]) / 3
                            corrected_power[i+1] = corrected_power[i] + trend
                
                power_heavily_smoothed = corrected_power
            
            # Final light smoothing to ensure curves are completely smooth
            final_sigma = 1.0
            final_power = gaussian_filter1d(power_heavily_smoothed, sigma=final_sigma)
            final_torque = gaussian_filter1d(torque_ultra_smoothed, sigma=final_sigma)  # Use ultra-smoothed torque
            
            print(f"Applied final smoothing: σ={final_sigma}")
            
        else:
            # For normal frequency data, use the existing targeted smoothing approach
            final_power, final_torque = self._smooth_rpm_steps(power_hp, torque_lbft, rpm_original)
        
        # No need to re-apply HP-Torque correction here since it was applied before smoothing
        
        return final_power, final_torque
    
    def _downsample_data(self, target_hz: float):
        """
        Downsample data to target frequency using uniform time intervals
        
        Args:
            target_hz: Target sampling frequency in Hz
            
        Returns:
            Downsampled DataFrame
        """
        pd = _import_pandas()
        
        if 'Section Time' not in self.data.columns:
            raise ValueError("Section Time column required for downsampling")
        
        # Calculate target time step
        target_dt = 1.0 / target_hz
        
        # Get time range
        time_col = self.data['Section Time']
        start_time = time_col.min()
        end_time = time_col.max()
        
        # Create uniform time grid
        uniform_times = np.arange(start_time, end_time + target_dt, target_dt)
        
        # Interpolate all numeric columns to uniform time grid
        downsampled_data = pd.DataFrame({'Section Time': uniform_times})
        
        for col in self.data.columns:
            if col != 'Section Time' and pd.api.types.is_numeric_dtype(self.data[col]):
                # Remove NaN values for interpolation
                valid_mask = ~pd.isna(self.data[col])
                if valid_mask.sum() > 1:  # Need at least 2 points to interpolate
                    downsampled_data[col] = np.interp(
                        uniform_times,
                        time_col[valid_mask],
                        self.data[col][valid_mask]
                    )
                else:
                    # If not enough valid data, fill with the first valid value
                    first_valid = self.data[col][valid_mask].iloc[0] if valid_mask.sum() > 0 else 0
                    downsampled_data[col] = first_valid
            elif col != 'Section Time':
                # For non-numeric columns, forward fill from nearest time
                downsampled_data[col] = self.data[col].iloc[0]  # Simple approach
        
        print(f"Downsampled from {len(self.data)} to {len(downsampled_data)} points "
              f"({1/np.mean(np.diff(time_col)):.1f}Hz -> {target_hz}Hz)")
        
        return downsampled_data
    
    def _apply_dyno_style_smoothing(self, power_hp: np.ndarray, torque_lbft: np.ndarray, rpm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dyno-style smoothing that balances noise reduction with curve characteristics
        Enhanced for high-frequency downsampled data to eliminate remaining artifacts
        """
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        
        # Apply different smoothing strategies based on data length
        data_length = len(power_hp)
        
        # Check if this data was heavily downsampled (indicator of high-frequency source)
        was_downsampled = hasattr(self, 'downsample_hz') and self.downsample_hz is not None
        
        # Targeted RPM-step smoothing for downsampled high-frequency data
        if was_downsampled and data_length > 20:
            print(f"Applying targeted RPM-step smoothing for downsampled data ({data_length} points)")
            
            # Target small RPM increments (like 200 RPM steps) without destroying overall curve
            smoothed_power, smoothed_torque = self._smooth_rpm_steps(power_hp, torque_lbft, rpm)
            print(f"Applied targeted RPM-step smoothing to preserve power characteristics")
                
        elif data_length >= 20:
            # For longer datasets, use Savitzky-Golay filter for better edge preservation
            try:
                # Dynamic window sizing based on data length and smoothing factor
                window_length = int(max(5, min(15, data_length * self.smoothing_factor / 10.0)))
                if window_length % 2 == 0:
                    window_length += 1  # Ensure odd number
                
                # Use polynomial order 3 for smooth curves with good dynamics
                poly_order = min(3, window_length - 1)
                
                smoothed_power = savgol_filter(power_hp, window_length, poly_order)
                smoothed_torque = savgol_filter(torque_lbft, window_length, poly_order)
                
                # Apply light Gaussian post-processing for final smoothness
                if self.smoothing_factor > 2.0:
                    light_sigma = self.smoothing_factor * 0.3
                    smoothed_power = gaussian_filter1d(smoothed_power, sigma=light_sigma)
                    smoothed_torque = gaussian_filter1d(smoothed_torque, sigma=light_sigma)
                
            except (ValueError, np.linalg.LinAlgError):
                # Fallback to Gaussian smoothing
                sigma = self.smoothing_factor * min(1.5, data_length / 15.0)
                smoothed_power = gaussian_filter1d(power_hp, sigma=sigma)
                smoothed_torque = gaussian_filter1d(torque_lbft, sigma=sigma)
        
        elif data_length >= 10:
            # For medium datasets, use adaptive Gaussian smoothing
            sigma = self.smoothing_factor * min(1.2, data_length / 12.0)
            smoothed_power = gaussian_filter1d(power_hp, sigma=sigma)
            smoothed_torque = gaussian_filter1d(torque_lbft, sigma=sigma)
        
        else:
            # For short datasets, minimal smoothing to preserve what little data we have
            sigma = self.smoothing_factor * 0.5
            smoothed_power = gaussian_filter1d(power_hp, sigma=sigma)
            smoothed_torque = gaussian_filter1d(torque_lbft, sigma=sigma)
        
        # Ensure HP-Torque relationship is maintained after smoothing
        if self.apply_hp_torque_correction:
            # Recalculate power from smoothed torque to maintain HP-Torque crossover
            smoothed_power = (smoothed_torque * rpm) / AnalysisConstants.HP_TORQUE_CROSSOVER_RPM
        
        return smoothed_power, smoothed_torque
    
    def find_power_runs(self, min_duration: float = 1.0, min_rpm_range: float = 2500, 
                       throttle_threshold: float = 96) -> List[Dict]:
        """
        Find periods where throttle is at WOT and RPM increases steadily
        
        Args:
            min_duration: Minimum duration in seconds for a valid run
            min_rpm_range: Minimum RPM range for a valid run
            throttle_threshold: Minimum throttle percentage for WOT detection
            
        Returns:
            List of power run dictionaries with start/end indices
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Use main TPS column, fallback to TPS 2 if needed
        tps_col = 'TPS (Main)' if 'TPS (Main)' in self.data.columns else 'TPS 2(Main)'
        
        if tps_col not in self.data.columns or 'Engine Speed' not in self.data.columns:
            raise ValueError("Required columns (TPS, Engine Speed) not found in data")
        
        # Enhanced WOT detection with hysteresis to handle brief throttle dips
        # Use a rolling window to smooth throttle readings and detect sustained high throttle
        throttle_window = min(5, len(self.data) // 40)
        if throttle_window < 3:
            throttle_window = 3
            
        # Rolling average throttle position
        smoothed_tps = self.data[tps_col].rolling(window=throttle_window, center=True, min_periods=1).mean()
        
        # Primary WOT condition: sustained high throttle
        primary_wot_mask = smoothed_tps >= throttle_threshold
        
        # Secondary WOT condition: allow slightly lower throttle (90%) if RPM is increasing rapidly
        # This catches the end of power runs where throttle may dip but power is still being made
        secondary_wot_mask = self.data[tps_col] >= (throttle_threshold - 6)  # 90% for 96% threshold
        
        # Progressive WOT condition: allow lower throttle at beginning if acceleration is strong
        # This catches the start of power runs where throttle is building up
        progressive_wot_mask = self.data[tps_col] >= 80
        
        # Enhanced sustained acceleration detection
        # Calculate time-based RPM acceleration (RPM/sec) using a rolling window
        time_diffs = self.data['Section Time'].diff()
        rpm_diffs = self.data['Engine Speed'].diff()
        
        # Calculate instantaneous acceleration (RPM/sec)
        instantaneous_accel = rpm_diffs / time_diffs
        
        # Use a rolling window to smooth acceleration and detect sustained increases
        window_size = min(10, len(self.data) // 20)  # Adaptive window size
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
        maintain_rpm = (sustained_accel > -100) & (self.data['Engine Speed'] > 4000)
        
        # Combine acceleration conditions
        acceleration_condition = strong_acceleration | (moderate_acceleration & secondary_wot_mask) | (maintain_rpm & secondary_wot_mask)
        
        # Combine WOT and acceleration conditions
        # Primary: Full WOT with any valid acceleration
        # Secondary: High throttle (90%+) with strong acceleration or high RPM maintenance  
        # Progressive: Lower throttle (80%+) but only with very strong acceleration (early power run)
        wot_condition = (primary_wot_mask | 
                        (secondary_wot_mask & (strong_acceleration | maintain_rpm)) |
                        (progressive_wot_mask & strong_acceleration & (self.data['Engine Speed'] < 3000)))
        
        # Final valid conditions
        valid_conditions = wot_condition & acceleration_condition & (self.data['Engine Speed'] > AnalysisConstants.DEFAULT_MIN_RPM)
        
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
                    print(f"Gap bridged: {gap_count} invalid samples at index {i-gap_count}-{i-1} (RPM: {self.data.iloc[i-gap_count]['Engine Speed']:.0f}-{self.data.iloc[i-1]['Engine Speed']:.0f})")
                gap_count = 0  # Reset gap counter when we have valid data
            elif not valid and in_run:
                gap_count += 1
                if gap_count == 1:
                    # First invalid sample in potential gap
                    print(f"Gap detected at index {i} (RPM: {self.data.iloc[i]['Engine Speed']:.0f}, TPS: {self.data.iloc[i][tps_col]:.1f}%)")
                if gap_count > max_gap:
                    # End the run only after max_gap consecutive invalid samples
                    end_idx = i - gap_count - 1
                    print(f"Gap exceeded max_gap ({max_gap}): ending run with {gap_count} consecutive invalid samples at index {i-gap_count+1}-{i}")
                    
                    # Check if run meets minimum criteria
                    duration = self.data.iloc[end_idx]['Section Time'] - self.data.iloc[start_idx]['Section Time']
                    rpm_range = self.data.iloc[end_idx]['Engine Speed'] - self.data.iloc[start_idx]['Engine Speed']
                    
                    if duration >= min_duration and rpm_range >= min_rpm_range:
                        runs.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'duration': duration,
                            'rpm_range': rpm_range,
                            'start_rpm': self.data.iloc[start_idx]['Engine Speed'],
                            'end_rpm': self.data.iloc[end_idx]['Engine Speed']
                        })
                    
                    in_run = False
                    gap_count = 0
        
        # Handle case where run continues to end of data
        if in_run and start_idx is not None:
            end_idx = len(self.data) - 1 - gap_count
            duration = self.data.iloc[end_idx]['Section Time'] - self.data.iloc[start_idx]['Section Time']
            rpm_range = self.data.iloc[end_idx]['Engine Speed'] - self.data.iloc[start_idx]['Engine Speed']
            
            if duration >= min_duration and rpm_range >= min_rpm_range:
                runs.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration': duration,
                    'rpm_range': rpm_range,
                    'start_rpm': self.data.iloc[start_idx]['Engine Speed'],
                    'end_rpm': self.data.iloc[end_idx]['Engine Speed']
                })
        
        self.power_runs = runs
        
        # Debug information about detection criteria
        if len(runs) > 0:
            print(f"Found {len(runs)} valid power runs with enhanced detection:")
            for i, run in enumerate(runs):
                print(f"  Run {i+1}: {run['start_rpm']:.0f}-{run['end_rpm']:.0f} RPM, {run['duration']:.1f}s")
        else:
            # Provide diagnostic information when no runs found
            wot_samples = np.sum(wot_mask)
            accel_samples = np.sum(acceleration_condition)
            rpm_samples = np.sum(self.data['Engine Speed'] > AnalysisConstants.DEFAULT_MIN_RPM)
            combined_samples = np.sum(valid_conditions)
            
            print(f"No power runs found. Diagnostic info:")
            print(f"  Samples with TPS ≥ {throttle_threshold}%: {wot_samples}")
            print(f"  Samples with good acceleration: {accel_samples}")
            print(f"  Samples with RPM > {AnalysisConstants.DEFAULT_MIN_RPM}: {rpm_samples}")
            print(f"  Samples meeting all conditions: {combined_samples}")
            print(f"  Try lowering --throttle-threshold or --min-rpm-range")
        
        return runs
    
    def calculate_power_torque(self, run_data, debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        """
        Calculate power and torque from vehicle dynamics using RPM rate of change
        
        Args:
            run_data: DataFrame slice containing a power run
            debug_mode: If True, print debug information about calculations
            
        Returns:
            Tuple of (power_hp, torque_lbft) arrays
        """
        # Calculate vehicle acceleration from RPM change and gearing
        rpm = run_data['Engine Speed'].values
        time_data = run_data['Section Time'].values
        
        if debug_mode:
            print(f"\n=== DEBUG: Power Calculation for {len(rpm)} data points ===")
            print(f"RPM range: {rpm[0]:.0f} - {rpm[-1]:.0f}")
            print(f"Time range: {time_data[0]:.3f} - {time_data[-1]:.3f}s")
            print(f"First 10 RPM values: {rpm[:10]}")
            print(f"RPM differences: {np.diff(rpm[:10])}")
            print(f"Time steps: {np.diff(time_data[:10])}")
        
        # CRITICAL FIX: Ensure RPM is monotonic BEFORE any calculations
        # This prevents backwards torque lines and power curve artifacts
        rpm_monotonic, rpm_fixes = self._enforce_rpm_monotonicity(rpm)
        
        if rpm_fixes > 0:
            print(f"Pre-calculation monotonicity fix: corrected {rpm_fixes} RPM reversals")
        
        # Calculate wheel RPM from monotonic engine RPM
        wheel_rpm = rpm_monotonic / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        
        # Convert wheel RPM to vehicle speed (m/s)
        wheel_speed_rad_per_sec = wheel_rpm * 2 * np.pi / 60  # Convert RPM to rad/s
        tire_radius_m = self.vehicle_specs.tire_circumference_m / (2 * np.pi)
        vehicle_speed_ms = wheel_speed_rad_per_sec * tire_radius_m
        
        if debug_mode:
            print(f"Vehicle speed first 10 values: {vehicle_speed_ms[:10]}")
            print(f"Speed differences: {np.diff(vehicle_speed_ms[:10])}")
        
        # Calculate acceleration from vehicle speed change
        time_step = np.mean(np.diff(time_data))  # Average time step
        
        # Check if this is downsampled high-frequency data
        was_downsampled = hasattr(self, 'downsample_hz') and self.downsample_hz is not None
        
        # CRITICAL: First calculate RAW power/torque for artifact detection (using unsmoothed RPM)
        # This allows us to detect step artifacts before they're smoothed away
        
        # Calculate vehicle speed and acceleration using ORIGINAL (not smoothed) RPM for artifact detection
        wheel_rpm_raw = rpm_monotonic / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        wheel_speed_rad_per_sec_raw = wheel_rpm_raw * 2 * np.pi / 60
        vehicle_speed_ms_raw = wheel_speed_rad_per_sec_raw * tire_radius_m
        
        # Calculate raw acceleration for artifact detection
        acceleration_raw = np.gradient(vehicle_speed_ms_raw, time_step)
        
        # Calculate raw power and torque for artifact detection
        mass_kg = self.vehicle_specs.total_weight_kg
        force_n_raw = mass_kg * acceleration_raw
        rolling_resistance_force = mass_kg * AnalysisConstants.GRAVITY_MS2 * self.rolling_resistance
        aero_drag_raw = 0.5 * AnalysisConstants.AIR_DENSITY_KG_M3 * self.drag_coefficient * self.frontal_area * vehicle_speed_ms_raw**2
        total_force_raw = force_n_raw + rolling_resistance_force + aero_drag_raw
        
        power_watts_raw = total_force_raw * vehicle_speed_ms_raw
        power_hp_raw = power_watts_raw / AnalysisConstants.WATTS_TO_HP
        
        wheel_torque_nm_raw = (total_force_raw * self.vehicle_specs.tire_circumference_m) / (2 * np.pi)
        crank_torque_nm_raw = wheel_torque_nm_raw / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        torque_lbft_raw = crank_torque_nm_raw * AnalysisConstants.NM_TO_LBFT
        
        # Apply drivetrain efficiency to raw values
        power_hp_raw = power_hp_raw / self.drivetrain_efficiency
        
        if debug_mode:
            print(f"Raw power calculation completed for artifact detection")
        
        # NOW apply RPM smoothing for the final smoothed calculations
        # CRITICAL: Pre-smooth RPM data BEFORE calculating vehicle speed to prevent acceleration amplification
        # Even after downsampling and cleanup, micro-variations in RPM cause massive power oscillations
        if len(rpm) > 5:
            if was_downsampled:
                # For downsampled high-frequency data, apply aggressive RPM smoothing
                # The stair-step artifacts still cause acceleration spikes even after downsampling
                rpm_smooth_sigma = max(2.0, len(rpm) / 50.0)  # Scale with data length
                print(f"Aggressive RPM pre-smoothing for downsampled data: σ={rpm_smooth_sigma:.2f}")
            elif time_step < 0.01:
                # For high-frequency data, moderate RPM smoothing
                rpm_smooth_sigma = max(1.5, len(rpm) / 30.0)
                print(f"Moderate RPM pre-smoothing for high-frequency data: σ={rpm_smooth_sigma:.2f}")
            else:
                # For normal frequency data, light RPM smoothing
                rpm_smooth_sigma = max(1.0, len(rpm) / 40.0)
                print(f"Light RPM pre-smoothing: σ={rpm_smooth_sigma:.2f}")
            
            # Apply RPM smoothing to the already monotonic RPM
            rpm_smoothed = gaussian_filter1d(rpm_monotonic.astype(float), sigma=rpm_smooth_sigma)
            
            # CRITICAL: Enforce monotonicity to eliminate backwards torque lines
            # Even after smoothing, ensure RPM always increases for proper power calculations
            rpm_smoothed, post_smooth_fixes = self._enforce_rpm_monotonicity(rpm_smoothed, min_increment=1.0)
            
            if post_smooth_fixes > 0:
                print(f"Post-smoothing monotonicity check: {post_smooth_fixes} additional violations fixed")
            
            if debug_mode:
                rpm_variation_reduction = (np.std(np.diff(rpm)) - np.std(np.diff(rpm_smoothed))) / np.std(np.diff(rpm)) * 100
                print(f"RPM smoothing reduced variation by {rpm_variation_reduction:.1f}%")
                print(f"Original RPM std: {np.std(rpm):.1f}, Smoothed RPM std: {np.std(rpm_smoothed):.1f}")
        else:
            rpm_smoothed = rpm
        
        # Recalculate wheel RPM and vehicle speed using smoothed RPM
        wheel_rpm = rpm_smoothed / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        wheel_speed_rad_per_sec = wheel_rpm * 2 * np.pi / 60
        vehicle_speed_ms = wheel_speed_rad_per_sec * tire_radius_m
        
        # Calculate acceleration using sliding window approach to eliminate instantaneous spikes
        if was_downsampled:
            # For downsampled high-frequency data, use windowed acceleration calculation
            # This prevents the "kinky steps" caused by instantaneous acceleration spikes
            window_size = max(5, min(15, len(vehicle_speed_ms) // 20))  # Adaptive window size
            if window_size % 2 == 0:
                window_size += 1  # Ensure odd window size
            
            print(f"Using windowed acceleration calculation: window={window_size}")
            
            # Calculate acceleration using centered finite differences over multiple points
            acceleration = np.zeros_like(vehicle_speed_ms)
            half_window = window_size // 2
            
            for i in range(len(vehicle_speed_ms)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(vehicle_speed_ms), i + half_window + 1)
                
                if end_idx - start_idx > 2:  # Need at least 3 points
                    window_speed = vehicle_speed_ms[start_idx:end_idx]
                    window_time = time_data[start_idx:end_idx]
                    
                    # Linear regression over the window to get smooth acceleration
                    time_centered = window_time - window_time[len(window_time)//2]
                    
                    # Simple linear fit: speed = a*t + b, so acceleration = a
                    if len(time_centered) > 1 and np.std(time_centered) > 1e-6:
                        acceleration[i] = np.polyfit(time_centered, window_speed, 1)[0]
                    else:
                        acceleration[i] = 0
                else:
                    acceleration[i] = 0
            
            if debug_mode:
                print(f"Windowed acceleration stats: mean={np.mean(acceleration):.2f}, std={np.std(acceleration):.2f}")
                
        elif time_step < 0.01:
            if debug_mode:
                print(f"Using robust differentiation (dt={time_step:.4f}s)")
            
            # For high-frequency data, use central difference
            acceleration = np.gradient(vehicle_speed_ms, time_step)
        else:
            # For normal frequency data, use standard gradient
            acceleration = np.gradient(vehicle_speed_ms, time_step)
        
        if debug_mode:
            print(f"Raw acceleration first 10 values: {acceleration[:10]}")
            print(f"Acceleration range: {np.min(acceleration):.3f} to {np.max(acceleration):.3f} m/s²")
        
        # Apply additional smoothing if needed (enhanced for downsampled data)
        if len(acceleration) > 5 and self.smoothing_factor > 0:
            if was_downsampled:
                # Very light smoothing for downsampled data - preserve power characteristics
                sigma = self.smoothing_factor * min(0.8, len(acceleration) / 20.0)
                print(f"Light acceleration smoothing for downsampled data: σ={sigma:.2f}")
            elif time_step < 0.01:
                # For high-frequency data, use lighter additional smoothing since we pre-smoothed
                sigma = self.smoothing_factor * min(1.0, len(acceleration) / 15.0)
            else:
                # For normal frequency data, use standard smoothing
                sigma = self.smoothing_factor * min(2.0, len(acceleration) / 10.0)
            
            if sigma > 0.1:  # Only smooth if sigma is meaningful
                acceleration_smoothed = gaussian_filter1d(acceleration, sigma=sigma)
                if debug_mode:
                    print(f"Additional smoothing applied with sigma: {sigma:.3f}")
                    print(f"Acceleration smoothing reduced oscillation by {(np.std(np.diff(acceleration)) - np.std(np.diff(acceleration_smoothed))) / np.std(np.diff(acceleration)) * 100:.1f}%")
                    print(f"Final smoothed acceleration first 10 values: {acceleration_smoothed[:10]}")
                acceleration = acceleration_smoothed
            elif debug_mode:
                print(f"Skipping additional smoothing (sigma={sigma:.3f} too small)")
        elif debug_mode:
            print("No additional smoothing applied")
        
        # Calculate force required to accelerate the vehicle
        mass_kg = self.vehicle_specs.total_weight_kg
        force_n = mass_kg * acceleration
        
        # Add rolling resistance and aerodynamic drag using configurable parameters
        rolling_resistance_force = mass_kg * AnalysisConstants.GRAVITY_MS2 * self.rolling_resistance
        # Use the calculated vehicle speed for aerodynamic drag
        aero_drag = 0.5 * AnalysisConstants.AIR_DENSITY_KG_M3 * self.drag_coefficient * self.frontal_area * vehicle_speed_ms**2
        
        total_force = force_n + rolling_resistance_force + aero_drag
        
        # Calculate power at wheels
        power_watts = total_force * vehicle_speed_ms
        power_hp = power_watts / AnalysisConstants.WATTS_TO_HP  # Convert to HP
        
        # Calculate torque at crankshaft (using smoothed RPM for consistency)
        wheel_torque_nm = (total_force * self.vehicle_specs.tire_circumference_m) / (2 * np.pi)
        crank_torque_nm = wheel_torque_nm / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        torque_lbft = crank_torque_nm * AnalysisConstants.NM_TO_LBFT  # Convert to lb-ft
        
        # Apply configurable drivetrain efficiency (transmission and drivetrain losses)
        power_hp = power_hp / self.drivetrain_efficiency
        
        # For downsampled data, smooth power and torque INDEPENDENTLY to eliminate kinks
        # Then optionally apply HP-Torque correction to the final smoothed curves
        if len(power_hp) > 5:
            # CRITICAL: First, detect and smooth step artifacts using angle analysis
            # Use the processed power/torque with smoothed RPM to maintain array consistency
            artifact_smoothed_power, artifact_smoothed_torque = self._detect_and_smooth_rpm_step_artifacts(
                power_hp, torque_lbft, rpm_smoothed  # Use processed values to maintain array consistency
            )
            
            # Then apply general physics-aware smoothing to the artifact-corrected data
            smoothed_power_hp, smoothed_torque_lbft = self._apply_physics_aware_smoothing(
                artifact_smoothed_power, artifact_smoothed_torque, rpm_smoothed, run_data['Engine Speed'].values
            )
            
            # OPTIONALLY apply HP-Torque relationship correction to final smoothed curves
            if self.apply_hp_torque_correction:
                # Average the independently smoothed power with torque-derived power for balance
                torque_derived_power = (smoothed_torque_lbft * rpm_smoothed) / AnalysisConstants.HP_TORQUE_CROSSOVER_RPM
                # Use weighted average: 70% physics-based power, 30% torque-derived for crossover
                final_power_hp = 0.7 * smoothed_power_hp + 0.3 * torque_derived_power
                final_torque_lbft = smoothed_torque_lbft
                print("Applied blended HP-Torque correction to maintain smoothness")
            else:
                final_power_hp, final_torque_lbft = smoothed_power_hp, smoothed_torque_lbft
        else:
            final_power_hp, final_torque_lbft = power_hp, torque_lbft
        
        return final_power_hp, final_torque_lbft
    
    def validate_hp_torque_crossover(self, run_data, power_hp: np.ndarray, torque_lbft: np.ndarray) -> Dict[str, float]:
        """
        Validate that HP and Torque curves cross at 5252 RPM
        
        Args:
            run_data: DataFrame slice containing a power run
            power_hp: Power values in HP
            torque_lbft: Torque values in lb-ft
            
        Returns:
            Dict with validation results
        """
        rpm = run_data['Engine Speed'].values
        
        # Find the RPM where HP and Torque are closest (should be 5252)
        diff = np.abs(power_hp - torque_lbft)
        crossover_idx = np.argmin(diff)
        crossover_rpm = rpm[crossover_idx]
        crossover_value = (power_hp[crossover_idx] + torque_lbft[crossover_idx]) / 2
        
        # Calculate theoretical values at HP-Torque crossover RPM if in range
        crossover_rpm = AnalysisConstants.HP_TORQUE_CROSSOVER_RPM
        theoretical_crossover = None
        if min(rpm) <= crossover_rpm <= max(rpm):
            # Interpolate to find values at exactly crossover RPM
            hp_at_crossover = np.interp(crossover_rpm, rpm, power_hp)
            torque_at_crossover = np.interp(crossover_rpm, rpm, torque_lbft)
            theoretical_crossover = {
                'hp_at_crossover': hp_at_crossover,
                'torque_at_crossover': torque_at_crossover,
                'difference': abs(hp_at_crossover - torque_at_crossover),
                'average_value': (hp_at_crossover + torque_at_crossover) / 2
            }
        
        return {
            'actual_crossover_rpm': crossover_rpm,
            'actual_crossover_value': crossover_value,
            'rpm_error': abs(crossover_rpm - AnalysisConstants.HP_TORQUE_CROSSOVER_RPM),
            'theoretical_at_crossover': theoretical_crossover,
            'in_rpm_range': min(rpm) <= AnalysisConstants.HP_TORQUE_CROSSOVER_RPM <= max(rpm)
        }
    
    def plot_power_curves(self, save_path: Optional[str] = None, title: Optional[str] = None) -> None:
        """
        Create dyno-style power and torque curves on single plot
        
        Args:
            save_path: Optional path to save the plot
            title: Optional custom title for the plot
        """
        if not self.power_runs:
            raise ValueError("No power runs found. Call find_power_runs() first.")
        
        # Create figure with two subplots - main power plot and dataset coverage plot
        plt = _import_matplotlib()
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        fig, (ax1, ax_coverage) = plt.subplots(2, 1, figsize=(12, 10), 
                                              gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3})
        
        # Create second y-axis for torque on main plot
        ax2 = ax1.twinx()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.power_runs)))
        
        all_power = []
        all_torque = []
        all_rpm = []
        
        # Plot individual runs
        for i, run in enumerate(self.power_runs):
            start_idx = run['start_idx']
            end_idx = run['end_idx']
                
            run_data = self.data.iloc[start_idx:end_idx+1].copy()
            power_hp, torque_lbft = self.calculate_power_torque(run_data)
            rpm = run_data['Engine Speed'].values
            
            # Filter out unrealistic values
            valid_mask = (power_hp > 0) & (power_hp < AnalysisConstants.MAX_REALISTIC_POWER) & (torque_lbft > 0) & (torque_lbft < AnalysisConstants.MAX_REALISTIC_TORQUE)
            
            if np.any(valid_mask):
                # Power as solid lines
                ax1.plot(rpm[valid_mask], power_hp[valid_mask], 
                        color=colors[i], alpha=0.7, linewidth=2, linestyle='-',
                        label=f'Power Run {i+1}')
                # Torque as dotted lines
                ax2.plot(rpm[valid_mask], torque_lbft[valid_mask], 
                        color=colors[i], alpha=0.7, linewidth=2, linestyle=':',
                        label=f'Torque Run {i+1}')
                
                all_power.extend(power_hp[valid_mask])
                all_torque.extend(torque_lbft[valid_mask])
                all_rpm.extend(rpm[valid_mask])
        
        # Plot average if multiple runs
        if len(self.power_runs) > 1 and all_rpm:
            # Create interpolated average curves
            rpm_range = np.linspace(min(all_rpm), max(all_rpm), 100)
            avg_power = np.interp(rpm_range, all_rpm, all_power)
            avg_torque = np.interp(rpm_range, all_rpm, all_torque)
            
            # Apply smoothing to averaged curves if smoothing is enabled
            if self.smoothing_factor > 0:
                avg_power = gaussian_filter1d(avg_power, sigma=self.smoothing_factor)
                avg_torque = gaussian_filter1d(avg_torque, sigma=self.smoothing_factor)
                label_suffix = ' (Smoothed)'
            else:
                label_suffix = ''
            
            # Average power as thick solid line
            ax1.plot(rpm_range, avg_power, 'k-', linewidth=3, 
                    label=f'Avg Power{label_suffix}')
            # Average torque as thick dotted line
            ax2.plot(rpm_range, avg_torque, 'k:', linewidth=3, 
                    label=f'Avg Torque{label_suffix}')
        
        # Add reference line at 5252 RPM where HP and Torque curves should cross
        crossover_rpm = AnalysisConstants.HP_TORQUE_CROSSOVER_RPM
        if all_rpm and min(all_rpm) < crossover_rpm < max(all_rpm):
            ax1.axvline(x=crossover_rpm, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(crossover_rpm, ax1.get_ylim()[1] * 0.95, f'{crossover_rpm} RPM\n(HP=Torque)', 
                    ha='center', va='top', fontsize=9, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Set matching scales starting from 0 with same indexing
        # This ensures 100 HP appears at same level as 100 lb-ft
        if all_power and all_torque:
            # Find the maximum values from the already calculated data
            max_power_val = max(all_power) if all_power else 0
            max_torque_val = max(all_torque) if all_torque else 0
            
            # Set the same scale for both axes (0 to max of either value + 10% margin)
            max_scale = max(max_power_val, max_torque_val) * 1.1
            ax1.set_ylim(0, max_scale)
            ax2.set_ylim(0, max_scale)
        
        # Format plots
        ax1.set_xlabel('Engine Speed (RPM)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Power (HP)', fontsize=12, fontweight='bold', color='blue')
        ax2.set_ylabel('Torque (lb-ft)', fontsize=12, fontweight='bold', color='red')
        
        # Color the y-axis labels
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plot_title = title if title else f'Power and Torque Curves - {self.vehicle_specs.engine_displacement}L Engine'
        ax1.set_title(plot_title, fontsize=14, fontweight='bold')
        
        # Calculate peak values for display
        peak_info = ""
        if all_power and all_torque and all_rpm:
            max_power = max(all_power)
            max_torque = max(all_torque)
            max_power_idx = all_power.index(max_power)
            max_torque_idx = all_torque.index(max_torque)
            rpm_at_max_power = all_rpm[max_power_idx]
            rpm_at_max_torque = all_rpm[max_torque_idx]
            
            peak_info = (f"Peak Power: {max_power:.1f} HP @ {rpm_at_max_power:.0f} RPM | "
                        f"Peak Torque: {max_torque:.1f} lb-ft @ {rpm_at_max_torque:.0f} RPM")
            
            # Print peak values to console
            print(f"\nEstimated Peak Values:")
            print(f"Peak Power: {max_power:.1f} HP @ {rpm_at_max_power:.0f} RPM")
            print(f"Peak Torque: {max_torque:.1f} lb-ft @ {rpm_at_max_torque:.0f} RPM")
        
        # Add vehicle info and peak values text
        info_text = (f"Vehicle: {self.vehicle_specs.total_weight_kg:.0f}kg, "
                    f"Gear: {self.vehicle_specs.gear_ratio:.3f}:1, "
                    f"Final Drive: {self.vehicle_specs.final_drive:.1f}:1")
        
        if peak_info:
            full_info = f"{info_text}\n{peak_info}"
        else:
            full_info = info_text
            
        plt.figtext(0.02, -0.03, full_info, fontsize=10, style='italic')
        
        # Add environmental conditions (BAP and IAT) to bottom right
        env_conditions = []
        if self.power_runs:
            # Collect environmental data from all power runs
            all_bap = []
            all_iat = []
            
            for run in self.power_runs:
                start_idx = run['start_idx']
                end_idx = run['end_idx']
                run_data = self.data.iloc[start_idx:end_idx+1]
                
                # Collect BAP values
                if 'BAP' in run_data.columns:
                    bap_values = run_data['BAP'].dropna()
                    if len(bap_values) > 0:
                        all_bap.extend(bap_values.tolist())
                
                # Collect IAT values
                if 'IAT' in run_data.columns:
                    iat_values = run_data['IAT'].dropna()
                    if len(iat_values) > 0:
                        all_iat.extend(iat_values.tolist())
            
            # Format environmental conditions
            if all_bap:
                if len(set([round(x, 1) for x in all_bap])) == 1:
                    # All values are the same (within 0.1 kPa)
                    env_conditions.append(f"BAP: {np.mean(all_bap):.1f} kPa")
                else:
                    # Show range
                    env_conditions.append(f"BAP: {min(all_bap):.1f} - {max(all_bap):.1f} kPa")
            
            if all_iat:
                if len(set([round(x) for x in all_iat])) == 1:
                    # All values are the same (within 1°C)
                    env_conditions.append(f"IAT: {np.mean(all_iat):.0f}°C")
                else:
                    # Show range
                    env_conditions.append(f"IAT: {min(all_iat):.0f} - {max(all_iat):.0f}°C")
        
        # Add environmental conditions text to bottom right if we have any
        if env_conditions:
            env_text = "\n".join(env_conditions)
            plt.figtext(0.98, -0.03, env_text, fontsize=10, style='italic', 
                       ha='right', va='bottom')
        
        # Plot dataset coverage on the coverage subplot
        if self.data is not None:
            # Plot the entire dataset timeline as a gray line
            total_time = self.data['Section Time'].values
            ax_coverage.plot(total_time, [1] * len(total_time), 'lightgray', linewidth=2, alpha=0.5, label='Total Dataset')
            
            # Highlight power runs with colored bars
            for i, run in enumerate(self.power_runs):
                # Show full detected run in light color
                full_run_data = self.data.iloc[run['start_idx']:run['end_idx']+1]
                full_run_time = full_run_data['Section Time'].values
                ax_coverage.plot(full_run_time, [1] * len(full_run_time), color=colors[i], linewidth=2, alpha=0.3)
                
                # Show analysis run data in bold color
                analysis_run_data = self.data.iloc[run['start_idx']:run['end_idx']+1]
                analysis_run_time = analysis_run_data['Section Time'].values
                ax_coverage.plot(analysis_run_time, [1] * len(analysis_run_time), color=colors[i], linewidth=4, alpha=0.8, label=f'Run {i+1}')
            
            # Format coverage plot
            ax_coverage.set_xlabel('Time (seconds)', fontsize=10)
            ax_coverage.set_ylabel('Runs', fontsize=10)
            ax_coverage.set_ylim(0.5, 1.5)
            ax_coverage.set_title('Dataset Coverage - Power Run Locations', fontsize=11)
            ax_coverage.grid(True, alpha=0.3)
            ax_coverage.legend(loc='upper right', fontsize=9)
            
            # Remove y-tick labels since they're not meaningful
            ax_coverage.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_debug_output(self, rpm_increment: int = 250, time_increment: float = None) -> str:
        """
        Generate debug output with tabular power/torque data at RPM or time increments
        
        Args:
            rpm_increment: RPM increment for output rows (default: 250)
            time_increment: Time increment for output rows (if specified, overrides rpm_increment)
            
        Returns:
            Formatted string with tabular output of power/torque curves
        """
        if not self.power_runs:
            return "No power runs found for debug output."
        
        debug_output = ["Debug Mode - Tabular Power/Torque Output", "=" * 50, ""]
        
        # Collect all power/torque data from all runs
        all_data = []
        
        for i, run in enumerate(self.power_runs):
            start_idx = run['start_idx']
            end_idx = run['end_idx']
                
            run_data = self.data.iloc[start_idx:end_idx+1].copy()
            power_hp, torque_lbft = self.calculate_power_torque(run_data)
            rpm = run_data['Engine Speed'].values
            time_values = run_data['Section Time'].values
            
            # Filter out unrealistic values
            valid_mask = (power_hp > 0) & (power_hp < AnalysisConstants.MAX_REALISTIC_POWER) & (torque_lbft > 0) & (torque_lbft < AnalysisConstants.MAX_REALISTIC_TORQUE)
            
            if np.any(valid_mask):
                for j in range(len(rpm)):
                    if valid_mask[j]:
                        all_data.append({
                            'rpm': rpm[j],
                            'time': time_values[j],
                            'power': power_hp[j], 
                            'torque': torque_lbft[j],
                            'run': i + 1
                        })
        
        if not all_data:
            return "No valid data points found for debug output."
        
        if time_increment is not None:
            # Sort by time for time-based output
            all_data.sort(key=lambda x: x['time'])
            
            # Find time range
            min_time = min(d['time'] for d in all_data)
            max_time = max(d['time'] for d in all_data)
            
            # Create output at specified time increments
            current_time = min_time
            
            debug_output.extend([
                f"Time Range: {min_time:.3f} - {max_time:.3f} seconds",
                f"Output Increment: {time_increment:.3f} seconds",
                f"Total Data Points: {len(all_data)} from {len(self.power_runs)} run(s)",
                "",
                f"{'Time (s)':>8} | {'RPM':>6} | {'Power (HP)':>10} | {'Torque (lb-ft)':>12} | {'Run #':>6} | Notes",
                "-" * 75
            ])
            
            while current_time <= max_time:
                # Find the closest data point to this time
                closest_data = min(all_data, key=lambda x: abs(x['time'] - current_time))
                time_diff = abs(closest_data['time'] - current_time)
                
                # Only output if we have data within reasonable range of target time
                if time_diff <= time_increment * 0.6:  # Within 60% of increment
                    notes = ""
                    if time_diff > time_increment * 0.3:
                        notes = f"±{time_diff:.3f}s"
                    
                    # Check for HP=Torque crossover at crossover RPM
                    if abs(closest_data['rpm'] - AnalysisConstants.HP_TORQUE_CROSSOVER_RPM) <= 50:
                        hp_torque_diff = abs(closest_data['power'] - closest_data['torque'])
                        if hp_torque_diff < 5:
                            notes += " HP≈Torque" if notes else "HP≈Torque"
                        else:
                            notes += f" HP≠Torque({hp_torque_diff:.1f})" if notes else f"HP≠Torque({hp_torque_diff:.1f})"
                    
                    debug_output.append(
                        f"{current_time:>8.3f} | {closest_data['rpm']:>6.0f} | {closest_data['power']:>10.1f} | {closest_data['torque']:>12.1f} | "
                        f"{closest_data['run']:>6} | {notes}"
                    )
                
                current_time += time_increment
        else:
            # Sort by RPM for RPM-based output (original behavior)
            all_data.sort(key=lambda x: x['rpm'])
            
            # Find RPM range
            min_rpm = min(d['rpm'] for d in all_data)
            max_rpm = max(d['rpm'] for d in all_data)
            
            # Create output at specified RPM increments
            current_rpm = int(min_rpm / rpm_increment) * rpm_increment
            if current_rpm < min_rpm:
                current_rpm += rpm_increment
                
            debug_output.extend([
                f"RPM Range: {min_rpm:.0f} - {max_rpm:.0f}",
                f"Output Increment: {rpm_increment} RPM",
                f"Total Data Points: {len(all_data)} from {len(self.power_runs)} run(s)",
                "",
                f"{'RPM':>6} | {'Power (HP)':>10} | {'Torque (lb-ft)':>12} | {'Run #':>6} | Notes",
                "-" * 55
            ])
            
            while current_rpm <= max_rpm:
                # Find the closest data point to this RPM
                closest_data = min(all_data, key=lambda x: abs(x['rpm'] - current_rpm))
                rpm_diff = abs(closest_data['rpm'] - current_rpm)
                
                # Only output if we have data within reasonable range of target RPM
                if rpm_diff <= rpm_increment * 0.6:  # Within 60% of increment
                    notes = ""
                    if rpm_diff > rpm_increment * 0.3:
                        notes = f"±{rpm_diff:.0f} RPM"
                    
                    # Check for HP=Torque crossover at crossover RPM
                    if abs(current_rpm - AnalysisConstants.HP_TORQUE_CROSSOVER_RPM) <= rpm_increment / 2:
                        hp_torque_diff = abs(closest_data['power'] - closest_data['torque'])
                        if hp_torque_diff < 5:
                            notes += " HP≈Torque" if notes else "HP≈Torque"
                        else:
                            notes += f" HP≠Torque({hp_torque_diff:.1f})" if notes else f"HP≠Torque({hp_torque_diff:.1f})"
                    
                    debug_output.append(
                        f"{current_rpm:>6} | {closest_data['power']:>10.1f} | {closest_data['torque']:>12.1f} | "
                        f"{closest_data['run']:>6} | {notes}"
                    )
                
                current_rpm += rpm_increment
        
        # Add summary statistics
        all_power = [d['power'] for d in all_data]
        all_torque = [d['torque'] for d in all_data]
        all_rpm_vals = [d['rpm'] for d in all_data]
        
        max_power = max(all_power)
        max_torque = max(all_torque)
        max_power_idx = all_power.index(max_power)
        max_torque_idx = all_torque.index(max_torque)
        
        debug_output.extend([
            "",
            "Summary Statistics:",
            "-" * 20,
            f"Peak Power:  {max_power:.1f} HP @ {all_rpm_vals[max_power_idx]:.0f} RPM",
            f"Peak Torque: {max_torque:.1f} lb-ft @ {all_rpm_vals[max_torque_idx]:.0f} RPM",
            f"Power Range: {min(all_power):.1f} - {max(all_power):.1f} HP",
            f"Torque Range: {min(all_torque):.1f} - {max(all_torque):.1f} lb-ft"
        ])
        
        # Add environmental conditions if available
        env_data = []
        for run in self.power_runs:
            start_idx = run['start_idx']
            end_idx = run['end_idx']
            run_data = self.data.iloc[start_idx:end_idx+1]
            
            if 'BAP' in run_data.columns:
                bap_values = run_data['BAP'].dropna()
                if len(bap_values) > 0:
                    env_data.append(f"BAP: {np.mean(bap_values):.1f} kPa")
            
            if 'IAT' in run_data.columns:
                iat_values = run_data['IAT'].dropna()
                if len(iat_values) > 0:
                    env_data.append(f"IAT: {np.mean(iat_values):.0f}°C")
            break  # Only need one run for env conditions
        
        if env_data:
            debug_output.extend(["", "Environmental Conditions:"] + env_data)
        
        return "\n".join(debug_output)

    def generate_report(self) -> str:
        """Generate a text report of the analysis"""
        if not self.power_runs:
            return "No power runs found to report."
        
        report = ["Power and Torque Analysis Report", "=" * 40, ""]
        
        # Vehicle specs
        report.extend([
            "Vehicle Specifications:",
            f"  Total Weight: {self.vehicle_specs.total_weight_kg:.0f} kg",
            f"  Engine: {self.vehicle_specs.engine_displacement}L {self.vehicle_specs.cylinders}-cylinder",
            f"  Gear Ratio: {self.vehicle_specs.gear_ratio:.3f}:1",
            f"  Final Drive: {self.vehicle_specs.final_drive:.1f}:1",
            f"  Tires: {self.vehicle_specs.tire_width}/{self.vehicle_specs.tire_sidewall}R{self.vehicle_specs.tire_diameter}",
            ""
        ])
        
        # Run summary
        report.extend([
            f"Found {len(self.power_runs)} valid power runs:",
            ""
        ])
        
        for i, run in enumerate(self.power_runs):
            start_idx = run['start_idx']
            end_idx = run['end_idx']
                
            run_data = self.data.iloc[start_idx:end_idx+1]
            power_hp, torque_lbft = self.calculate_power_torque(run_data)
            
            valid_mask = (power_hp > 0) & (power_hp < AnalysisConstants.MAX_REALISTIC_POWER) & (torque_lbft > 0) & (torque_lbft < AnalysisConstants.MAX_REALISTIC_TORQUE)
            
            if np.any(valid_mask):
                max_power = np.max(power_hp[valid_mask])
                max_torque = np.max(torque_lbft[valid_mask])
                rpm_at_max_power = run_data['Engine Speed'].values[valid_mask][np.argmax(power_hp[valid_mask])]
                rpm_at_max_torque = run_data['Engine Speed'].values[valid_mask][np.argmax(torque_lbft[valid_mask])]
                
                # Validate HP-Torque crossover
                validation = self.validate_hp_torque_crossover(run_data, power_hp[valid_mask], torque_lbft[valid_mask])
                
                report.extend([
                    f"Run {i+1}:",
                    f"  RPM Range: {run['start_rpm']:.0f} - {run['end_rpm']:.0f}",
                    f"  Duration: {run['duration']:.1f} seconds",
                    f"  Max Power: {max_power:.1f} HP @ {rpm_at_max_power:.0f} RPM",
                    f"  Max Torque: {max_torque:.1f} lb-ft @ {rpm_at_max_torque:.0f} RPM",
                ])
                
                # Add crossover validation if in range
                if validation['in_rpm_range']:
                    report.extend([
                        f"  HP=Torque Crossover: {validation['actual_crossover_rpm']:.0f} RPM (error: {validation['rpm_error']:.0f} RPM)",
                        f"  Values at {AnalysisConstants.HP_TORQUE_CROSSOVER_RPM} RPM: HP={validation['theoretical_at_crossover']['hp_at_crossover']:.1f}, Torque={validation['theoretical_at_crossover']['torque_at_crossover']:.1f} (diff: {validation['theoretical_at_crossover']['difference']:.1f})"
                    ])
                else:
                    report.append(f"  HP=Torque Crossover: Not in RPM range ({AnalysisConstants.HP_TORQUE_CROSSOVER_RPM} RPM outside {run['start_rpm']:.0f}-{run['end_rpm']:.0f})")
                
                # Add environmental conditions for this run
                start_bap = run_data['BAP'].iloc[0] if 'BAP' in run_data.columns else None
                end_bap = run_data['BAP'].iloc[-1] if 'BAP' in run_data.columns else None
                start_iat = run_data['IAT'].iloc[0] if 'IAT' in run_data.columns else None
                end_iat = run_data['IAT'].iloc[-1] if 'IAT' in run_data.columns else None
                
                if start_bap is not None:
                    if end_bap is not None and abs(end_bap - start_bap) > 0.1:
                        report.append(f"  Barometric Pressure: {start_bap:.1f} - {end_bap:.1f} kPa")
                    else:
                        report.append(f"  Barometric Pressure: {start_bap:.1f} kPa")
                
                if start_iat is not None:
                    if end_iat is not None and abs(end_iat - start_iat) > 1:
                        report.append(f"  Intake Air Temperature: {start_iat:.0f} - {end_iat:.0f}°C")
                    else:
                        report.append(f"  Intake Air Temperature: {start_iat:.0f}°C")
                
                report.append("")
        
        return "\n".join(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze ECU log data for power and torque curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (Honda EP3 Type R specs)
  python power_analyzer.py log.csv
  
  # Custom vehicle specs
  python power_analyzer.py log.csv --weight 1134 --final-drive 4.3 --gear-ratio 1.000
  
  # Different tire size
  python power_analyzer.py log.csv --tire-width 225 --tire-sidewall 45 --tire-diameter 17
  
  # Turbocharged setup with efficiency adjustments
  python power_analyzer.py log.csv --drivetrain-efficiency 0.82 --rolling-resistance 0.012
        """
    )
    
    # Required arguments
    parser.add_argument('csv_file', help='Path to CSV log file')
    
    # Vehicle weight
    parser.add_argument('--weight', type=float, default=998, 
                       help='Vehicle curb weight in kg (default: 998)')
    parser.add_argument('--occupant', type=float, default=91, 
                       help='Occupant + gear weight in kg (default: 91)')
    
    # Drivetrain
    parser.add_argument('--final-drive', type=float, default=4.7, 
                       help='Final drive ratio (default: 4.7)')
    parser.add_argument('--gear-ratio', type=float, default=1.212, 
                       help='Current gear ratio (default: 1.212 for EP3 Type R 4th)')
    
    # Engine specs
    parser.add_argument('--displacement', type=float, default=2.0, 
                       help='Engine displacement in liters (default: 2.0)')
    parser.add_argument('--cylinders', type=int, default=4, 
                       help='Number of cylinders (default: 4)')
    
    # Tire specifications
    parser.add_argument('--tire-width', type=int, default=195, 
                       help='Tire width in mm (default: 195)')
    parser.add_argument('--tire-sidewall', type=int, default=50, 
                       help='Tire sidewall ratio in %% (default: 50)')
    parser.add_argument('--tire-diameter', type=int, default=15, 
                       help='Wheel diameter in inches (default: 15)')
    
    # Performance parameters
    parser.add_argument('--drivetrain-efficiency', type=float, default=0.85, 
                       help='Drivetrain efficiency factor (default: 0.85)')
    parser.add_argument('--rolling-resistance', type=float, default=0.015, 
                       help='Rolling resistance coefficient (default: 0.015)')
    parser.add_argument('--drag-coefficient', type=float, default=0.35, 
                       help='Aerodynamic drag coefficient (default: 0.35)')
    parser.add_argument('--frontal-area', type=float, default=2.5, 
                       help='Vehicle frontal area in m² (default: 2.5)')
    parser.add_argument('--smoothing-factor', type=float, default=AnalysisConstants.DEFAULT_SMOOTHING_FACTOR, 
                       help='Data smoothing factor - 0 disables, higher values = more smoothing (default: 2.5)')
    parser.add_argument('--no-hp-torque-correction', action='store_true', 
                       help='Disable HP-Torque relationship correction (HP = Torque * RPM / 5252)')
    parser.add_argument('--no-rpm-filtering', action='store_true', 
                       help='Disable RPM data filtering (keeps duplicate/bad ECU readings)')
    parser.add_argument('--max-gap', type=int, default=AnalysisConstants.DEFAULT_MAX_GAP, 
                       help='Maximum consecutive invalid samples allowed before ending a power run (default: 5)')
    parser.add_argument('--downsample-hz', type=float, default=None, 
                       help='Downsample data to specified frequency in Hz (e.g., 50 for high-frequency ECU logs)')
    
    # Analysis parameters
    parser.add_argument('--min-duration', type=float, default=AnalysisConstants.DEFAULT_MIN_DURATION, 
                       help='Minimum power run duration in seconds (default: 1.0)')
    parser.add_argument('--min-rpm-range', type=float, default=AnalysisConstants.DEFAULT_MIN_RPM_RANGE, 
                       help='Minimum RPM range for valid run (default: 2500)')
    parser.add_argument('--throttle-threshold', type=float, default=AnalysisConstants.DEFAULT_THROTTLE_THRESHOLD, 
                       help='Minimum throttle %% for WOT detection (default: 96)')
    
    # Output options
    parser.add_argument('--out', help='Output file for plot (optional)')
    parser.add_argument('--title', help='Custom title for the plot')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plot')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode - output tabular data instead of graph')
    parser.add_argument('--debug-rpm-increment', type=int, default=250, 
                       help='RPM increment for debug mode output (default: 250)')
    parser.add_argument('--debug-time-increment', type=float, default=None, 
                       help='Time increment for debug mode output in seconds (overrides rpm increment if specified)')
    
    args = parser.parse_args()
    
    # Create vehicle specs from arguments
    vehicle_specs = VehicleSpecs(
        weight_kg=args.weight,
        occupant_weight_kg=args.occupant,
        final_drive=args.final_drive,
        gear_ratio=args.gear_ratio,
        tire_width=args.tire_width,
        tire_sidewall=args.tire_sidewall,
        tire_diameter=args.tire_diameter,
        engine_displacement=args.displacement,
        cylinders=args.cylinders
    )
    
    try:
        # Create analyzer with all configurable parameters
        analyzer = PowerAnalyzer(
            vehicle_specs=vehicle_specs,
            drivetrain_efficiency=args.drivetrain_efficiency,
            rolling_resistance=args.rolling_resistance,
            drag_coefficient=args.drag_coefficient,
            frontal_area=args.frontal_area,
            smoothing_factor=args.smoothing_factor,
            apply_hp_torque_correction=not args.no_hp_torque_correction,
            filter_rpm_data=not args.no_rpm_filtering,
            max_gap=args.max_gap,
            downsample_hz=args.downsample_hz
        )
        
        analyzer.load_data(args.csv_file)
        analyzer.find_power_runs(
            min_duration=args.min_duration,
            min_rpm_range=args.min_rpm_range,
            throttle_threshold=args.throttle_threshold
        )
        
        if analyzer.power_runs:
            if args.debug:
                # Debug mode - output tabular data
                print(analyzer.generate_debug_output(args.debug_rpm_increment, args.debug_time_increment))
            else:
                # Normal mode - generate report and plot
                print(analyzer.generate_report())
                if not args.no_plot:
                    analyzer.plot_power_curves(args.out, args.title)
        else:
            print("No valid power runs found in the data.")
            print(f"Try adjusting --throttle-threshold (currently {args.throttle_threshold}%)")
            print(f"or --min-duration (currently {args.min_duration}s)")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
=============================================================================
CSV FILE REQUIREMENTS FOR POWER_PREDICTOR.PY
=============================================================================

REQUIRED COLUMNS (Script will not work without these):
--------------------------------------------------------
1. 'Section Time' - Time stamps for each data point (seconds)
   - Must be continuously increasing values
   - Used for calculating acceleration and power runs duration
   
2. 'Engine Speed' - Engine RPM values
   - Must be numeric values > 0
   - Used for all power/torque calculations and run detection
   
3. 'TPS (Main)' OR 'TPS 2(Main)' - Throttle Position Sensor percentage
   - Must be numeric values 0-100
   - Used to detect Wide Open Throttle (WOT) conditions
   - Script looks for 'TPS (Main)' first, falls back to 'TPS 2(Main)'

CSV FORMAT REQUIREMENTS:
------------------------
- Header must be in row 2 (0-indexed row 1)
- Row 3 should contain units (will be skipped)
- Data starts from row 4
- Column names will be cleaned (quotes and whitespace removed)
- File format: CSV with standard comma separation

NICE-TO-HAVE COLUMNS (Enhance analysis but not required):
---------------------------------------------------------
4. 'Driven Wheel Speed' - Wheel speed data
   - Provides additional validation for calculations
   
5. 'Acceleration' - Direct acceleration measurements
   - Can supplement calculated acceleration values
   
6. 'MAP' - Manifold Absolute Pressure
   - Useful for engine load analysis
   
7. 'BAP' - Barometric Pressure (kPa)
   - Used for environmental condition reporting
   - Shows atmospheric pressure during runs
   
8. 'IAT' - Intake Air Temperature (°C)
   - Used for environmental condition reporting
   - Important for density altitude corrections
   
9. 'Lambda 1' OR 'Lambda Avg' - Air/Fuel ratio data
   - Can be displayed on graphs for tuning analysis
   - Shows if engine is running rich/lean during power runs

POWER RUN DETECTION CRITERIA:
-----------------------------
The script automatically detects power runs based on:
- Throttle position >= threshold (default 96%, configurable)
- Engine RPM steadily increasing (> 5 RPM per sample)
- Engine RPM > 2000 (configurable)
- Minimum duration of 2.0 seconds (configurable)
- Minimum RPM range of 2500 RPM (configurable)

EXAMPLE CSV STRUCTURE:
----------------------
Line 1: [File info/metadata - ignored]
Line 2: "Section Time","Engine Speed","TPS (Main)","BAP","IAT",...
Line 3: "s","RPM","%","kPa","°C",...
Line 4: 0.000,32500,99.2,101.3,25,...
Line 5: 0.010,3515,99.5,101.3,25,...
...

TROUBLESHOOTING:
----------------
- If no power runs found, try lowering --throttle-threshold
- If runs are too short, adjust --min-duration or --min-rpm-range
- If RPM data looks erratic, keep --filter-rpm-data enabled (default)
"""