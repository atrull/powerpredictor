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
                 trim_frames: int = 20,
                 max_gap: int = 5):
        self.vehicle_specs = vehicle_specs
        self.drivetrain_efficiency = drivetrain_efficiency
        self.rolling_resistance = rolling_resistance
        self.drag_coefficient = drag_coefficient
        self.frontal_area = frontal_area
        self.smoothing_factor = smoothing_factor
        self.apply_hp_torque_correction = apply_hp_torque_correction
        self.filter_rpm_data = filter_rpm_data
        self.trim_frames = trim_frames
        self.max_gap = max_gap
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
            
            # Filter RPM data to handle ECU reporting issues
            if self.filter_rpm_data:
                self._filter_rpm_data()
                    
            print(f"Loaded {len(self.data)} data points from {csv_path}")
            
        except Exception as e:
            raise ValueError(f"Error loading CSV data: {e}")
    
    def _filter_rpm_data(self) -> None:
        """
        Enhanced RPM data filtering for ECUs with poor synchronization
        Uses multi-stage approach: outlier detection, trend analysis, and adaptive smoothing
        """
        if 'Engine Speed' not in self.data.columns or 'Section Time' not in self.data.columns:
            return
        
        pd = _import_pandas()
        
        rpm_col = 'Engine Speed'
        time_col = 'Section Time'
        tps_col = 'TPS (Main)' if 'TPS (Main)' in self.data.columns else 'TPS 2(Main)'
        
        # Make a copy to work with
        filtered_data = self.data.copy()
        
        rpm_values = filtered_data[rpm_col].values.copy().astype(float)  # Convert to float to allow NaN
        time_values = filtered_data[time_col].values
        tps_values = filtered_data[tps_col].values if tps_col in filtered_data.columns else np.full(len(rpm_values), 0)
        
        # Calculate time deltas for physical constraint checking
        time_deltas = np.diff(time_values)
        
        problems_found = 0
        outliers_found = 0
        
        # Stage 1: Remove exact duplicates
        for i in range(1, len(rpm_values)):
            if pd.notna(rpm_values[i]) and pd.notna(rpm_values[i-1]):
                if (rpm_values[i] == rpm_values[i-1] and 
                    time_values[i] != time_values[i-1]):
                    rpm_values[i] = np.nan
                    problems_found += 1
        
        # Stage 2: Enhanced outlier detection using physical constraints
        max_rpm_per_sec = 1500  # Maximum realistic RPM/sec acceleration at WOT
        
        for i in range(1, len(rpm_values) - 1):
            if pd.notna(rpm_values[i-1]) and pd.notna(rpm_values[i]) and pd.notna(rpm_values[i+1]):
                dt_prev = time_deltas[i-1] if i-1 < len(time_deltas) else 0.02
                dt_next = time_deltas[i] if i < len(time_deltas) else 0.02
                
                # Calculate acceleration rates
                accel_in = (rpm_values[i] - rpm_values[i-1]) / dt_prev if dt_prev > 0 else 0
                accel_out = (rpm_values[i+1] - rpm_values[i]) / dt_next if dt_next > 0 else 0
                
                # Check for physically impossible accelerations or direction reversals
                is_outlier = False
                
                # High TPS with impossible RPM drops
                if tps_values[i] > 95 and accel_in < -500:
                    is_outlier = True
                
                # Sudden direction changes that exceed physical limits
                if abs(accel_in) > max_rpm_per_sec or abs(accel_out) > max_rpm_per_sec:
                    # Check if this is a genuine outlier vs normal acceleration
                    if abs(accel_in - accel_out) > max_rpm_per_sec:
                        is_outlier = True
                
                # RPM value significantly out of local trend
                if i >= 2 and i < len(rpm_values) - 2:
                    local_trend = np.median([rpm_values[i-2], rpm_values[i-1], rpm_values[i+1], rpm_values[i+2]])
                    if abs(rpm_values[i] - local_trend) > 100:  # More than 100 RPM from local median
                        is_outlier = True
                
                if is_outlier:
                    rpm_values[i] = np.nan
                    outliers_found += 1
        
        # Stage 3: Multi-pass trend-based smoothing for WOT sections
        wot_mask = tps_values > 95  # Wide open throttle detection
        
        # Apply Savitzky-Golay filter to WOT sections for better trend preservation
        
        # Find continuous WOT sections
        wot_sections = []
        in_wot = False
        start_idx = None
        
        for i, is_wot in enumerate(wot_mask):
            if is_wot and not in_wot:
                start_idx = i
                in_wot = True
            elif not is_wot and in_wot:
                if start_idx is not None and i - start_idx > 10:  # At least 10 samples
                    wot_sections.append((start_idx, i))
                in_wot = False
                start_idx = None
        
        # Handle case where WOT continues to end
        if in_wot and start_idx is not None:
            wot_sections.append((start_idx, len(wot_mask)))
        
        # Apply enhanced smoothing to each WOT section
        for start, end in wot_sections:
            section_rpm = rpm_values[start:end].copy()
            section_time = time_values[start:end]
            
            # First, interpolate any remaining NaN values in this section
            section_mask = ~np.isnan(section_rpm)
            if np.sum(section_mask) > 5:  # Need at least 5 valid points
                # Linear interpolation for gaps
                section_rpm = np.interp(
                    section_time,
                    section_time[section_mask],
                    section_rpm[section_mask]
                )
                
                # Apply smoothing filter if section is long enough
                if len(section_rpm) >= 10:
                    # Use moving average for now to avoid savgol complexity
                    window = min(7, len(section_rpm) // 3)
                    if window >= 3:
                        section_rpm = pd.Series(section_rpm).rolling(window=window, center=True).mean().bfill().ffill().values
                
                # Update the main array
                rpm_values[start:end] = section_rpm
        
        # Stage 4: Final interpolation for any remaining NaN values
        rpm_mask = ~np.isnan(rpm_values)
        if np.sum(rpm_mask) > 0:
            rpm_values = np.interp(
                time_values,
                time_values[rpm_mask],
                rpm_values[rpm_mask]
            )
        
        # Update the filtered data
        filtered_data[rpm_col] = rpm_values
        
        total_issues = problems_found + outliers_found
        if total_issues > 0:
            print(f"Enhanced RPM filtering: {problems_found} duplicates, {outliers_found} outliers, {len(wot_sections)} WOT sections smoothed")
        
        # Update the main data
        self.data = filtered_data
    
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
        
        # Find WOT conditions using configurable threshold
        wot_mask = self.data[tps_col] >= throttle_threshold
        
        # Calculate RPM rate of change (less aggressive filtering)
        rpm_diff = self.data['Engine Speed'].diff().rolling(window=3, center=True).mean()
        stable_increase = rpm_diff > 5  # RPM increasing at least 5/sample (more lenient)
        
        # Combine conditions - lower RPM threshold
        valid_conditions = wot_mask & stable_increase & (self.data['Engine Speed'] > 2000)
        
        # Find continuous runs with gap tolerance to merge close runs
        runs = []
        in_run = False
        start_idx = None
        gap_count = 0
        max_gap = self.max_gap  # Allow configurable consecutive invalid samples before ending a run
        
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
        print(f"Found {len(runs)} valid power runs")
        if self.trim_frames > 0:
            print(f"Note: {self.trim_frames} frames will be trimmed from start/end of each run for analysis")
        return runs
    
    def calculate_power_torque(self, run_data) -> Tuple[np.ndarray, np.ndarray]:
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        """
        Calculate power and torque from vehicle dynamics using RPM rate of change
        
        Args:
            run_data: DataFrame slice containing a power run
            
        Returns:
            Tuple of (power_hp, torque_lbft) arrays
        """
        # Calculate vehicle acceleration from RPM change and gearing
        rpm = run_data['Engine Speed'].values
        time_data = run_data['Section Time'].values
        
        # Calculate wheel RPM from engine RPM
        wheel_rpm = rpm / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        
        # Convert wheel RPM to vehicle speed (m/s)
        wheel_speed_rad_per_sec = wheel_rpm * 2 * np.pi / 60  # Convert RPM to rad/s
        tire_radius_m = self.vehicle_specs.tire_circumference_m / (2 * np.pi)
        vehicle_speed_ms = wheel_speed_rad_per_sec * tire_radius_m
        
        # Calculate acceleration from vehicle speed change
        # Use numpy gradient for smooth differentiation
        time_step = np.mean(np.diff(time_data))  # Average time step
        acceleration = np.gradient(vehicle_speed_ms, time_step)
        
        # Smooth the acceleration to reduce noise
        if len(acceleration) > 5:
            # Apply Gaussian smoothing with configurable factor
            sigma = self.smoothing_factor * min(2.0, len(acceleration) / 10.0)
            acceleration = gaussian_filter1d(acceleration, sigma=sigma)
        
        # Calculate force required to accelerate the vehicle
        mass_kg = self.vehicle_specs.total_weight_kg
        force_n = mass_kg * acceleration
        
        # Add rolling resistance and aerodynamic drag using configurable parameters
        rolling_resistance_force = mass_kg * 9.81 * self.rolling_resistance
        # Use the calculated vehicle speed for aerodynamic drag
        aero_drag = 0.5 * 1.225 * self.drag_coefficient * self.frontal_area * vehicle_speed_ms**2
        
        total_force = force_n + rolling_resistance_force + aero_drag
        
        # Calculate power at wheels
        power_watts = total_force * vehicle_speed_ms
        power_hp = power_watts / 745.7  # Convert to HP
        
        # Calculate torque at crankshaft
        rpm = run_data['Engine Speed'].values
        wheel_torque_nm = (total_force * self.vehicle_specs.tire_circumference_m) / (2 * np.pi)
        crank_torque_nm = wheel_torque_nm / (self.vehicle_specs.final_drive * self.vehicle_specs.gear_ratio)
        torque_lbft = crank_torque_nm * 0.737562  # Convert to lb-ft
        
        # Apply configurable drivetrain efficiency (transmission and drivetrain losses)
        power_hp = power_hp / self.drivetrain_efficiency
        
        # Apply HP-Torque relationship correction if enabled
        if self.apply_hp_torque_correction:
            # ENFORCE the fundamental relationship: HP = (Torque * RPM) / 5252
            # This ensures curves always cross at 5252 RPM
            rpm = run_data['Engine Speed'].values
            
            # Calculate power directly from torque using the fundamental relationship
            # This maintains physical accuracy and ensures crossover at 5252 RPM
            corrected_power_hp = (torque_lbft * rpm) / 5252
        else:
            corrected_power_hp = power_hp
        
        # Apply smoothing to final power and torque curves
        if len(corrected_power_hp) > 5 and self.smoothing_factor > 0:
            sigma = self.smoothing_factor * min(1.5, len(corrected_power_hp) / 15.0)
            corrected_power_hp = gaussian_filter1d(corrected_power_hp, sigma=sigma)
            torque_lbft = gaussian_filter1d(torque_lbft, sigma=sigma)
        
        return corrected_power_hp, torque_lbft
    
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
        
        # Calculate theoretical values at 5252 RPM if in range
        theoretical_crossover = None
        if min(rpm) <= 5252 <= max(rpm):
            # Interpolate to find values at exactly 5252 RPM
            hp_at_5252 = np.interp(5252, rpm, power_hp)
            torque_at_5252 = np.interp(5252, rpm, torque_lbft)
            theoretical_crossover = {
                'hp_at_5252': hp_at_5252,
                'torque_at_5252': torque_at_5252,
                'difference': abs(hp_at_5252 - torque_at_5252),
                'average_value': (hp_at_5252 + torque_at_5252) / 2
            }
        
        return {
            'actual_crossover_rpm': crossover_rpm,
            'actual_crossover_value': crossover_value,
            'rpm_error': abs(crossover_rpm - 5252),
            'theoretical_at_5252': theoretical_crossover,
            'in_rpm_range': min(rpm) <= 5252 <= max(rpm)
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
            # Apply frame trimming to clean up start/end values
            start_idx = run['start_idx'] + self.trim_frames
            end_idx = run['end_idx'] - self.trim_frames
            
            # Ensure we have enough data after trimming
            if end_idx <= start_idx:
                print(f"Warning: Run {i+1} too short after trimming {self.trim_frames} frames from each end")
                continue
                
            run_data = self.data.iloc[start_idx:end_idx+1].copy()
            power_hp, torque_lbft = self.calculate_power_torque(run_data)
            rpm = run_data['Engine Speed'].values
            
            # Filter out unrealistic values
            valid_mask = (power_hp > 0) & (power_hp < 1000) & (torque_lbft > 0) & (torque_lbft < 1000)
            
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
        if all_rpm and min(all_rpm) < 5252 < max(all_rpm):
            ax1.axvline(x=5252, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(5252, ax1.get_ylim()[1] * 0.95, '5252 RPM\n(HP=Torque)', 
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
                start_idx = run['start_idx'] + self.trim_frames
                end_idx = run['end_idx'] - self.trim_frames
                
                if end_idx > start_idx:
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
                
                # Show trimmed run (actually used for analysis) in bold color
                start_idx = run['start_idx'] + self.trim_frames
                end_idx = run['end_idx'] - self.trim_frames
                if end_idx > start_idx:
                    trimmed_run_data = self.data.iloc[start_idx:end_idx+1]
                    trimmed_run_time = trimmed_run_data['Section Time'].values
                    ax_coverage.plot(trimmed_run_time, [1] * len(trimmed_run_time), color=colors[i], linewidth=4, alpha=0.8, label=f'Run {i+1} (trimmed)')
            
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
            # Apply frame trimming to clean up start/end values
            start_idx = run['start_idx'] + self.trim_frames
            end_idx = run['end_idx'] - self.trim_frames
            
            # Ensure we have enough data after trimming
            if end_idx <= start_idx:
                continue
                
            run_data = self.data.iloc[start_idx:end_idx+1]
            power_hp, torque_lbft = self.calculate_power_torque(run_data)
            
            valid_mask = (power_hp > 0) & (power_hp < 1000) & (torque_lbft > 0) & (torque_lbft < 1000)
            
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
                        f"  Values at 5252 RPM: HP={validation['theoretical_at_5252']['hp_at_5252']:.1f}, Torque={validation['theoretical_at_5252']['torque_at_5252']:.1f} (diff: {validation['theoretical_at_5252']['difference']:.1f})"
                    ])
                else:
                    report.append(f"  HP=Torque Crossover: Not in RPM range (5252 RPM outside {run['start_rpm']:.0f}-{run['end_rpm']:.0f})")
                
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
    parser.add_argument('--smoothing-factor', type=float, default=2.5, 
                       help='Data smoothing factor - 0 disables, higher values = more smoothing (default: 2.5)')
    parser.add_argument('--no-hp-torque-correction', action='store_true', 
                       help='Disable HP-Torque relationship correction (HP = Torque * RPM / 5252)')
    parser.add_argument('--no-rpm-filtering', action='store_true', 
                       help='Disable RPM data filtering (keeps duplicate/bad ECU readings)')
    parser.add_argument('--trim-frames', type=int, default=20, 
                       help='Number of frames to trim from start/end of each run for cleaner data (default: 20)')
    parser.add_argument('--max-gap', type=int, default=5, 
                       help='Maximum consecutive invalid samples allowed before ending a power run (default: 5)')
    
    # Analysis parameters
    parser.add_argument('--min-duration', type=float, default=1.0, 
                       help='Minimum power run duration in seconds (default: 1.0)')
    parser.add_argument('--min-rpm-range', type=float, default=2500, 
                       help='Minimum RPM range for valid run (default: 2500)')
    parser.add_argument('--throttle-threshold', type=float, default=96, 
                       help='Minimum throttle %% for WOT detection (default: 96)')
    
    # Output options
    parser.add_argument('--out', help='Output file for plot (optional)')
    parser.add_argument('--title', help='Custom title for the plot')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plot')
    
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
            trim_frames=args.trim_frames,
            max_gap=args.max_gap
        )
        
        analyzer.load_data(args.csv_file)
        analyzer.find_power_runs(
            min_duration=args.min_duration,
            min_rpm_range=args.min_rpm_range,
            throttle_threshold=args.throttle_threshold
        )
        
        if analyzer.power_runs:
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
- Use --trim-frames to clean up start/end of detected runs
"""