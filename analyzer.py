"""
Main PowerAnalyzer class that orchestrates all modules
"""

import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from constants import AnalysisConstants
from vehicle_specs import VehicleSpecs
from data_loader import DataLoader
from data_processing import DataProcessor
from power_calculator import PowerCalculator
from run_detector import RunDetector
from plotting import Plotter

# Lazy imports for heavy dependencies
def _import_pandas():
    import pandas as pd
    return pd


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
        
        # Initialize sub-modules
        self.data_loader = DataLoader(downsample_hz, filter_rpm_data)
        self.data_processor = DataProcessor(max_gap)
        self.power_calculator = PowerCalculator(
            vehicle_specs, drivetrain_efficiency, rolling_resistance,
            drag_coefficient, frontal_area, smoothing_factor,
            apply_hp_torque_correction, downsample_hz
        )
        self.run_detector = RunDetector(max_gap)
        self.plotter = Plotter(vehicle_specs, smoothing_factor)
        
        self.data = None
        self.power_runs = []
        
    def load_data(self, csv_path: str) -> None:
        """Load CSV data and clean column names"""
        self.data = self.data_loader.load_data(csv_path)
        
        # Filter RPM data AFTER downsampling to handle ECU reporting issues on final dataset
        if self.filter_rpm_data:
            self.data = self.data_processor.filter_rpm_data(self.data)
    
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
        
        self.power_runs = self.run_detector.find_power_runs(
            self.data, min_duration, min_rpm_range, throttle_threshold
        )
        return self.power_runs
    
    def calculate_power_torque(self, run_data, debug_mode: bool = False):
        """Calculate power and torque from vehicle dynamics using RPM rate of change"""
        return self.power_calculator.calculate_power_torque(run_data, debug_mode)
    
    def validate_hp_torque_crossover(self, run_data, power_hp, torque_lbft):
        """Validate that HP and Torque curves cross at 5252 RPM"""
        return self.power_calculator.validate_hp_torque_crossover(run_data, power_hp, torque_lbft)
    
    def plot_power_curves(self, save_path: Optional[str] = None, title: Optional[str] = None) -> None:
        """Create dyno-style power and torque curves on single plot"""
        if not self.power_runs:
            raise ValueError("No power runs found. Call find_power_runs() first.")
        
        self.plotter.plot_power_curves(
            self.data, self.power_runs, self.power_calculator, save_path, title
        )
    
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
            
            # After calculate_power_torque, the run_data may have been cleaned (RPM reversions removed)
            # So we need to get the cleaned data dimensions
            cleaned_run_data = self.data_processor.remove_rpm_reversion_datapoints(run_data)
            rpm = cleaned_run_data['Engine Speed'].values
            time_values = cleaned_run_data['Section Time'].values
            
            # Ensure all arrays have the same length after cleaning
            min_length = min(len(power_hp), len(torque_lbft), len(rpm), len(time_values))
            power_hp = power_hp[:min_length]
            torque_lbft = torque_lbft[:min_length] 
            rpm = rpm[:min_length]
            time_values = time_values[:min_length]
            
            # Filter out unrealistic values
            valid_mask = (power_hp > 0) & (power_hp < AnalysisConstants.MAX_REALISTIC_POWER) & (torque_lbft > 0) & (torque_lbft < AnalysisConstants.MAX_REALISTIC_TORQUE)
            
            if np.any(valid_mask):
                for j in range(len(rpm)):
                    if j < len(valid_mask) and valid_mask[j]:
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
            
            used_data_indices = set()  # Track which data points we've already used
            
            while current_time <= max_time:
                # Find the closest data point to this time that hasn't been used yet
                available_data = [(i, d) for i, d in enumerate(all_data) if i not in used_data_indices]
                if not available_data:
                    break
                    
                closest_idx, closest_data = min(available_data, key=lambda x: abs(x[1]['time'] - current_time))
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
                    
                    used_data_indices.add(closest_idx)  # Mark this data point as used
                
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
                
            run_data = self.data.iloc[start_idx:end_idx+1].copy()
            power_hp, torque_lbft = self.calculate_power_torque(run_data)
            
            # After calculate_power_torque, the run_data may have been cleaned (RPM reversions removed)
            # So we need to get the cleaned data dimensions
            cleaned_run_data = self.data_processor.remove_rpm_reversion_datapoints(run_data)
            
            # Ensure all arrays have the same length after cleaning
            min_length = min(len(power_hp), len(torque_lbft), len(cleaned_run_data))
            power_hp = power_hp[:min_length]
            torque_lbft = torque_lbft[:min_length] 
            
            valid_mask = (power_hp > 0) & (power_hp < AnalysisConstants.MAX_REALISTIC_POWER) & (torque_lbft > 0) & (torque_lbft < AnalysisConstants.MAX_REALISTIC_TORQUE)
            
            if np.any(valid_mask):
                max_power = np.max(power_hp[valid_mask])
                max_torque = np.max(torque_lbft[valid_mask])
                
                # Use cleaned run data for RPM values
                cleaned_rpm = cleaned_run_data['Engine Speed'].values[:min_length]
                rpm_at_max_power = cleaned_rpm[valid_mask][np.argmax(power_hp[valid_mask])]
                rpm_at_max_torque = cleaned_rpm[valid_mask][np.argmax(torque_lbft[valid_mask])]
                
                # Validate HP-Torque crossover
                validation = self.validate_hp_torque_crossover(cleaned_run_data, power_hp[valid_mask], torque_lbft[valid_mask])
                
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