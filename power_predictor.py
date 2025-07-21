#!/usr/bin/env python3
"""
Power and Torque Analysis Tool for ECU Log Data
Analyzes CSV logs to calculate power and torque curves like a dynamometer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import warnings
from scipy import signal
from scipy.ndimage import gaussian_filter1d
warnings.filterwarnings('ignore')

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
                 drag_coefficient: float = 0.3,
                 frontal_area: float = 2.5,
                 smoothing_factor: float = 1.0,
                 apply_hp_torque_correction: bool = True):
        self.vehicle_specs = vehicle_specs
        self.drivetrain_efficiency = drivetrain_efficiency
        self.rolling_resistance = rolling_resistance
        self.drag_coefficient = drag_coefficient
        self.frontal_area = frontal_area
        self.smoothing_factor = smoothing_factor
        self.apply_hp_torque_correction = apply_hp_torque_correction
        self.data = None
        self.power_runs = []
        
    def load_data(self, csv_path: str) -> None:
        """Load CSV data and clean column names"""
        try:
            # Read CSV - header is in row 2 (0-indexed row 1), data starts from row 4
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
                    
            print(f"Loaded {len(self.data)} data points from {csv_path}")
            
        except Exception as e:
            raise ValueError(f"Error loading CSV data: {e}")
    
    def find_power_runs(self, min_duration: float = 1.0, min_rpm_range: float = 500, 
                       throttle_threshold: float = 99.5) -> List[Dict]:
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
        valid_conditions = wot_mask & stable_increase & (self.data['Engine Speed'] > 3000)
        
        # Find continuous runs
        runs = []
        in_run = False
        start_idx = None
        
        for i, valid in enumerate(valid_conditions):
            if valid and not in_run:
                start_idx = i
                in_run = True
            elif not valid and in_run:
                end_idx = i - 1
                
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
        
        # Handle case where run continues to end of data
        if in_run and start_idx is not None:
            end_idx = len(self.data) - 1
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
        return runs
    
    def calculate_power_torque(self, run_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def validate_hp_torque_crossover(self, run_data: pd.DataFrame, power_hp: np.ndarray, torque_lbft: np.ndarray) -> Dict[str, float]:
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
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create second y-axis for torque
        ax2 = ax1.twinx()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.power_runs)))
        
        all_power = []
        all_torque = []
        all_rpm = []
        
        # Plot individual runs
        for i, run in enumerate(self.power_runs):
            run_data = self.data.iloc[run['start_idx']:run['end_idx']+1].copy()
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
            
        plt.figtext(0.02, 0.02, full_info, fontsize=10, style='italic')
        
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
            run_data = self.data.iloc[run['start_idx']:run['end_idx']+1]
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
    parser.add_argument('--drag-coefficient', type=float, default=0.3, 
                       help='Aerodynamic drag coefficient (default: 0.3)')
    parser.add_argument('--frontal-area', type=float, default=2.5, 
                       help='Vehicle frontal area in m² (default: 2.5)')
    parser.add_argument('--smoothing-factor', type=float, default=1.0, 
                       help='Data smoothing factor - 0 disables, higher values = more smoothing (default: 1.0)')
    parser.add_argument('--no-hp-torque-correction', action='store_true', 
                       help='Disable HP-Torque relationship correction (HP = Torque * RPM / 5252)')
    
    # Analysis parameters
    parser.add_argument('--min-duration', type=float, default=1.0, 
                       help='Minimum power run duration in seconds (default: 1.0)')
    parser.add_argument('--min-rpm-range', type=float, default=500, 
                       help='Minimum RPM range for valid run (default: 500)')
    parser.add_argument('--throttle-threshold', type=float, default=99.5, 
                       help='Minimum throttle %% for WOT detection (default: 99.5)')
    
    # Output options
    parser.add_argument('--output', help='Output file for plot (optional)')
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
            apply_hp_torque_correction=not args.no_hp_torque_correction
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
                analyzer.plot_power_curves(args.output, args.title)
        else:
            print("No valid power runs found in the data.")
            print(f"Try adjusting --throttle-threshold (currently {args.throttle_threshold}%)")
            print(f"or --min-duration (currently {args.min_duration}s)")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())