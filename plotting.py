"""
Plotting and visualization functions
"""

import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

from constants import AnalysisConstants
from vehicle_specs import VehicleSpecs

# Lazy imports for heavy dependencies
def _import_matplotlib():
    import matplotlib.pyplot as plt
    return plt

def _import_scipy():
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    return signal, gaussian_filter1d, savgol_filter


class Plotter:
    """Handles plotting and visualization of power/torque curves"""
    
    def __init__(self, vehicle_specs: VehicleSpecs, smoothing_factor: float = 2.5):
        self.vehicle_specs = vehicle_specs
        self.smoothing_factor = smoothing_factor
    
    def plot_power_curves(self, data: 'pd.DataFrame', power_runs: List, power_calculator, 
                         save_path: Optional[str] = None, title: Optional[str] = None) -> None:
        """
        Create dyno-style power and torque curves on single plot
        
        Args:
            data: Main DataFrame containing all ECU data
            power_runs: List of power run dictionaries
            power_calculator: PowerCalculator instance for calculations
            save_path: Optional path to save the plot
            title: Optional custom title for the plot
        """
        if not power_runs:
            raise ValueError("No power runs found to plot.")
        
        # Create figure with two subplots - main power plot and dataset coverage plot
        plt = _import_matplotlib()
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        fig, (ax1, ax_coverage) = plt.subplots(2, 1, figsize=(12, 10), 
                                              gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3})
        
        # Create second y-axis for torque on main plot
        ax2 = ax1.twinx()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(power_runs)))
        
        all_power = []
        all_torque = []
        all_rpm = []
        
        # Plot individual runs
        for i, run in enumerate(power_runs):
            start_idx = run['start_idx']
            end_idx = run['end_idx']
                
            run_data = data.iloc[start_idx:end_idx+1].copy()
            power_hp, torque_lbft = power_calculator.calculate_power_torque(run_data)
            
            # After calculate_power_torque, the run_data may have been cleaned (RPM reversions removed)
            # So we need to get the cleaned data dimensions
            cleaned_run_data = power_calculator.data_processor.remove_rpm_reversion_datapoints(run_data)
            rpm = cleaned_run_data['Engine Speed'].values
            
            # Ensure all arrays have the same length after cleaning
            min_length = min(len(power_hp), len(torque_lbft), len(rpm))
            power_hp = power_hp[:min_length]
            torque_lbft = torque_lbft[:min_length] 
            rpm = rpm[:min_length]
            
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
        if len(power_runs) > 1 and all_rpm:
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
        if power_runs:
            # Collect environmental data from all power runs
            all_bap = []
            all_iat = []
            
            for run in power_runs:
                start_idx = run['start_idx']
                end_idx = run['end_idx']
                run_data = data.iloc[start_idx:end_idx+1]
                
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
        if data is not None:
            # Plot the entire dataset timeline as a gray line
            total_time = data['Section Time'].values
            ax_coverage.plot(total_time, [1] * len(total_time), 'lightgray', linewidth=2, alpha=0.5, label='Total Dataset')
            
            # Highlight power runs with colored bars
            for i, run in enumerate(power_runs):
                # Show full detected run in light color
                full_run_data = data.iloc[run['start_idx']:run['end_idx']+1]
                full_run_time = full_run_data['Section Time'].values
                ax_coverage.plot(full_run_time, [1] * len(full_run_time), color=colors[i], linewidth=2, alpha=0.3)
                
                # Show analysis run data in bold color
                analysis_run_data = data.iloc[run['start_idx']:run['end_idx']+1]
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