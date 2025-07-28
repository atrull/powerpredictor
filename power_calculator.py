"""
Power and torque calculation functions
"""

import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from constants import AnalysisConstants
from vehicle_specs import VehicleSpecs
from data_processing import DataProcessor

# Lazy imports for heavy dependencies
def _import_pandas():
    import pandas as pd
    return pd

def _import_scipy():
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    return signal, gaussian_filter1d, savgol_filter


class PowerCalculator:
    """Handles power and torque calculations from vehicle dynamics"""
    
    def __init__(self, 
                 vehicle_specs: VehicleSpecs,
                 drivetrain_efficiency: float = 0.85,
                 rolling_resistance: float = 0.015,
                 drag_coefficient: float = 0.35,
                 frontal_area: float = 2.5,
                 smoothing_factor: float = 2.5,
                 apply_hp_torque_correction: bool = True,
                 downsample_hz: float = None):
        self.vehicle_specs = vehicle_specs
        self.drivetrain_efficiency = drivetrain_efficiency
        self.rolling_resistance = rolling_resistance
        self.drag_coefficient = drag_coefficient
        self.frontal_area = frontal_area
        self.smoothing_factor = smoothing_factor
        self.apply_hp_torque_correction = apply_hp_torque_correction
        self.downsample_hz = downsample_hz
        self.data_processor = DataProcessor()
    
    def calculate_power_torque(self, run_data: 'pd.DataFrame', debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power and torque from vehicle dynamics using RPM rate of change
        
        Args:
            run_data: DataFrame slice containing a power run
            debug_mode: If True, print debug information about calculations
            
        Returns:
            Tuple of (power_hp, torque_lbft) arrays
        """
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        
        # FIRST: Remove RPM reversion data points before any processing
        # This eliminates ECU step artifacts at the source
        run_data = self.data_processor.remove_rpm_reversion_datapoints(run_data)
        
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
        
        # Since we removed reversion data points, ensure remaining data is monotonic
        # This prevents any remaining backwards torque lines and power curve artifacts
        rpm_monotonic, rpm_fixes = self.data_processor.enforce_rpm_monotonicity(rpm)
        
        if rpm_fixes > 0:
            print(f"Post-cleaning monotonicity fix: corrected {rpm_fixes} remaining RPM issues")
        
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
        was_downsampled = self.downsample_hz is not None
        
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
            
            # CRITICAL: Enforce strict monotonicity to eliminate backwards torque lines
            # Even after smoothing, ensure RPM always increases for proper power calculations
            # Use strict enforcement since we're in WOT acceleration
            for i in range(1, len(rpm_smoothed)):
                if rpm_smoothed[i] < rpm_smoothed[i-1]:
                    rpm_smoothed[i] = rpm_smoothed[i-1] + 0.5  # Force small increase
            
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
        # Both power and torque must be corrected by the same efficiency for HP-Torque relationship to hold
        power_hp = power_hp / self.drivetrain_efficiency
        # Note: Torque efficiency correction will be applied after smoothing to maintain calculation consistency
        
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
            
            # Apply drivetrain efficiency to torque to match power correction
            # This ensures both power and torque are at the same reference point (crankshaft)
            corrected_torque_lbft = smoothed_torque_lbft / self.drivetrain_efficiency
            
            # Use the physics-based calculations directly - they should naturally cross at 5252 RPM
            # The HP = (Torque × RPM) / 5252 relationship is fundamental to rotational mechanics
            # and emerges naturally when both values are corrected by the same efficiency factor
            final_power_hp, final_torque_lbft = smoothed_power_hp, corrected_torque_lbft
            
            if self.apply_hp_torque_correction:
                print("Using pure physics-based calculations with consistent efficiency correction for proper 5252 RPM crossover")
        else:
            # Apply efficiency correction to torque for consistency with power calculation
            corrected_torque_lbft = torque_lbft / self.drivetrain_efficiency
            final_power_hp, final_torque_lbft = power_hp, corrected_torque_lbft
        
        return final_power_hp, final_torque_lbft
    
    def validate_hp_torque_crossover(self, run_data: 'pd.DataFrame', power_hp: np.ndarray, torque_lbft: np.ndarray) -> Dict[str, float]:
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
        was_downsampled = self.downsample_hz is not None
        
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
        
        return final_power, final_torque
    
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