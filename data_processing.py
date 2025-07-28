"""
RPM filtering and data cleanup functions
"""

import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Lazy imports for heavy dependencies
def _import_pandas():
    import pandas as pd
    return pd

def _import_scipy():
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    return signal, gaussian_filter1d, savgol_filter


class DataProcessor:
    """Handles RPM filtering and data processing"""
    
    def __init__(self, max_gap: int = 5):
        self.max_gap = max_gap
    
    def filter_rpm_data(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Advanced RPM data filtering for ECUs with poor synchronization
        Uses multi-stage approach: outlier detection, trend validation, and adaptive smoothing
        """
        if 'Engine Speed' not in data.columns or 'Section Time' not in data.columns:
            return data
        
        pd = _import_pandas()
        signal, gaussian_filter1d, savgol_filter = _import_scipy()
        
        rpm_col = 'Engine Speed'
        time_col = 'Section Time'
        tps_col = 'TPS (Main)' if 'TPS (Main)' in data.columns else 'TPS 2(Main)'
        
        # Make a copy to work with
        filtered_data = data.copy()
        
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
        
        return filtered_data
    
    def remove_rpm_reversion_datapoints(self, run_data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Remove data points with static or decreasing RPM during WOT acceleration.
        
        During WOT acceleration, RPM should only increase. Any data points that show
        static or decreasing RPM are ECU reporting artifacts and should be removed entirely.
        
        Args:
            run_data: DataFrame containing the power run data
            
        Returns:
            Cleaned DataFrame with reversion data points removed
        """
        pd = _import_pandas()
        
        # Make a copy to work with
        cleaned_data = run_data.copy()
        
        rpm_col = 'Engine Speed'
        time_col = 'Section Time'
        
        if rpm_col not in cleaned_data.columns or time_col not in cleaned_data.columns:
            return cleaned_data
        
        # Calculate overall RPM trend to confirm this is acceleration
        time_span = cleaned_data[time_col].iloc[-1] - cleaned_data[time_col].iloc[0]
        rpm_span = cleaned_data[rpm_col].iloc[-1] - cleaned_data[rpm_col].iloc[0]
        overall_rpm_rate = rpm_span / time_span if time_span > 0 else 0
        
        if overall_rpm_rate > 50:  # This is clearly an acceleration phase
            # Identify indices to remove (RPM reversions)
            indices_to_remove = []
            
            for i in range(1, len(cleaned_data)):
                current_rpm = cleaned_data[rpm_col].iloc[i]
                prev_rpm = cleaned_data[rpm_col].iloc[i-1]
                
                # Remove data points where RPM is static OR decreasing during WOT acceleration
                # Both indicate ECU artifacts since RPM should only increase during acceleration
                if current_rpm <= prev_rpm + 1:  # Static or decreasing (allow 1 RPM tolerance for noise)
                    indices_to_remove.append(i)
            
            if indices_to_remove:
                print(f"Removing {len(indices_to_remove)} RPM reversion data points during WOT acceleration")
                # Remove the problematic indices
                cleaned_data = cleaned_data.drop(cleaned_data.index[indices_to_remove]).reset_index(drop=True)
        
        return cleaned_data

    def enforce_rpm_monotonicity(self, rpm_values: np.ndarray, min_increment: float = 0.1) -> Tuple[np.ndarray, int]:
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