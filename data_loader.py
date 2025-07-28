"""
Data loading and quality assessment for ECU log files
"""

import numpy as np
from typing import Dict, Optional, Tuple
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


class DataLoader:
    """Handles CSV data loading and quality assessment"""
    
    def __init__(self, downsample_hz: Optional[float] = None, filter_rpm_data: bool = True):
        self.downsample_hz = downsample_hz
        self.filter_rpm_data = filter_rpm_data
        self.data = None
    
    def load_data(self, csv_path: str) -> 'pd.DataFrame':
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
            
            print(f"Loaded {len(self.data)} data points from {csv_path}")
            
            return self.data
            
        except Exception as e:
            raise ValueError(f"Error loading CSV data: {e}")
    
    def _assess_data_quality(self) -> Dict:
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

    def _detect_optimal_frequency(self) -> Optional[float]:
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
        """Apply generic cleanup to Engine Speed only - other parameters can legitimately be static"""
        critical_params = [
            'Engine Speed'  # Only clean RPM - other parameters like TPS, MAP, etc. can be static
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
    
    def _downsample_data(self, target_hz: float) -> 'pd.DataFrame':
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