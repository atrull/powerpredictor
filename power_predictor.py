#!/usr/bin/env python3
"""
Power and Torque Analysis Tool for ECU Log Data
Analyzes CSV logs to calculate power and torque curves like a dynamometer

Refactored version 2.0 - now uses modular architecture with separate modules
for constants, vehicle specs, data loading, processing, calculations, and plotting.
"""

import argparse
import sys

# Import the modular components
from constants import AnalysisConstants
from vehicle_specs import VehicleSpecs
from analyzer import PowerAnalyzer


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