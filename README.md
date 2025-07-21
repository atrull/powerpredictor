# Power Predictor

A tool for analyzing ECU log data to calculate power and torque curves, similar to a dynamometer.

## Overview

This tool analyzes CSV logs from ECUs to determine power and torque by analyzing vehicle dynamics during wide-open-throttle (WOT) conditions. It uses acceleration data, vehicle weight, gearing, and tire specifications to calculate engine power and torque.

## Features

- Filters data for WOT conditions (99-100% throttle) with stable RPM increases
- Calculates power (HP) and torque (lb-ft) from vehicle dynamics
- Generates dyno-style graphs showing power and torque vs RPM
- Supports multiple power runs in a single log file
- Configurable vehicle parameters (weight, gearing, tires, etc.)

## Installation

1. Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib
```

## Usage

Basic usage with default Honda EP3 Type R specifications:
```bash
python power_analyzer.py "your_log.csv"
```

With custom vehicle specifications:
```bash
python power_analyzer.py "your_log.csv" \
  --weight 2500 \
  --occupant 180 \
  --final-drive 4.3 \
  --gear-ratio 1.000 \
  --displacement 2.0 \
  --cylinders 4 \
  --output power_curves.png
```

### Command Line Options

#### Vehicle Specifications
- `--weight`: Vehicle curb weight in lbs (default: 2200)
- `--occupant`: Occupant + gear weight in lbs (default: 200)
- `--displacement`: Engine displacement in liters (default: 2.0 )
- `--cylinders`: Number of cylinders (default: 4)

#### Drivetrain
- `--final-drive`: Final drive ratio (default: 4.7)
- `--gear-ratio`: Current gear ratio (default: 1.212 for EP3 Type R 4th gear)
- `--drivetrain-efficiency`: Drivetrain efficiency factor 0-1 (default: 0.85)

#### Tire Specifications
- `--tire-width`: Tire width in mm (default: 195)
- `--tire-sidewall`: Tire sidewall ratio in % (default: 50)
- `--tire-diameter`: Wheel diameter in inches (default: 15)

#### Vehicle Dynamics
- `--rolling-resistance`: Rolling resistance coefficient (default: 0.015)
- `--drag-coefficient`: Aerodynamic drag coefficient (default: 0.3)
- `--frontal-area`: Vehicle frontal area in m² (default: 2.5)

#### Analysis Parameters
- `--min-duration`: Minimum power run duration in seconds (default: 1.0)
- `--min-rpm-range`: Minimum RPM range for valid run (default: 500)
- `--throttle-threshold`: Minimum throttle % for WOT detection (default: 99.5)

#### Output Options
- `--output`: Output file for the graph (optional)
- `--title`: Custom title for the plot
- `--no-plot`: Skip generating plot (text report only)

## CSV Format Requirements

The tool expects a CSV file with ECU log data containing these columns:
- `Section Time` - Timestamp
- `Engine Speed` - RPM
- `TPS (Main)` - Throttle position (%)
- `Driven Wheel Speed` - Vehicle speed (optional, will calculate from RPM if missing)
- `BAP` - Barometric pressure
- `IAT` - Intake air temperature
- `Lambda 1` or `Lambda Avg` - Air/fuel ratio data

## How It Works

1. **Data Filtering**: Identifies periods where throttle position ≥ 99.5% and RPM is increasing steadily
2. **Power Calculation**: Uses vehicle dynamics (F = ma) to calculate the force required to accelerate the vehicle
3. **Torque Calculation**: Converts wheel force back to crankshaft torque using gear ratios
4. **Corrections**: Applies estimates for rolling resistance, aerodynamic drag, and drivetrain efficiency

## Example Output

The tool generates:
- A text report with vehicle specifications and run summaries
- Power and torque curves plotted vs RPM
- Individual runs plus averaged curves if multiple runs are found

### Sample Analysis Results

**Honda EP3 Type R (Stock Setup):**
```
Run 1: 4369-6242 RPM, Max 219.8 HP @ 6191 RPM, 162.3 lb-ft @ 4412 RPM
Run 2: 6240-6775 RPM, Max 236.8 HP @ 6714 RPM, 166.7 lb-ft @ 6276 RPM
```

*Note: Power differences reflect changes in wheel diameter, weight, and rolling resistance.*

## Limitations

- Power calculations are estimates based on vehicle dynamics
- Requires accurate vehicle specifications for best results
- Rolling resistance and aerodynamic drag are simplified estimates
- Best accuracy when using actual wheel speed data vs calculated from RPM

## Default Vehicle Specifications

The tool defaults to Honda EP3 Type R specifications:
- Weight: 2200 lbs + 200 lbs occupant
- Engine: 2.0L 4-cylinder
- 4th gear ratio: 1.212:1
- Final drive: 4.7:1
- Tires: 195/50R15