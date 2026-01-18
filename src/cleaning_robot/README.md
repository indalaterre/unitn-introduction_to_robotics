# Cleaning Robot on Tilting Platform

## Overview

This package implements an 8-DoF redundant manipulator for table cleaning tasks:
- **2-DoF Pan-Tilt Platform**: Roll (X-axis) and Pitch (Y-axis) joints
- **6-DoF Unitree Z1 Arm**: Mounted on top of the platform

The robot tracks a horizontal circular trajectory (radius 0.20 m) while keeping the end-effector perpendicular to the table surface. The 2 extra degrees of freedom are exploited to maximize manipulability in the null space.

## System Architecture

```
world
  └── table (fixed, cleaning surface)
  └── platform_base (fixed)
        └── platform_roll (revolute, X-axis)
              └── platform_pitch (revolute, Y-axis)
                    └── z1_base (fixed)
                          └── z1_joint1 (shoulder yaw, Z-axis)
                                └── z1_joint2 (shoulder pitch, Y-axis)
                                      └── z1_joint3 (elbow pitch, Y-axis)
                                            └── z1_joint4 (wrist yaw, Z-axis)
                                                  └── z1_joint5 (wrist pitch, Y-axis)
                                                        └── z1_joint6 (wrist roll, X-axis)
                                                              └── ee_link → tool_tip
```

## Dependencies

### Python Packages
```bash
pip install numpy scipy matplotlib pinocchio
```

### ROS (Optional, for visualization)
- ROS Noetic
- `robot_state_publisher`
- `rviz`

### System
- Python 3.8+
- Pinocchio (robotics library)

## File Structure

```
cleaning_robot/
├── urdf/
│   └── assembly.urdf          # Robot URDF (platform + Z1 arm)
├── scripts/
│   ├── robot_model.py         # Pinocchio model wrapper (FK, Jacobian, dynamics)
│   ├── trajectory.py          # Circular trajectory generator
│   ├── controller.py          # Task-space inverse dynamics controller
│   ├── run_sim.py             # Main simulation runner
│   └── plot_results.py        # Result visualization
├── launch/
│   └── display.launch         # ROS launch file for RViz
├── config/
│   └── cleaning_robot.rviz    # RViz configuration
├── CMakeLists.txt
├── package.xml
└── README.md
```

## Quick Start

### Option A: Run Everything with One Command

```bash
cd src/cleaning_robot
chmod +x run_all.sh
./run_all.sh
```

This runs the comparison experiment and generates all plots automatically.

### Option B: Step-by-Step

#### 1. Standalone Simulation (No ROS)

```bash
cd src/cleaning_robot/scripts

# Run simulation with manipulability optimization
python3 run_sim.py --mode standalone --duration 20

# Generate plots
python3 plot_results.py simulation_data.npz --show
```

#### 2. Comparison Experiment

Compare performance with and without null-space manipulability optimization:

```bash
cd src/cleaning_robot/scripts

# Run comparison
python3 run_sim.py --mode compare --duration 20

# Generate comparison plots
python3 plot_results.py --compare simulation_data_with_manip.npz simulation_data_without_manip.npz --show
```

#### 3. ROS/Locosim Visualization

```bash
# Terminal 1: Launch RViz
roslaunch cleaning_robot display.launch

# Terminal 2: Run controller
cd src/cleaning_robot/scripts
python3 run_sim.py --mode ros --duration 20
```

## Controller Details

### Primary Task: Pose Tracking

Task-space inverse dynamics controller:

```
τ = M(q) · q̈ + h(q, q̇)
```

where:
```
q̈_task = J# · (ẍ* - J̇·q̇)
ẍ* = ẍ_ref + Kd·(ẋ_ref - ẋ) + Kp·e
e = [e_pos; e_rot]
```

- **Position error**: `e_pos = p_ref - p`
- **Orientation error**: `e_rot = log(R_ref · R^T)` (SO(3) log map)
- **Damped pseudoinverse**: `J# = J^T (J·J^T + λ²I)^{-1}`

### Secondary Task: Manipulability Maximization

Null-space optimization to maximize dexterity:

```
q̈_null = k_null · N · ∇w(q)
N = I - J# · J
w(q) = √det(Jp · Jp^T)
```

where `Jp` is the 3×8 position Jacobian.

### Total Control

```
q̈ = q̈_task + q̈_null
τ = M(q) · q̈ + h(q, q̇)
```

## Trajectory Specification

### Circular Cleaning Trajectory

- **Center**: `p0 = [0.35, 0.0, 0.05]` m (configurable)
- **Radius**: `r = 0.20` m
- **Height**: `z = 0.05` m above table
- **Angular velocity**: `ω = 0.4` rad/s

Position reference:
```
p_ref(t) = p0 + [r·cos(ωt), r·sin(ωt), 0]
```

Orientation reference: Tool z-axis pointing DOWN (perpendicular to table)
```
R_ref = Ry(π) = diag(1, -1, -1)
```

### Approach Phase

Smooth quintic polynomial trajectory from initial pose to circle start point (3 seconds).

## Command-Line Options

```bash
python run_sim.py --help

Options:
  --mode {ros,standalone,compare}  Simulation mode
  --duration FLOAT                 Simulation duration (seconds)
  --dt FLOAT                       Time step (seconds)
  --radius FLOAT                   Circle radius (meters)
  --omega FLOAT                    Angular velocity (rad/s)
  --k_null FLOAT                   Null-space gain
  --output STRING                  Output filename prefix
```

## Output Plots

1. **Tracking Errors**: Position and orientation error over time
2. **Manipulability**: w(q) over time (shows improvement with null-space optimization)
3. **Joint Torques**: All 8 joint torques over time
4. **Joint Positions**: Platform and arm joint angles
5. **3D Trajectory**: Reference vs actual end-effector path
6. **XY Trajectory**: Top view of cleaning circle

## Expected Results

With proper tuning:
- **Position error**: < 5 mm (steady-state)
- **Orientation error**: < 2° (steady-state)
- **Manipulability improvement**: 10-30% compared to baseline

## Troubleshooting

### Pinocchio Import Error
```bash
# Install Pinocchio
pip install pin
# or
conda install -c conda-forge pinocchio
```

### ROS Not Found
Run in standalone mode:
```bash
python run_sim.py --mode standalone
```

### Simulation Diverges
- Reduce gains: `Kp_pos`, `Kd_pos`
- Increase damping in pseudoinverse
- Check joint limits

### Poor Tracking
- Increase gains
- Reduce trajectory speed (`--omega`)
- Check initial configuration

## Tuning Guidelines

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `Kp_pos` | Position stiffness | 100-500 |
| `Kd_pos` | Position damping | 20-50 |
| `Kp_rot` | Orientation stiffness | 100-300 |
| `Kd_rot` | Orientation damping | 20-40 |
| `k_null` | Null-space gain | 5-20 |
| `damping` | Pseudoinverse damping | 1e-4 to 1e-2 |

## References

- Siciliano, B., et al. "Robotics: Modelling, Planning and Control"
- Pinocchio documentation: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/
- Unitree Z1 specifications: https://www.unitree.com/z1

## License

MIT License
