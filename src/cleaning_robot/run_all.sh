#!/bin/bash
# Cleaning Robot - Full Simulation and Plotting Script
# =====================================================
# This script runs the complete simulation workflow:
# 1. Runs comparison experiment (with/without manipulability optimization)
# 2. Generates all plots
# 3. Displays summary statistics

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/scripts"

echo "=============================================="
echo "Cleaning Robot Simulation"
echo "=============================================="
echo ""

# Default parameters
DURATION=${1:-20}
RADIUS=${2:-0.20}
OMEGA=${3:-0.4}

echo "Parameters:"
echo "  Duration: ${DURATION}s"
echo "  Circle radius: ${RADIUS}m"
echo "  Angular velocity: ${OMEGA} rad/s"
echo ""

# Run comparison experiment
echo "=============================================="
echo "Step 1: Running comparison experiment..."
echo "=============================================="
python3 run_sim.py --mode compare --duration $DURATION --radius $RADIUS --omega $OMEGA --output simulation_data

echo ""
echo "=============================================="
echo "Step 2: Generating plots..."
echo "=============================================="
python3 plot_results.py --compare simulation_data_with_manip.npz simulation_data_without_manip.npz --output results

echo ""
echo "=============================================="
echo "Simulation Complete!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - simulation_data_with_manip.npz"
echo "  - simulation_data_without_manip.npz"
echo "  - results_comparison.png"
echo "  - results_with_manip_tracking.png"
echo "  - results_with_manip_joints.png"
echo "  - results_with_manip_trajectory_3d.png"
echo "  - results_with_manip_trajectory_xy.png"
echo "  - results_without_manip_tracking.png"
echo "  - results_without_manip_joints.png"
echo "  - results_without_manip_trajectory_3d.png"
echo "  - results_without_manip_trajectory_xy.png"
echo ""
echo "To view plots interactively, run:"
echo "  python3 plot_results.py --compare simulation_data_with_manip.npz simulation_data_without_manip.npz --show"
