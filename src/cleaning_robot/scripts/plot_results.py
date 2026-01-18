#!/usr/bin/env python3
"""
Plot Results Module
===================
Generates plots from simulation logs:
- Position tracking error over time
- Orientation tracking error over time
- Manipulability over time
- Joint torques over time
- Joint positions over time
- 3D trajectory visualization

Usage:
    python plot_results.py simulation_data.npz
    python plot_results.py --compare simulation_data_with_manip.npz simulation_data_without_manip.npz
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(filename):
    """Load simulation data from .npz file."""
    data = np.load(filename, allow_pickle=True)
    return dict(data)


def plot_single_simulation(data, save_prefix='results'):
    """
    Generate plots for a single simulation run.
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
    """
    time = data['time']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cleaning Robot Simulation Results', fontsize=14, fontweight='bold')
    
    # Plot 1: Position Error
    ax = axes[0, 0]
    ax.plot(time, data['error_pos_norm'] * 1000, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add approach phase indicator
    if 'phase' in data:
        phases = data['phase']
        approach_end = None
        for i, p in enumerate(phases):
            if p == 'circle':
                approach_end = time[i]
                break
        if approach_end:
            ax.axvline(x=approach_end, color='r', linestyle='--', alpha=0.5, label='Approach end')
            ax.legend()
    
    # Plot 2: Orientation Error
    ax = axes[0, 1]
    ax.plot(time, np.rad2deg(data['error_rot_norm']), 'r-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Orientation Error (deg)')
    ax.set_title('Orientation Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 3: Manipulability
    ax = axes[1, 0]
    ax.plot(time, data['manipulability'], 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Manipulability')
    ax.set_title('Manipulability Index w(q)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 4: Joint Torques
    ax = axes[1, 1]
    tau = data['tau']
    joint_labels = ['Roll', 'Pitch', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(tau.shape[1]):
        ax.plot(time, tau[:, i], color=colors[i], linewidth=1, label=joint_labels[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_tracking.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_tracking.png")
    
    # Additional plot: Joint positions
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('Joint Motion', fontsize=14, fontweight='bold')
    
    q = data['q']
    
    # Platform joints
    ax = axes2[0]
    ax.plot(time, np.rad2deg(q[:, 0]), 'b-', linewidth=1.5, label='Platform Roll')
    ax.plot(time, np.rad2deg(q[:, 1]), 'r-', linewidth=1.5, label='Platform Pitch')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Platform Joint Angles')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Arm joints
    ax = axes2[1]
    for i in range(2, 8):
        ax.plot(time, np.rad2deg(q[:, i]), linewidth=1, label=f'Z1 Joint {i-1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Z1 Arm Joint Angles')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_joints.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_joints.png")
    
    # 3D Trajectory plot
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    pos_ref = data['pose_ref_pos']
    pos_cur = data['pose_cur_pos']
    
    ax3.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'g-', linewidth=2, label='Reference')
    ax3.plot(pos_cur[:, 0], pos_cur[:, 1], pos_cur[:, 2], 'b--', linewidth=1.5, label='Actual')
    ax3.scatter(pos_ref[0, 0], pos_ref[0, 1], pos_ref[0, 2], c='green', s=100, marker='o', label='Start')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('End-Effector Trajectory (3D)')
    ax3.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        pos_ref[:, 0].max() - pos_ref[:, 0].min(),
        pos_ref[:, 1].max() - pos_ref[:, 1].min(),
        pos_ref[:, 2].max() - pos_ref[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (pos_ref[:, 0].max() + pos_ref[:, 0].min()) * 0.5
    mid_y = (pos_ref[:, 1].max() + pos_ref[:, 1].min()) * 0.5
    mid_z = (pos_ref[:, 2].max() + pos_ref[:, 2].min()) * 0.5
    
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(f'{save_prefix}_trajectory_3d.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_trajectory_3d.png")
    
    # XY trajectory (top view)
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.plot(pos_ref[:, 0], pos_ref[:, 1], 'g-', linewidth=2, label='Reference')
    ax4.plot(pos_cur[:, 0], pos_cur[:, 1], 'b--', linewidth=1.5, label='Actual')
    ax4.scatter(pos_ref[0, 0], pos_ref[0, 1], c='green', s=100, marker='o', label='Start')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('End-Effector Trajectory (Top View)')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.savefig(f'{save_prefix}_trajectory_xy.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_trajectory_xy.png")
    
    return fig, fig2, fig3, fig4


def plot_comparison(data_with, data_without, save_prefix='comparison'):
    """
    Generate comparison plots: with vs without manipulability optimization.
    
    Args:
        data_with: Data with manipulability optimization
        data_without: Data without manipulability optimization
        save_prefix: Prefix for saved figure files
    """
    time_with = data_with['time']
    time_without = data_without['time']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison: With vs Without Manipulability Optimization', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Position Error Comparison
    ax = axes[0, 0]
    ax.plot(time_with, data_with['error_pos_norm'] * 1000, 'b-', 
            linewidth=1.5, label='With Manip. Opt.')
    ax.plot(time_without, data_without['error_pos_norm'] * 1000, 'r--', 
            linewidth=1.5, label='Without Manip. Opt.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    
    # Plot 2: Orientation Error Comparison
    ax = axes[0, 1]
    ax.plot(time_with, np.rad2deg(data_with['error_rot_norm']), 'b-', 
            linewidth=1.5, label='With Manip. Opt.')
    ax.plot(time_without, np.rad2deg(data_without['error_rot_norm']), 'r--', 
            linewidth=1.5, label='Without Manip. Opt.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Orientation Error (deg)')
    ax.set_title('Orientation Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    
    # Plot 3: Manipulability Comparison
    ax = axes[1, 0]
    ax.plot(time_with, data_with['manipulability'], 'b-', 
            linewidth=1.5, label='With Manip. Opt.')
    ax.plot(time_without, data_without['manipulability'], 'r--', 
            linewidth=1.5, label='Without Manip. Opt.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Manipulability')
    ax.set_title('Manipulability Index w(q)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    
    # Compute statistics
    # Skip approach phase for fair comparison
    approach_end_idx = int(3.0 / (time_with[1] - time_with[0]))  # 3s approach
    
    manip_with_mean = np.mean(data_with['manipulability'][approach_end_idx:])
    manip_without_mean = np.mean(data_without['manipulability'][approach_end_idx:])
    improvement = (manip_with_mean - manip_without_mean) / manip_without_mean * 100
    
    ax.text(0.02, 0.98, f'Mean (circle phase):\n  With: {manip_with_mean:.4f}\n  Without: {manip_without_mean:.4f}\n  Improvement: {improvement:.1f}%',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Torque Norm Comparison
    ax = axes[1, 1]
    tau_norm_with = np.linalg.norm(data_with['tau'], axis=1)
    tau_norm_without = np.linalg.norm(data_without['tau'], axis=1)
    ax.plot(time_with, tau_norm_with, 'b-', linewidth=1.5, label='With Manip. Opt.')
    ax.plot(time_without, tau_norm_without, 'r--', linewidth=1.5, label='Without Manip. Opt.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque Norm (Nm)')
    ax.set_title('Total Torque Magnitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_comparison.png")
    
    # Print statistics
    print("\n" + "=" * 50)
    print("COMPARISON STATISTICS (Circle Phase Only)")
    print("=" * 50)
    
    pos_err_with = np.mean(data_with['error_pos_norm'][approach_end_idx:]) * 1000
    pos_err_without = np.mean(data_without['error_pos_norm'][approach_end_idx:]) * 1000
    
    rot_err_with = np.mean(np.rad2deg(data_with['error_rot_norm'][approach_end_idx:]))
    rot_err_without = np.mean(np.rad2deg(data_without['error_rot_norm'][approach_end_idx:]))
    
    print(f"\nPosition Error (mean):")
    print(f"  With Manip. Opt.:    {pos_err_with:.3f} mm")
    print(f"  Without Manip. Opt.: {pos_err_without:.3f} mm")
    
    print(f"\nOrientation Error (mean):")
    print(f"  With Manip. Opt.:    {rot_err_with:.3f} deg")
    print(f"  Without Manip. Opt.: {rot_err_without:.3f} deg")
    
    print(f"\nManipulability (mean):")
    print(f"  With Manip. Opt.:    {manip_with_mean:.4f}")
    print(f"  Without Manip. Opt.: {manip_without_mean:.4f}")
    print(f"  Improvement:         {improvement:.1f}%")
    
    print("=" * 50)
    
    return fig


def print_summary(data):
    """Print summary statistics from simulation data."""
    time = data['time']
    
    # Find approach end
    approach_end_idx = 0
    if 'phase' in data:
        phases = data['phase']
        for i, p in enumerate(phases):
            if p == 'circle':
                approach_end_idx = i
                break
    else:
        approach_end_idx = int(3.0 / (time[1] - time[0]))  # Assume 3s approach
    
    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    
    print(f"\nDuration: {time[-1]:.2f} s")
    print(f"Samples: {len(time)}")
    print(f"Sample rate: {1.0/(time[1]-time[0]):.0f} Hz")
    
    # Approach phase
    print(f"\n--- Approach Phase (0 to {time[approach_end_idx]:.1f}s) ---")
    pos_err_approach = data['error_pos_norm'][:approach_end_idx] * 1000
    rot_err_approach = np.rad2deg(data['error_rot_norm'][:approach_end_idx])
    print(f"Position Error: max={np.max(pos_err_approach):.2f}mm, final={pos_err_approach[-1]:.2f}mm")
    print(f"Orientation Error: max={np.max(rot_err_approach):.2f}deg, final={rot_err_approach[-1]:.2f}deg")
    
    # Circle phase
    print(f"\n--- Circle Phase ({time[approach_end_idx]:.1f}s to {time[-1]:.1f}s) ---")
    pos_err_circle = data['error_pos_norm'][approach_end_idx:] * 1000
    rot_err_circle = np.rad2deg(data['error_rot_norm'][approach_end_idx:])
    manip_circle = data['manipulability'][approach_end_idx:]
    
    print(f"Position Error: mean={np.mean(pos_err_circle):.2f}mm, max={np.max(pos_err_circle):.2f}mm")
    print(f"Orientation Error: mean={np.mean(rot_err_circle):.2f}deg, max={np.max(rot_err_circle):.2f}deg")
    print(f"Manipulability: mean={np.mean(manip_circle):.4f}, min={np.min(manip_circle):.4f}, max={np.max(manip_circle):.4f}")
    
    # Torques
    tau = data['tau']
    print(f"\nTorque Statistics:")
    print(f"  Max absolute torque: {np.max(np.abs(tau)):.2f} Nm")
    print(f"  Mean torque norm: {np.mean(np.linalg.norm(tau, axis=1)):.2f} Nm")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Plot simulation results')
    parser.add_argument('data_file', nargs='?', default='simulation_data.npz',
                        help='Simulation data file (.npz)')
    parser.add_argument('--compare', nargs=2, metavar=('WITH', 'WITHOUT'),
                        help='Compare two simulation files')
    parser.add_argument('--output', type=str, default='results',
                        help='Output filename prefix')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        print(f"Loading: {args.compare[0]}")
        data_with = load_data(args.compare[0])
        print(f"Loading: {args.compare[1]}")
        data_without = load_data(args.compare[1])
        
        plot_comparison(data_with, data_without, args.output)
        
        # Also plot individual results
        plot_single_simulation(data_with, f'{args.output}_with_manip')
        plot_single_simulation(data_without, f'{args.output}_without_manip')
        
    else:
        # Single file mode
        print(f"Loading: {args.data_file}")
        data = load_data(args.data_file)
        
        print_summary(data)
        plot_single_simulation(data, args.output)
    
    if args.show:
        plt.show()
    
    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
