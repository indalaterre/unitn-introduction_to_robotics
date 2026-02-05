import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_data(filename):
    """
    Load simulation data from a .npz file.
    
    Args:
        filename: Path to .npz file containing simulation data
        
    Returns:
        Dictionary with simulation data arrays
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    try:
        data = np.load(filename, allow_pickle=True)
        return dict(data)
    except Exception as e:
        raise ValueError(f"Failed to load data from {filename}: {e}")


def plot_tracking_errors(data, save_prefix='results'):
    """
    Generate tracking error plots (position and orientation).
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    time = data['time']
    
    # Create the figure with 2 subplots for tracking errors
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Find approach phase end
    approach_end_time = _get_approach_end_time(data)
    
    # Plot 1: Position Error
    ax = axes[0]
    ax.plot(time, data['error_pos_norm'] * 1000, 'b-', linewidth=2, label='Position Error')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position Error (mm)', fontsize=11)
    ax.set_title('Position Tracking Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add approach phase indicator
    if approach_end_time is not None:
        ax.axvline(x=approach_end_time, color='r', linestyle='--', 
                   alpha=0.5, linewidth=1.5, label='Approach → Circle')
        ax.legend(loc='best', fontsize=9)
    
    # Add statistics text
    mean_err = np.mean(data['error_pos_norm']) * 1000
    max_err = np.max(data['error_pos_norm']) * 1000
    ax.text(0.02, 0.98, f'Mean: {mean_err:.2f} mm\nMax: {max_err:.2f} mm',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 2: Orientation Error
    ax = axes[1]
    ax.plot(time, np.rad2deg(data['error_rot_norm']), 'r-', linewidth=2, label='Orientation Error')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Orientation Error (deg)', fontsize=11)
    ax.set_title('Orientation Tracking Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add approach phase indicator
    if approach_end_time is not None:
        ax.axvline(x=approach_end_time, color='r', linestyle='--', 
                   alpha=0.5, linewidth=1.5, label='Approach → Circle')
        ax.legend(loc='best', fontsize=9)
    
    # Add statistics text
    mean_err_rot = np.mean(np.rad2deg(data['error_rot_norm']))
    max_err_rot = np.max(np.rad2deg(data['error_rot_norm']))
    ax.text(0.02, 0.98, f'Mean: {mean_err_rot:.2f}°\nMax: {max_err_rot:.2f}°',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    filename = f'{save_prefix}_tracking_errors.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def plot_performance_metrics(data, save_prefix='results'):
    """
    Generate performance metric plots (manipulability and torques).
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    time = data['time']
    
    # Create the figure with 2 subplots for performance metrics
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Find approach phase end
    approach_end_time = _get_approach_end_time(data)
    
    # Plot 1: Manipulability
    ax = axes[0]
    ax.plot(time, data['manipulability'], 'g-', linewidth=2, label='Manipulability')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Manipulability w(q)', fontsize=11)
    ax.set_title('Manipulability Index', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add approach phase indicator
    if approach_end_time is not None:
        ax.axvline(x=approach_end_time, color='r', linestyle='--', 
                   alpha=0.5, linewidth=1.5, label='Approach → Circle')
    
    # Add statistics
    mean_manip = np.mean(data['manipulability'])
    min_manip = np.min(data['manipulability'])
    ax.text(0.89, 0.89, f'Mean: {mean_manip:.4f}\nMin: {min_manip:.4f}',
            transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.legend(loc='best', fontsize=9)
    
    # Plot 2: Joint Torques
    ax = axes[1]
    tau = data['tau']
    joint_labels = ['Platform Roll', 'Platform Pitch', 'Z1-J1', 'Z1-J2', 
                    'Z1-J3', 'Z1-J4', 'Z1-J5', 'Z1-J6']
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i in range(tau.shape[1]):
        ax.plot(time, tau[:, i], color=colors[i], linewidth=1.2, 
                label=joint_labels[i], alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Torque (Nm)', fontsize=11)
    ax.set_title('Joint Torques', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', ncol=2, fontsize=8, framealpha=0.9)
    
    # Add approach phase indicator
    if approach_end_time is not None:
        ax.axvline(x=approach_end_time, color='r', linestyle='--', 
                   alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    filename = f'{save_prefix}_performance.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def plot_single_simulation(data, save_prefix='results'):
    """
    Generate all plots for a single simulation run.
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
        
    Returns:
        Tuple of figure objects
    """
    # Generate separate plot sets
    fig_errors = plot_tracking_errors(data, save_prefix)
    fig_performance = plot_performance_metrics(data, save_prefix)
    
    # Joint positions plot
    fig_joints = _plot_joint_positions(data, save_prefix)
    
    # Trajectory plots
    fig_3d = _plot_trajectory_3d(data, save_prefix)
    fig_xy = _plot_trajectory_xy(data, save_prefix)
    
    return fig_errors, fig_performance, fig_joints, fig_3d, fig_xy


def _get_approach_end_time(data):
    """
    Find the time when approach phase ends.
    
    Args:
        data: Dictionary with simulation data
        
    Returns:
        Time in seconds when circle phase starts, or None
    """
    if 'phase' not in data:
        return None
    
    time = data['time']
    phases = data['phase']
    
    for i, phase in enumerate(phases):
        if phase == 'circle':
            return time[i]
    
    return None


def _plot_joint_positions(data, save_prefix):
    """
    Plot joint positions over time.
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    time = data['time']
    q = data['q']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Joint Motion Analysis', fontsize=14, fontweight='bold')
    
    # Platform joints
    ax = axes[0]
    ax.plot(time, np.rad2deg(q[:, 0]), 'b-', linewidth=2, label='Platform Roll')
    ax.plot(time, np.rad2deg(q[:, 1]), 'r-', linewidth=2, label='Platform Pitch')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.set_title('Platform Joint Angles', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Arm joints
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for i in range(2, 8):
        ax.plot(time, np.rad2deg(q[:, i]), linewidth=1.5, 
                color=colors[i-2], label=f'Z1 Joint {i-1}', alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.set_title('Z1 Arm Joint Angles', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', ncol=2, fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    filename = f'{save_prefix}_joints.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def _plot_trajectory_3d(data, save_prefix):
    """
    Plot 3D trajectory visualization.
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    pos_ref = data['pose_ref_pos']
    pos_cur = data['pose_cur_pos']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 
            'g-', linewidth=2.5, label='Reference', alpha=0.8)
    ax.plot(pos_cur[:, 0], pos_cur[:, 1], pos_cur[:, 2], 
            'b--', linewidth=2, label='Actual', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(pos_ref[0, 0], pos_ref[0, 1], pos_ref[0, 2], 
               c='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=1.5)
    ax.scatter(pos_ref[-1, 0], pos_ref[-1, 1], pos_ref[-1, 2], 
               c='red', s=150, marker='s', label='End', edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('X (m)', fontsize=11, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=11, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=11, labelpad=10)
    ax.set_title('End-Effector Trajectory (3D View)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([
        pos_ref[:, 0].max() - pos_ref[:, 0].min(),
        pos_ref[:, 1].max() - pos_ref[:, 1].min(),
        pos_ref[:, 2].max() - pos_ref[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (pos_ref[:, 0].max() + pos_ref[:, 0].min()) * 0.5
    mid_y = (pos_ref[:, 1].max() + pos_ref[:, 1].min()) * 0.5
    mid_z = (pos_ref[:, 2].max() + pos_ref[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    filename = f'{save_prefix}_trajectory_3d.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def _plot_trajectory_xy(data, save_prefix):
    """
    Plot XY trajectory (top view).
    
    Args:
        data: Dictionary with simulation data
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    pos_ref = data['pose_ref_pos']
    pos_cur = data['pose_cur_pos']
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Plot trajectories
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], 'g-', linewidth=2.5, 
            label='Reference', alpha=0.8)
    ax.plot(pos_cur[:, 0], pos_cur[:, 1], 'b--', linewidth=2, 
            label='Actual', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(pos_ref[0, 0], pos_ref[0, 1], c='green', s=150, 
               marker='o', label='Start', edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(pos_ref[-1, 0], pos_ref[-1, 1], c='red', s=150, 
               marker='s', label='End', edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('End-Effector Trajectory (Top View)', fontsize=13, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add circle center indicator if circular trajectory
    if len(pos_ref) > 10:
        center_x = np.mean(pos_ref[:, 0])
        center_y = np.mean(pos_ref[:, 1])
        ax.plot(center_x, center_y, 'k+', markersize=15, markeredgewidth=2, label='Center')
    
    filename = f'{save_prefix}_trajectory_xy.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def plot_comparison(data_with, data_without, save_prefix='comparison'):
    """
    Generate comparison plots: with vs without manipulability optimization.
    
    Args:
        data_with: Data with manipulability optimization
        data_without: Data without manipulability optimization
        save_prefix: Prefix for saved figure files
        
    Returns:
        Tuple of figure objects
    """
    # Separate tracking errors comparison
    fig_errors = _plot_comparison_tracking_errors(data_with, data_without, save_prefix)
    
    # Performance metrics comparison
    fig_performance = _plot_comparison_performance(data_with, data_without, save_prefix)
    
    # Print statistics
    _print_comparison_statistics(data_with, data_without)
    
    return fig_errors, fig_performance


def _plot_comparison_tracking_errors(data_with, data_without, save_prefix):
    """
    Plot tracking error comparison.
    
    Args:
        data_with: Data with manipulability optimization
        data_without: Data without manipulability optimization
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    time_with = data_with['time']
    time_without = data_without['time']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Tracking Error Comparison: With vs Without Manipulability Optimization', 
                 fontsize=13, fontweight='bold')
    
    # Plot 1: Position Error Comparison
    ax = axes[0]
    ax.plot(time_with, data_with['error_pos_norm'] * 1000, 'b-', 
            linewidth=2, label='With Manip. Opt.', alpha=0.9)
    ax.plot(time_without, data_without['error_pos_norm'] * 1000, 'r--', 
            linewidth=2, label='Without Manip. Opt.', alpha=0.9)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position Error (mm)', fontsize=11)
    ax.set_title('Position Tracking Error Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    # Plot 2: Orientation Error Comparison
    ax = axes[1]
    ax.plot(time_with, np.rad2deg(data_with['error_rot_norm']), 'b-', 
            linewidth=2, label='With Manip. Opt.', alpha=0.9)
    ax.plot(time_without, np.rad2deg(data_without['error_rot_norm']), 'r--', 
            linewidth=2, label='Without Manip. Opt.', alpha=0.9)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Orientation Error (deg)', fontsize=11)
    ax.set_title('Orientation Tracking Error Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    filename = f'{save_prefix}_tracking_errors_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def _plot_comparison_performance(data_with, data_without, save_prefix):
    """
    Plot performance metrics comparison.
    
    Args:
        data_with: Data with manipulability optimization
        data_without: Data without manipulability optimization
        save_prefix: Prefix for saved figure files
        
    Returns:
        Figure object
    """
    time_with = data_with['time']
    time_without = data_without['time']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Performance Metrics Comparison: With vs Without Manipulability Optimization', 
                 fontsize=13, fontweight='bold')
    
    # Plot 1: Manipulability Comparison
    ax = axes[0]
    ax.plot(time_with, data_with['manipulability'], 'b-', 
            linewidth=2, label='With Manip. Opt.', alpha=0.9)
    ax.plot(time_without, data_without['manipulability'], 'r--', 
            linewidth=2, label='Without Manip. Opt.', alpha=0.9)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Manipulability w(q)', fontsize=11)
    ax.set_title('Manipulability Index Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    # Compute statistics (skip approach phase)
    approach_end_idx = int(3.0 / (time_with[1] - time_with[0]))  # 3s approach
    manip_with_mean = np.mean(data_with['manipulability'][approach_end_idx:])
    manip_without_mean = np.mean(data_without['manipulability'][approach_end_idx:])
    improvement = (manip_with_mean - manip_without_mean) / manip_without_mean * 100
    
    ax.text(0.02, 0.98, 
            f'Circle Phase Statistics:\n'
            f'  With Opt: {manip_with_mean:.4f}\n'
            f'  Without Opt: {manip_without_mean:.4f}\n'
            f'  Improvement: {improvement:.1f}%',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 2: Torque Norm Comparison
    ax = axes[1]
    tau_norm_with = np.linalg.norm(data_with['tau'], axis=1)
    tau_norm_without = np.linalg.norm(data_without['tau'], axis=1)
    ax.plot(time_with, tau_norm_with, 'b-', linewidth=2, 
            label='With Manip. Opt.', alpha=0.9)
    ax.plot(time_without, tau_norm_without, 'r--', linewidth=2, 
            label='Without Manip. Opt.', alpha=0.9)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Torque Norm (Nm)', fontsize=11)
    ax.set_title('Total Torque Magnitude Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    filename = f'{save_prefix}_performance_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig


def _print_comparison_statistics(data_with, data_without):
    """
    Print comparison statistics.
    
    Args:
        data_with: Data with manipulability optimization
        data_without: Data without manipulability optimization
    """
    time_with = data_with['time']
    approach_end_idx = int(3.0 / (time_with[1] - time_with[0]))  # 3s approach
    
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS (Circle Phase Only)")
    print("=" * 60)
    
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
    
    print("=" * 60)


def print_summary(data):
    """
    Print comprehensive summary statistics from simulation data.
    
    Args:
        data: Dictionary with simulation data
    """
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
    """
    Main entry point for plot generation.
    """
    parser = argparse.ArgumentParser(
        description='Generate plots from cleaning robot simulation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --data_file simulation.npz
  %(prog)s --compare data_with.npz data_without.npz --output comparison
  %(prog)s --data_file results.npz --show
        ''')
    
    parser.add_argument('--data_file', nargs='?', default='simulation_data.npz',
                        help='Simulation data file (.npz) [default: simulation_data.npz]')
    parser.add_argument('--compare', nargs=2, metavar=('WITH_MANIP', 'WITHOUT_MANIP'),
                        help='Compare two simulation files (with vs without optimization)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output filename prefix [default: results]')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (in addition to saving)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Cleaning Robot Simulation - Plot Generator")
    print("=" * 60)
    
    
    try:
        if args.compare:
            # Comparison mode
            print(f"\n[Comparison Mode]")
            print(f"Loading WITH optimization: {args.compare[0]}")
            data_with = load_data(args.compare[0])
            print(f"Loading WITHOUT optimization: {args.compare[1]}")
            data_without = load_data(args.compare[1])
            
            print("\nGenerating comparison plots...")
            plot_comparison(data_with, data_without, args.output)
            
            print("\nGenerating individual plots...")
            print("  - With manipulability optimization:")
            plot_single_simulation(data_with, f'{args.output}_with_manip')
            print("  - Without manipulability optimization:")
            plot_single_simulation(data_without, f'{args.output}_without_manip')
            
        else:
            # Single file mode
            print(f"\n[Single File Mode]")
            print(f"Loading: {args.data_file}")
            data = load_data(args.data_file)
            
            print_summary(data)
            
            print("\nGenerating plots...")
            plot_single_simulation(data, args.output)
        
        if args.show:
            print("\nDisplaying plots interactively...")
            plt.show()
        
        print("\n" + "=" * 60)
        print("Plotting complete! All figures saved successfully.")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check that the data file exists.\n")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your data file format.\n")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
