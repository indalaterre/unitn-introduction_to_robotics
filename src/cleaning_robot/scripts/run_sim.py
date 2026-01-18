#!/usr/bin/env python3
"""
Locosim/ROS Simulation Runner
=============================
Interfaces with Locosim/ROS to run the cleaning robot simulation.
- Loads robot model and publishes joint states
- Runs task-space inverse dynamics control loop
- Logs data for analysis

Usage:
    # With ROS/Locosim:
    roslaunch cleaning_robot display.launch
    python run_sim.py --mode ros
    
    # Standalone (no ROS):
    python run_sim.py --mode standalone
"""

import sys
import os
import argparse
import numpy as np

# Add ROS path if available
try:
    sys.path.append('/opt/ros/noetic/lib/python3.8/site-packages')
    import rospy
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[Warning] ROS not available, running in standalone mode only")

from robot_model import RobotModel
from trajectory import CleaningTrajectory, CircularTrajectory
from controller import TaskSpaceController, SimulationLogger


class ROSSimulation:
    """
    ROS-based simulation interface for Locosim.
    """
    
    def __init__(self, robot, controller, trajectory, dt=0.001):
        """
        Initialize ROS simulation.
        
        Args:
            robot: RobotModel instance
            controller: TaskSpaceController instance
            trajectory: CleaningTrajectory instance
            dt: Control loop time step
        """
        self.robot = robot
        self.controller = controller
        self.trajectory = trajectory
        self.dt = dt
        
        # Initialize ROS node
        rospy.init_node('cleaning_robot_controller', anonymous=True)
        
        # Publishers
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.traj_marker_pub = rospy.Publisher('trajectory_marker', Marker, queue_size=10)
        
        # Rate
        self.rate = rospy.Rate(int(1.0 / dt))
        
        # Joint names (must match URDF)
        self.joint_names = [
            'platform_roll', 'platform_pitch',
            'z1_joint1', 'z1_joint2', 'z1_joint3',
            'z1_joint4', 'z1_joint5', 'z1_joint6'
        ]
        
        # Logger
        self.logger = SimulationLogger()
        
        print("[ROSSimulation] Initialized")
    
    def publish_joint_states(self, q, dq=None, tau=None):
        """Publish joint states to ROS."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = q.tolist()
        
        if dq is not None:
            msg.velocity = dq.tolist()
        if tau is not None:
            msg.effort = tau.tolist()
        
        self.joint_pub.publish(msg)
    
    def publish_trajectory_marker(self, trajectory, num_points=100):
        """Publish trajectory visualization marker."""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.005  # Line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Generate trajectory points
        if hasattr(self.trajectory, 'circle_traj'):
            traj = self.trajectory.circle_traj
            T = traj.period
            for i in range(num_points + 1):
                t = i * T / num_points
                pose = traj.get_pose_reference(t)
                p = Point()
                p.x, p.y, p.z = pose.translation
                marker.points.append(p)
        
        self.traj_marker_pub.publish(marker)
    
    def publish_table_marker(self):
        """Publish table visualization."""
        marker_array = MarkerArray()
        
        # Table surface
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'table'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = 0.4
        marker.pose.position.y = 0.0
        marker.pose.position.z = -0.025
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 0.05
        
        marker.color.r = 0.6
        marker.color.g = 0.4
        marker.color.b = 0.2
        marker.color.a = 0.8
        
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
    
    def run(self, duration, q_init):
        """
        Run the simulation loop.
        
        Args:
            duration: Simulation duration in seconds
            q_init: Initial joint configuration
        """
        print(f"\n[ROSSimulation] Starting simulation for {duration}s")
        
        # Initial state
        q = q_init.copy()
        dq = np.zeros(self.robot.nv)
        
        # Publish initial markers
        self.publish_table_marker()
        self.publish_trajectory_marker(self.trajectory)
        
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            t = (current_time - start_time).to_sec()
            
            if t >= duration:
                break
            
            # Get reference
            pose_ref, twist_ref, accel_ref, phase = self.trajectory.get_reference(t)
            
            # Compute control
            tau, info = self.controller.compute_control(q, dq, pose_ref, twist_ref, accel_ref)
            
            # Log
            self.logger.log(t, q, dq, tau, info, phase)
            
            # Forward dynamics (simulation)
            ddq = self.robot.forward_dynamics(q, dq, tau)
            
            # Euler integration
            dq = dq + ddq * self.dt
            q = q + dq * self.dt
            
            # Clip to joint limits
            q = self.robot.clip_to_limits(q)
            
            # Publish joint states
            self.publish_joint_states(q, dq, tau)
            
            # Print progress
            if int(t * 10) % 10 == 0 and abs(t - round(t)) < self.dt:
                print(f"  t={t:.1f}s | pos_err={info['error_pos_norm']:.4f}m | "
                      f"rot_err={info['error_rot_norm']:.4f}rad | "
                      f"manip={info['manipulability']:.4f}")
            
            self.rate.sleep()
        
        print("[ROSSimulation] Complete")
        return self.logger


class StandaloneSimulation:
    """
    Standalone simulation without ROS.
    """
    
    def __init__(self, robot, controller, trajectory, dt=0.001):
        """
        Initialize standalone simulation.
        """
        self.robot = robot
        self.controller = controller
        self.trajectory = trajectory
        self.dt = dt
        self.logger = SimulationLogger()
        
        print("[StandaloneSimulation] Initialized")
    
    def run(self, duration, q_init):
        """
        Run the simulation loop.
        """
        print(f"\n[StandaloneSimulation] Starting simulation for {duration}s")
        
        # Initial state
        q = q_init.copy()
        dq = np.zeros(self.robot.nv)
        
        num_steps = int(duration / self.dt)
        
        for i in range(num_steps):
            t = i * self.dt
            
            # Get reference
            pose_ref, twist_ref, accel_ref, phase = self.trajectory.get_reference(t)
            
            # Compute control
            tau, info = self.controller.compute_control(q, dq, pose_ref, twist_ref, accel_ref)
            
            # Log
            self.logger.log(t, q, dq, tau, info, phase)
            
            # Forward dynamics
            ddq = self.robot.forward_dynamics(q, dq, tau)
            
            # Euler integration
            dq = dq + ddq * self.dt
            q = q + dq * self.dt
            
            # Clip to joint limits
            q = self.robot.clip_to_limits(q)
            
            # Print progress
            if i % 1000 == 0:
                print(f"  t={t:.2f}s | pos_err={info['error_pos_norm']:.4f}m | "
                      f"rot_err={info['error_rot_norm']:.4f}rad | "
                      f"manip={info['manipulability']:.4f} | phase={phase}")
        
        print("[StandaloneSimulation] Complete")
        return self.logger


def run_comparison_experiment(robot, q_init, trajectory_params, controller_params, duration, dt=0.001):
    """
    Run comparison experiment: with vs without manipulability optimization.
    
    Returns:
        logger_with: Logger with manipulability optimization
        logger_without: Logger without manipulability optimization
    """
    print("\n" + "=" * 60)
    print("Running Comparison Experiment")
    print("=" * 60)
    
    # Experiment 1: WITH manipulability optimization
    print("\n--- Experiment 1: WITH manipulability optimization ---")
    
    trajectory1 = CleaningTrajectory(
        robot_model=robot,
        q_init=q_init,
        **trajectory_params
    )
    
    controller1 = TaskSpaceController(
        robot_model=robot,
        **controller_params,
        use_manipulability=True
    )
    
    sim1 = StandaloneSimulation(robot, controller1, trajectory1, dt)
    logger_with = sim1.run(duration, q_init)
    
    # Experiment 2: WITHOUT manipulability optimization
    print("\n--- Experiment 2: WITHOUT manipulability optimization ---")
    
    trajectory2 = CleaningTrajectory(
        robot_model=robot,
        q_init=q_init,
        **trajectory_params
    )
    
    controller2 = TaskSpaceController(
        robot_model=robot,
        **controller_params,
        use_manipulability=False
    )
    
    sim2 = StandaloneSimulation(robot, controller2, trajectory2, dt)
    logger_without = sim2.run(duration, q_init)
    
    return logger_with, logger_without


def main():
    parser = argparse.ArgumentParser(description='Cleaning Robot Simulation')
    parser.add_argument('--mode', type=str, default='standalone',
                        choices=['ros', 'standalone', 'compare'],
                        help='Simulation mode')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Simulation duration (seconds)')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Time step (seconds)')
    parser.add_argument('--radius', type=float, default=0.15,
                        help='Circle radius (meters)')
    parser.add_argument('--omega', type=float, default=0.3,
                        help='Angular velocity (rad/s)')
    parser.add_argument('--k_null', type=float, default=5.0,
                        help='Null-space gain')
    parser.add_argument('--output', type=str, default='simulation_data',
                        help='Output filename prefix')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cleaning Robot Simulation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Duration: {args.duration}s")
    print(f"Circle radius: {args.radius}m")
    print(f"Angular velocity: {args.omega} rad/s")
    
    # Load robot model
    robot = RobotModel()

    q_seed = np.array([
        0.0, 0.0,        # Platform (keep zero)
        0.0,             # z1_joint1 (Base Yaw)
        -0.5,            # z1_joint2 (Shoulder Pitch) - Tilt arm down/forward
        1.2,             # z1_joint3 (Elbow Pitch) - Bend elbow FORWARD (away from base)
        0.0,             # z1_joint4 (Wrist Yaw)
        -0.7,            # z1_joint5 (Wrist Pitch) - Compensate to keep tool down
        0.0              # z1_joint6 (Wrist Roll)
    ])
    
    # Get home configuration and check tool position/orientation
    q_home = robot.get_home_configuration()
    pos_home, rot_home = robot.get_tool_pose(q_home)
    print(f"\nHome config tool position: {pos_home}")
    print(f"Home config tool z-axis: {rot_home[:, 2]}")
    
    # Desired orientation: tool z-axis pointing DOWN (-Z world)
    # R_desired rotates so tool z points in -Z direction
    R_desired = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    
    # Set trajectory center based on reachable workspace
    # The arm at home extends ~0.55m in X, so center circle closer to base
    # Use a center that's reachable with the arm bent
    circle_center = [0.45, 0.0, 0.15]  # Fixed center within workspace
    
    # Circle start point
    circle_start = np.array([circle_center[0] + args.radius, circle_center[1], circle_center[2]])
    print(f"\nCircle center: {circle_center}")
    print(f"Circle start point: {circle_start}")
    
    # Use full pose IK to find configuration with correct position AND orientation
    q_init, ik_success = robot.compute_ik_for_pose(circle_start, R_desired, q_init=q_seed)
    if ik_success:
        print(f"IK converged to reach circle start with correct orientation")
    else:
        print(f"IK did not fully converge, using best estimate")
    
    pos_init, rot_init = robot.get_tool_pose(q_init)
    print(f"Initial config tool position: {pos_init}")
    print(f"Initial config tool z-axis: {rot_init[:, 2]}")
    print(f"Initial position error: {np.linalg.norm(circle_start - pos_init):.4f}m")
    
    # Trajectory parameters
    trajectory_params = {
        'circle_center': circle_center,
        'circle_radius': args.radius,
        'circle_height': 0.15,  # Fixed height above table
        'circle_omega': args.omega,
        'approach_duration': 3.0  # Allow time for approach
    }
    
    # Controller parameters - high gains for accurate tracking
    controller_params = {
        'Kp_pos': 800.0,
        'Kd_pos': 80.0,
        'Kp_rot': 600.0,
        'Kd_rot': 60.0,
        'k_null': min(args.k_null, 5.0),  # Limit null-space gain to prioritize tracking
        'damping': 0.05  # Higher damping for stability near singularities
    }
    
    if args.mode == 'compare':
        # Run comparison experiment
        logger_with, logger_without = run_comparison_experiment(
            robot, q_init, trajectory_params, controller_params,
            args.duration, args.dt
        )
        
        # Save both logs
        logger_with.save(f'{args.output}_with_manip.npz')
        logger_without.save(f'{args.output}_without_manip.npz')
        
        print("\n[Main] Comparison experiment complete")
        print(f"  Data saved to: {args.output}_with_manip.npz")
        print(f"                 {args.output}_without_manip.npz")
        
    elif args.mode == 'ros':
        if not ROS_AVAILABLE:
            print("[Error] ROS not available. Use --mode standalone")
            sys.exit(1)
        
        # Create trajectory
        trajectory = CleaningTrajectory(
            robot_model=robot,
            q_init=q_init,
            **trajectory_params
        )
        
        # Create controller
        controller = TaskSpaceController(
            robot_model=robot,
            **controller_params,
            use_manipulability=True
        )
        
        # Run ROS simulation
        sim = ROSSimulation(robot, controller, trajectory, args.dt)
        logger = sim.run(args.duration, q_init)
        
        # Save log
        logger.save(f'{args.output}.npz')
        
    else:  # standalone
        # Create trajectory
        trajectory = CleaningTrajectory(
            robot_model=robot,
            q_init=q_init,
            **trajectory_params
        )
        
        # Create controller
        controller = TaskSpaceController(
            robot_model=robot,
            **controller_params,
            use_manipulability=True
        )
        
        # Run standalone simulation
        sim = StandaloneSimulation(robot, controller, trajectory, args.dt)
        logger = sim.run(args.duration, q_init)
        
        # Save log
        logger.save(f'{args.output}.npz')
    
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
