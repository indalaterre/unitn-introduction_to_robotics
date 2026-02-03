"""
Trajectory Generator Module
===========================
Generates reference trajectories for the cleaning robot:
- Horizontal circular trajectory of radius 0.20 m
- End-effector perpendicular to table (tool z-axis pointing down)
- Outputs: SE3 pose, twist, and task acceleration references
"""

import numpy as np
import pinocchio as pin


class CircularTrajectory:
    """
    Generates a horizontal circular trajectory for table cleaning.
    
    The trajectory is a circle in the XY plane (table surface) with:
    - Center at p0 (configurable)
    - Radius r = 0.20 m
    - Constant height z above table
    - End-effector orientation: tool z-axis pointing DOWN (-Z world)
    """
    
    def __init__(self, center=None, radius=0.20, height=0.05, omega=0.5):
        """
        Initialize the circular trajectory generator.
        
        Args:
            center: (3,) Center point of circle in world frame [x, y, z]
                   If None, defaults to [0.4, 0.0, height]
            radius: Circle radius in meters (default 0.20 m)
            height: Height above table surface (default 0.05 m)
            omega: Angular velocity in rad/s (default 0.5 rad/s)
        """
        self.radius = radius
        self.height = height
        self.omega = omega
        
        if center is None:
            # Default center: in front of robot, at the specified height
            self.center = np.array([0.4, 0.0, self.height])
        else:
            self.center = np.array(center)
            self.center[2] = self.height  # Ensure the correct height
        
        # Desired orientation: tool z-axis pointing DOWN
        # This means R_des rotates the world frame so that:
        # - Tool x-axis points in the world +X (forward)
        # - Tool y-axis points in the world +Y (left)
        # - Tool z-axis points in world -Z (down toward table)
        # R_des = Ry(pi) to flip z-axis
        self.R_desired = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])
        
        # Period of one full circle
        self.period = 2 * np.pi / self.omega
        
        print(f"[Trajectory] Circular trajectory initialized:")
        print(f"  Center: {self.center}")
        print(f"  Radius: {self.radius} m")
        print(f"  Height: {self.height} m")
        print(f"  Angular velocity: {self.omega} rad/s")
        print(f"  Period: {self.period:.2f} s")
    
    def get_pose_reference(self, t):
        """
        Get reference SE3 pose at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            pose: pinocchio.SE3 pose
        """

        pos_reference = self.get_position_reference(t)
        rot_reference = self.get_orientation_reference(t)
        return pin.SE3(rot_reference, pos_reference)
    
    def get_position_reference(self, t):
        """
        Get reference position at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            p_ref: (3,) position vector
        """
        theta = self.omega * t
        
        p = self.center.copy()
        p[0] += self.radius * np.cos(theta)
        p[1] += self.radius * np.sin(theta)
        
        return p
    
    def get_orientation_reference(self, t):
        """
        Get reference orientation at time t.
        For cleaning, orientation is constant (perpendicular to table).
        
        Args:
            t: Time in seconds
            
        Returns:
            R_ref: (3,3) rotation matrix
        """
        return self.R_desired.copy()
    
    def get_twist_reference(self, t):
        """
        Get reference twist (spatial velocity) at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            twist: (6,) vector [linear_vel; angular_vel]
        """
        theta = self.omega * t
        
        # Linear velocity: derivative of position
        # p = center + [r*cos(θ), r*sin(θ), 0]
        # dp/dt = [-r*ω*sin(θ), r*ω*cos(θ), 0]
        v_linear = np.array([
            -self.radius * self.omega * np.sin(theta),
             self.radius * self.omega * np.cos(theta),
             0.0
        ])
        
        # Angular velocity: zero (constant orientation)
        v_angular = np.zeros(3)
        
        return np.concatenate([v_linear, v_angular])
    
    def get_acceleration_reference(self, t):
        """
        Get reference task acceleration at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            accel: (6,) vector [linear_accel; angular_accel]
        """
        theta = self.omega * t
        
        # Linear acceleration: second derivative of position
        # d²p/dt² = [-r*ω²*cos(θ), -r*ω²*sin(θ), 0]
        a_linear = np.array([
            -self.radius * self.omega**2 * np.cos(theta),
            -self.radius * self.omega**2 * np.sin(theta),
             0.0
        ])
        
        # Angular acceleration: zero (constant orientation)
        a_angular = np.zeros(3)
        
        return np.concatenate([a_linear, a_angular])
    
    def get_full_reference(self, t):
        """
        Get complete reference: pose, twist, and acceleration.
        
        Args:
            t: Time in seconds
            
        Returns:
            pose: pinocchio.SE3
            twist: (6,) twist vector
            accel: (6,) acceleration vector
        """
        pose = self.get_pose_reference(t)
        twist = self.get_twist_reference(t)
        accel = self.get_acceleration_reference(t)
        
        return pose, twist, accel


class PointToPointTrajectory:
    """
    Smooth point-to-point trajectory using quintic polynomial.
    Used for the initial approach to the circle.
    """
    
    def __init__(self, p_start, p_end, R_start, R_end, duration):
        """
        Initialize the point-to-point trajectory.
        
        Args:
            p_start: (3,) starting position
            p_end: (3,) ending position
            R_start: (3,3) starting rotation
            R_end: (3,3) ending rotation
            duration: Time duration in seconds
        """
        self.p_start = np.array(p_start)
        self.p_end = np.array(p_end)
        self.R_start = np.array(R_start)
        self.R_end = np.array(R_end)
        self.duration = duration
        
        # Convert rotations to quaternions for interpolation
        self.q_start = pin.Quaternion(self.R_start)
        self.q_end = pin.Quaternion(self.R_end)
        
        # Ensure quaternions are normalized and use shortest path
        if self.q_start.dot(self.q_end) < 0:
            self.q_end.coeffs *= -1  # Take shortest path
    
    def _quintic_profile(self, t):
        """
        Quintic polynomial s(t) with s(0)=0, s(T)=1, zero vel/accel at endpoints.
        
        Returns:
            s: Position profile [0, 1]
            ds: Velocity profile
            dds: Acceleration profile
        """
        if t <= 0:
            return 0.0, 0.0, 0.0
        elif t >= self.duration:
            return 1.0, 0.0, 0.0
        
        T = self.duration
        tau = t / T
        
        # s = 10τ³ - 15τ⁴ + 6τ⁵
        tau2 = tau**2
        tau3 = tau**3
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        s = 10 * tau3 - 15 * tau4 + 6 * tau5
        ds = (30 * tau2 - 60 * tau3 + 30 * tau4) / T
        dds = (60 * tau - 180 * tau2 + 120 * tau3) / T**2
        
        return s, ds, dds
    
    def get_pose_reference(self, t):
        """Get reference pose at time t."""
        s, _, _ = self._quintic_profile(t)
        
        # Interpolate position
        p = self.p_start + s * (self.p_end - self.p_start)
        
        # Interpolate rotation using quaternion SLERP
        q_interp = self.q_start.slerp(self.q_end, s)
        R = q_interp.toRotationMatrix()
        
        return pin.SE3(R, p)
    
    def get_twist_reference(self, t):
        """Get reference twist at time t."""
        s, ds, _ = self._quintic_profile(t)
        
        # Linear velocity
        v_linear = ds * (self.p_end - self.p_start)
        
        # Angular velocity from quaternion derivative
        # For SLERP: ω = 2 * q̇ * q* (where q* is quaternion conjugate)
        q_current = self.q_start.slerp(self.q_end, s)
        # Approximate quaternion derivative for SLERP
        q_dot = (self.q_end - self.q_start) / self.duration * ds
        omega_quat = 2.0 * pin.Quaternion(q_dot.coeffs) * q_current.conjugate()
        v_angular = omega_quat.vec()  # Extract vector part (angular velocity)
        
        return np.concatenate([v_linear, v_angular])
    
    def get_acceleration_reference(self, t):
        """Get reference acceleration at time t."""
        s, ds, dds = self._quintic_profile(t)
        
        # Linear acceleration
        a_linear = dds * (self.p_end - self.p_start)
        
        # Angular acceleration from quaternion second derivative
        q_current = self.q_start.slerp(self.q_end, s)
        # Approximate quaternion second derivative for SLERP
        q_ddot = (self.q_end - self.q_start) / self.duration * dds
        alpha_quat = 2.0 * pin.Quaternion(q_ddot.coeffs) * q_current.conjugate()
        a_angular = alpha_quat.vec()  # Extract vector part (angular acceleration)
        
        return np.concatenate([a_linear, a_angular])
    
    def get_full_reference(self, t):
        """Get complete reference."""
        pose = self.get_pose_reference(t)
        twist = self.get_twist_reference(t)
        accel = self.get_acceleration_reference(t)
        return pose, twist, accel


class CartesianTrajectory:
    """
    Linear (Cartesian) point-to-point trajectory with trapezoidal velocity profile.
    Uses smooth acceleration ramps at start and end to avoid velocity discontinuities.
    """
    
    def __init__(self, p_start, p_end, R_start, R_end, duration, accel_time=0.5):
        """
        Initialize Cartesian trajectory with trapezoidal velocity profile.
        
        Args:
            p_start: (3,) starting position
            p_end: (3,) ending position
            R_start: (3,3) starting rotation
            R_end: (3,3) ending rotation
            duration: Time duration in seconds
            accel_time: Time for acceleration/deceleration ramps (default 0.5s)
        """
        self.p_start = np.array(p_start)
        self.p_end = np.array(p_end)
        self.R_start = np.array(R_start)
        self.R_end = np.array(R_end)
        self.duration = duration
        
        # Limit acceleration time to at most half the duration
        self.t_accel = min(accel_time, duration / 2.0)
        self.t_decel = self.t_accel
        self.t_const = duration - self.t_accel - self.t_decel
        
        # Convert rotations to quaternions for interpolation
        self.q_start = pin.Quaternion(self.R_start)
        self.q_end = pin.Quaternion(self.R_end)
        
        # Ensure quaternions are normalized and use shortest path
        if self.q_start.dot(self.q_end) < 0:
            self.q_end.coeffs *= -1  # Take shortest path
        
        # Compute maximum velocity during constant phase
        # Total distance = area under velocity trapezoid
        # s = 0.5 * v_max * t_accel + v_max * t_const + 0.5 * v_max * t_decel
        # s = v_max * (duration - 0.5 * t_accel - 0.5 * t_decel)
        total_distance = np.linalg.norm(self.p_end - self.p_start)
        if total_distance > 1e-6:
            self.v_max = total_distance / (duration - 0.5 * self.t_accel - 0.5 * self.t_decel)
            self.direction = (self.p_end - self.p_start) / total_distance
        else:
            self.v_max = 0.0
            self.direction = np.zeros(3)

        # Quaternion interpolation setup
        # Store quaternion difference for angular velocity calculations
        self.q_diff = self.q_start.conjugate() * self.q_end
        self.total_rotation_angle = 2.0 * np.arccos(np.clip(abs(self.q_diff.w), -1.0, 1.0))
    
    def _get_velocity_profile(self, t):
        """
        Compute velocity and acceleration for trapezoidal profile.
        
        Returns:
            v: velocity magnitude [0, v_max]
            a: acceleration magnitude
        """
        if t <= 0:
            return 0.0, 0.0
        elif t >= self.duration:
            return 0.0, 0.0
        elif t < self.t_accel:
            # Acceleration phase: linear ramp up
            a = self.v_max / self.t_accel
            v = a * t
            return v, a
        elif t < self.t_accel + self.t_const:
            # Constant velocity phase
            return self.v_max, 0.0
        else:
            # Deceleration phase: linear ramp down
            t_decel_phase = t - (self.t_accel + self.t_const)
            a = -self.v_max / self.t_decel
            v = self.v_max + a * t_decel_phase
            return v, a
    
    def _get_position_from_velocity(self, t):
        """
        Integrate velocity profile to get position along trajectory.
        
        Returns:
            s: position parameter [0, 1]
        """
        if t <= 0:
            return 0.0
        elif t >= self.duration:
            return 1.0
        elif t < self.t_accel:
            # Acceleration phase: s = 0.5 * a * t^2
            a = self.v_max / self.t_accel
            distance = 0.5 * a * t**2
        elif t < self.t_accel + self.t_const:
            # Constant velocity phase
            distance_accel = 0.5 * self.v_max * self.t_accel
            distance_const = self.v_max * (t - self.t_accel)
            distance = distance_accel + distance_const
        else:
            # Deceleration phase
            t_decel_phase = t - (self.t_accel + self.t_const)
            distance_accel = 0.5 * self.v_max * self.t_accel
            distance_const = self.v_max * self.t_const
            distance_decel = self.v_max * t_decel_phase - 0.5 * (self.v_max / self.t_decel) * t_decel_phase**2
            distance = distance_accel + distance_const + distance_decel
        
        # Normalize to [0, 1]
        total_distance = np.linalg.norm(self.p_end - self.p_start)
        if total_distance > 1e-6:
            return distance / total_distance
        else:
            return 1.0 if t >= self.duration else 0.0
    
    def get_pose_reference(self, t):
        """Get reference pose at time t using trapezoidal velocity profile."""
        # Get position parameter from velocity integration
        s = self._get_position_from_velocity(t)
        
        # Interpolate position
        p = self.p_start + s * (self.p_end - self.p_start)
        
        # Interpolate rotation using quaternion SLERP
        q_interp = self.q_start.slerp(self.q_end, s)
        R = q_interp.toRotationMatrix()
        
        return pin.SE3(R, p)
    
    def get_twist_reference(self, t):
        """Get reference twist at time t (trapezoidal velocity profile)."""
        v, _ = self._get_velocity_profile(t)
        
        # Linear velocity
        v_linear = v * self.direction
        
        # Angular velocity from quaternion SLERP
        s = self._get_position_from_velocity(t)
        q_current = self.q_start.slerp(self.q_end, s)
        # Angular velocity magnitude based on trapezoidal profile
        if self.total_rotation_angle > 1e-6 and self.duration > 0:
            omega_mag = self.total_rotation_angle / (self.duration - 0.5 * self.t_accel - 0.5 * self.t_decel)
            omega_mag *= (v / self.v_max) if self.v_max > 1e-6 else 0.0
            # Convert to angular velocity vector
            q_dot = pin.Quaternion(0, *self.q_diff.vec()) * omega_mag / self.total_rotation_angle
            omega_quat = 2.0 * q_dot * q_current.conjugate()
            v_angular = omega_quat.vec()
        else:
            v_angular = np.zeros(3)
        
        return np.concatenate([v_linear, v_angular])
    
    def get_acceleration_reference(self, t):
        """Get reference acceleration at time t (trapezoidal profile)."""
        _, a = self._get_velocity_profile(t)
        
        # Linear acceleration
        a_linear = a * self.direction
        
        # Angular acceleration from quaternion SLERP
        s = self._get_position_from_velocity(t)
        q_current = self.q_start.slerp(self.q_end, s)
        # Angular acceleration magnitude based on trapezoidal profile
        if self.total_rotation_angle > 1e-6 and self.duration > 0:
            alpha_mag = self.total_rotation_angle / (self.duration - 0.5 * self.t_accel - 0.5 * self.t_decel)
            alpha_mag *= (a / self.v_max) if self.v_max > 1e-6 else 0.0
            # Convert to angular acceleration vector
            q_ddot = pin.Quaternion(0, *self.q_diff.vec()) * alpha_mag / self.total_rotation_angle
            alpha_quat = 2.0 * q_ddot * q_current.conjugate()
            a_angular = alpha_quat.vec()
        else:
            a_angular = np.zeros(3)
        
        return np.concatenate([a_linear, a_angular])
    
    def get_full_reference(self, t):
        """Get complete reference."""
        pose = self.get_pose_reference(t)
        twist = self.get_twist_reference(t)
        accel = self.get_acceleration_reference(t)
        return pose, twist, accel


class CleaningTrajectory:
    """
    Complete cleaning trajectory manager.
    Combines approach phase and circular cleaning phase.
    """
    
    def __init__(self, robot_model, q_init, 
                 circle_center=None, circle_radius=0.20, 
                 circle_height=0.05, circle_omega=0.5,
                 approach_duration=3.0,
                 use_quintic=True):
        """
        Initialize cleaning trajectory.
        
        Args:
            robot_model: RobotModel instance
            q_init: Initial joint configuration
            circle_center: Center of cleaning circle
            circle_radius: Radius of cleaning circle
            circle_height: Height above table
            circle_omega: Angular velocity for circle
            approach_duration: Duration of approach phase
        """
        self.robot = robot_model
        self.use_quintic = use_quintic
        self.approach_duration = approach_duration
        
        # Create circular trajectory
        self.circle_traj = CircularTrajectory(
            center=circle_center,
            radius=circle_radius,
            height=circle_height,
            omega=circle_omega
        )
        
        # Get starting point on circle (t=0)
        circle_start_pose = self.circle_traj.get_pose_reference(0)

        # Get initial pose from the robot
        p_init, R_init = self.robot.get_tool_pose(q_init)

        # Create the approach trajectory based on the interpolation type
        if self.use_quintic:
            # Quintic polynomial (smooth acceleration)
            self.approach_traj = PointToPointTrajectory(
                p_start=p_init,
                p_end=circle_start_pose.translation,
                R_start=R_init,
                R_end=circle_start_pose.rotation,
                duration=approach_duration
            )
            print(f"[CleaningTrajectory] Using QUINTIC polynomial interpolation")
        else:
            # Cartesian linear interpolation (constant velocity)
            self.approach_traj = CartesianTrajectory(
                p_start=p_init,
                p_end=circle_start_pose.translation,
                R_start=R_init,
                R_end=circle_start_pose.rotation,
                duration=approach_duration
            )
            print(f"[CleaningTrajectory] Using CARTESIAN linear interpolation")
        
        print(f"[CleaningTrajectory] Initialized:")
        print(f"  Approach duration: {approach_duration} s")
        print(f"  Circle start: {circle_start_pose.translation}")
        print(f"  Initial position: {p_init}")
    
    def get_reference(self, t):
        """
        Get reference at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            pose: SE3 reference pose
            twist: (6,) reference twist
            accel: (6,) reference acceleration
            phase: 'approach' or 'circle'
        """
        if t < self.approach_duration:
            # Approach phase (quintic or Cartesian)
            phase = 'approach'
            pose, twist, accel = self.approach_traj.get_full_reference(t)
        else:
            # Circular cleaning phase
            t_circle = t - self.approach_duration
            pose, twist, accel = self.circle_traj.get_full_reference(t_circle)
            phase = 'circle'
        
        return pose, twist, accel, phase


def compute_pose_error(pose_current, pose_desired):
    """
    Compute pose error between current and desired SE3 poses.
    
    Args:
        pose_current: Current SE3 pose
        pose_desired: Desired SE3 pose
        
    Returns:
        e_pos: (3,) position error
        e_rot: (3,) orientation error (axis-angle from log map)
    """
    # Position error
    e_pos = pose_desired.translation - pose_current.translation
    
    # Orientation error using SO(3) log map
    # e_R = log(R_des * R_cur^T)
    R_error = pose_desired.rotation @ pose_current.rotation.T
    e_rot = pin.log3(R_error)
    
    return e_pos, e_rot


def test_trajectory():
    """Test trajectory generation."""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing Trajectory Generator")
    print("=" * 60)
    
    # Create circular trajectory
    traj = CircularTrajectory(
        center=[0.4, 0.0, 0.05],
        radius=0.20,
        omega=0.5
    )
    
    # Generate trajectory for 2 periods
    T = 2 * traj.period
    dt = 0.01
    t_array = np.arange(0, T, dt)
    
    positions = []
    velocities = []
    accelerations = []
    
    for t in t_array:
        pose, twist, accel = traj.get_full_reference(t)
        positions.append(pose.translation.copy())
        velocities.append(twist[:3].copy())
        accelerations.append(accel[:3].copy())
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.scatter(traj.center[0], traj.center[1], c='r', marker='x', s=100, label='Center')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Circular Trajectory (XY Plane)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    
    # Position vs time
    ax = axes[0, 1]
    ax.plot(t_array, positions[:, 0], 'r-', label='X')
    ax.plot(t_array, positions[:, 1], 'g-', label='Y')
    ax.plot(t_array, positions[:, 2], 'b-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position vs Time')
    ax.grid(True)
    ax.legend()
    
    # Velocity vs time
    ax = axes[1, 0]
    ax.plot(t_array, velocities[:, 0], 'r-', label='Vx')
    ax.plot(t_array, velocities[:, 1], 'g-', label='Vy')
    ax.plot(t_array, velocities[:, 2], 'b-', label='Vz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity vs Time')
    ax.grid(True)
    ax.legend()
    
    # Acceleration vs time
    ax = axes[1, 1]
    ax.plot(t_array, accelerations[:, 0], 'r-', label='Ax')
    ax.plot(t_array, accelerations[:, 1], 'g-', label='Ay')
    ax.plot(t_array, accelerations[:, 2], 'b-', label='Az')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Acceleration vs Time')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_test.png', dpi=150)
    print("\nTrajectory plot saved to trajectory_test.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Trajectory Test Complete")
    print("=" * 60)


if __name__ == '__main__':
    test_trajectory()
