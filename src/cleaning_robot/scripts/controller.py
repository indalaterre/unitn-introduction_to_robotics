"""
Task-Space Inverse Dynamics Controller
=======================================
Implements:
1. Primary task: Pose tracking using task-space inverse dynamics
2. Secondary task: Manipulability maximization in null space

Controller equation:
    tau = M(q) * qddot + h(q, qdot)

where:
    qddot = qddot_task + qddot_null
    qddot_task = J# * (xddot* - Jdot*qdot)
    qddot_null = k_null * N * grad(w)
    
    xddot* = xddot_ref + Kd*(xdot_ref - xdot) + Kp*e
    e = [e_pos; e_rot]  (position and orientation errors)
"""

import numpy as np
import pinocchio as pin
from trajectory import compute_pose_error


class TaskSpaceController:
    """
    Task-space inverse dynamics controller with null-space optimization.
    """
    
    def __init__(self, robot_model, 
                 Kp_pos=100.0, Kd_pos=20.0,
                 Kp_rot=100.0, Kd_rot=20.0,
                 k_null=5.0,
                 damping=1e-3,
                 use_manipulability=True):
        """
        Initialize controller.
        
        Args:
            robot_model: RobotModel instance
            Kp_pos: Position proportional gain
            Kd_pos: Position derivative gain
            Kp_rot: Orientation proportional gain
            Kd_rot: Orientation derivative gain
            k_null: Null-space gain for manipulability
            damping: Damping factor for pseudoinverse
            use_manipulability: Enable/disable null-space optimization
        """
        self.robot = robot_model
        
        # Task-space gains (6x6 diagonal)
        self.Kp = np.diag([Kp_pos, Kp_pos, Kp_pos, Kp_rot, Kp_rot, Kp_rot])
        self.Kd = np.diag([Kd_pos, Kd_pos, Kd_pos, Kd_rot, Kd_rot, Kd_rot])
        
        # Null-space gain
        self.k_null = k_null
        
        # Damping for pseudoinverse
        self.damping = damping
        
        # Enable/disable manipulability optimization
        self.use_manipulability = use_manipulability
        
        # Safety limits
        self.tau_max = self.robot.tau_max
        self.ddq_max = 50.0  # Max joint acceleration
        
        # Logging
        self.last_error_pos = np.zeros(3)
        self.last_error_rot = np.zeros(3)
        self.last_manipulability = 0.0
        self.last_tau = np.zeros(self.robot.nq)

        self.q_posture = np.array([
            0.0, 0.0,        # Platform
            0.0,             # Base Yaw
            0.5,             # Shoulder Pitch (Up)
            -1.2,            # Elbow Pitch (Bent)
            0.0,             # Wrist Yaw
            0.7,             # Wrist Pitch
            0.0              # Wrist Roll
        ])
        self.kp_posture = 100
        self.kd_posture = 10
        
        print(f"[Controller] Initialized:")
        print(f"  Kp_pos: {Kp_pos}, Kd_pos: {Kd_pos}")
        print(f"  Kp_rot: {Kp_rot}, Kd_rot: {Kd_rot}")
        print(f"  k_null: {k_null}")
        print(f"  Manipulability optimization: {use_manipulability}")

        # --- DEBUG PRINT TO VERIFY RELOAD ---
        print(f"[Controller] RE-INITIALIZED with CONSTANT damping: {damping}")

        self.print_timer = 0
    
    def compute_control(self, q, dq, pose_ref, twist_ref, accel_ref):
        # Update robot kinematics
        self.robot.update_kinematics(q, dq)
        pose_current = self.robot.forward_kinematics(q)
        e_pos, e_rot = compute_pose_error(pose_current, pose_ref)
        error = np.concatenate([e_pos, e_rot])
        J = self.robot.get_jacobian(q)

        # <--- FIX 2: CALCULATE DAMPING FIRST (Before Primary Task)
        # Previously, you calculated this too late. Now we do it first.
        manip = self.robot.compute_manipulability(q, use_position_only=True)
        current_damping = self.damping
        
        # Dynamic Damping: If manip < 0.05, increase damping to prevent explosion
        if manip < 0.05:
            current_damping = 0.01 + (0.05 - manip) * 20.0
            
        # Create the SAFE Pseudoinverse using the calculated damping
        J_pinv = self.robot.damped_pseudoinverse(J, current_damping)

        # --- PRIMARY TASK (Cleaning) ---
        xdot = J @ dq
        xdot_error = twist_ref - xdot
        xddot_star = accel_ref + self.Kd @ xdot_error + self.Kp @ error
        Jdot_qdot = self.robot.get_jacobian_derivative(q, dq)
        
        # Now the primary task uses the SAFE J_pinv
        qddot_task = J_pinv @ (xddot_star - Jdot_qdot)
        
        # --- FIX 3: UNIFIED SECONDARY TASK (Posture + Manipulability) ---
        qddot_null = np.zeros(self.robot.nq)
        
        if self.use_manipulability:
            # Step A: Calculate Posture Correction (Pull towards Elbow-Up)
            # q_accel = Kp*(q_des - q) - Kd*dq
            qddot_secondary = self.kp_posture * (self.q_posture - q) - self.kd_posture * dq
            
            # Step B: Add Manipulability Optimization (Optional)
            # Only add this if we are safe (manip > 0.04) and gain > 0
            if self.k_null > 0 and manip > 0.04:
                try:
                    grad_w = self.robot.compute_manipulability_gradient(q, use_position_only=True)
                    qddot_secondary += self.k_null * grad_w
                except:
                    pass

            # Step C: Project EVERYTHING into Null Space at once
            N = self.robot.null_space_projector(J, self.damping)
            qddot_null_raw = N @ qddot_secondary

            # Step D: Safety Clamp
            max_null_ac = .5
            norm_null = np.linalg.norm(qddot_null_raw)
            if norm_null > max_null_ac:
                qddot_null = qddot_null_raw * (max_null_ac / norm_null)
            else:
                qddot_null = qddot_null_raw
        
        # Summation
        qddot = qddot_task + qddot_null
        qddot = np.clip(qddot, -self.ddq_max, self.ddq_max)
        
        # Dynamics
        M = self.robot.get_mass_matrix(q)
        h = self.robot.get_bias_forces(q, dq)
        
        if np.any(np.isnan(M)) or np.any(np.isnan(h)):
            tau = np.zeros(self.robot.nq)
        else:
            tau = M @ qddot + h
            
        tau = self.robot.saturate_torques(tau)
        
        # Store Info
        self.last_error_pos = e_pos
        self.last_error_rot = e_rot
        self.last_manipulability = manip
        self.last_tau = tau
        
        info = {
            'error_pos': e_pos.copy(),
            'error_rot': e_rot.copy(),
            'error_pos_norm': np.linalg.norm(e_pos),
            'error_rot_norm': np.linalg.norm(e_rot),
            'manipulability': manip,
            'qddot_task': qddot_task.copy(),
            'qddot_null': qddot_null.copy(),
            'qddot': qddot.copy(),
            'tau': tau.copy(),
            'xdot': xdot.copy(),
            'xdot_ref': twist_ref.copy(),
            'pose_current': pose_current,
            'pose_ref': pose_ref
        }
        
        return tau, info

    def set_gains(self, Kp_pos=None, Kd_pos=None, Kp_rot=None, Kd_rot=None, k_null=None):
        """Update controller gains."""
        if Kp_pos is not None:
            self.Kp[:3, :3] = Kp_pos * np.eye(3)
        if Kd_pos is not None:
            self.Kd[:3, :3] = Kd_pos * np.eye(3)
        if Kp_rot is not None:
            self.Kp[3:, 3:] = Kp_rot * np.eye(3)
        if Kd_rot is not None:
            self.Kd[3:, 3:] = Kd_rot * np.eye(3)
        if k_null is not None:
            self.k_null = k_null
    
    def enable_manipulability(self, enable=True):
        """Enable or disable manipulability optimization."""
        self.use_manipulability = enable


class SimulationLogger:
    """
    Logger for simulation data.
    """
    
    def __init__(self):
        """Initialize empty log."""
        self.reset()
    
    def reset(self):
        """Clear all logged data."""
        self.data = {
            'time': [],
            'q': [],
            'dq': [],
            'tau': [],
            'error_pos': [],
            'error_rot': [],
            'error_pos_norm': [],
            'error_rot_norm': [],
            'manipulability': [],
            'pose_ref_pos': [],
            'pose_cur_pos': [],
            'phase': []
        }
    
    def log(self, t, q, dq, tau, info, phase=''):
        """
        Log simulation step.
        
        Args:
            t: Time
            q: Joint positions
            dq: Joint velocities
            tau: Joint torques
            info: Controller info dictionary
            phase: Trajectory phase string
        """
        self.data['time'].append(t)
        self.data['q'].append(q.copy())
        self.data['dq'].append(dq.copy())
        self.data['tau'].append(tau.copy())
        self.data['error_pos'].append(info['error_pos'].copy())
        self.data['error_rot'].append(info['error_rot'].copy())
        self.data['error_pos_norm'].append(info['error_pos_norm'])
        self.data['error_rot_norm'].append(info['error_rot_norm'])
        self.data['manipulability'].append(info['manipulability'])
        self.data['pose_ref_pos'].append(info['pose_ref'].translation.copy())
        self.data['pose_cur_pos'].append(info['pose_current'].translation.copy())
        self.data['phase'].append(phase)
    
    def to_arrays(self):
        """Convert lists to numpy arrays."""
        result = {}
        for key, value in self.data.items():
            if key == 'phase':
                result[key] = value
            else:
                result[key] = np.array(value)
        return result
    
    def save(self, filename):
        """Save logged data to file."""
        arrays = self.to_arrays()
        np.savez(filename, **arrays)
        print(f"[Logger] Data saved to {filename}")
    
    @staticmethod
    def load(filename):
        """Load logged data from file."""
        data = np.load(filename, allow_pickle=True)
        return dict(data)


def simulate_standalone(robot, controller, trajectory, 
                        q_init, duration, dt=0.001):
    """
    Run standalone simulation (no ROS/Locosim).
    
    Args:
        robot: RobotModel instance
        controller: TaskSpaceController instance
        trajectory: CleaningTrajectory instance
        q_init: Initial joint configuration
        duration: Simulation duration in seconds
        dt: Time step
        
    Returns:
        logger: SimulationLogger with recorded data
    """
    logger = SimulationLogger()
    
    # Initial state
    q = q_init.copy()
    dq = np.zeros(robot.nv)
    
    num_steps = int(duration / dt)
    
    print(f"\n[Simulation] Starting standalone simulation:")
    print(f"  Duration: {duration} s")
    print(f"  Time step: {dt} s")
    print(f"  Steps: {num_steps}")
    
    for i in range(num_steps):
        t = i * dt
        
        # Get reference
        pose_ref, twist_ref, accel_ref, phase = trajectory.get_reference(t)
        
        # Compute control
        tau, info = controller.compute_control(q, dq, pose_ref, twist_ref, accel_ref)
        
        # Log
        logger.log(t, q, dq, tau, info, phase)
        
        # Forward dynamics
        ddq = robot.forward_dynamics(q, dq, tau)
        
        # Euler integration
        dq = dq + ddq * dt
        q = q + dq * dt
        
        # Clip to joint limits
        q = robot.clip_to_limits(q)
        
        # Print progress
        if i % 1000 == 0:
            print(f"  t={t:.2f}s | pos_err={info['error_pos_norm']:.4f}m | "
                  f"rot_err={info['error_rot_norm']:.4f}rad | "
                  f"manip={info['manipulability']:.4f} | phase={phase}")
    
    print("[Simulation] Complete")
    return logger


def test_controller():
    """Test the controller with standalone simulation."""
    from robot_model import RobotModel
    from trajectory import CleaningTrajectory
    
    print("=" * 60)
    print("Testing Task-Space Controller")
    print("=" * 60)
    
    # Load robot
    robot = RobotModel()
    
    # Initial configuration
    q_init = robot.get_home_configuration()
    
    # Create trajectory
    trajectory = CleaningTrajectory(
        robot_model=robot,
        q_init=q_init,
        circle_center=[0.35, 0.0, 0.05],
        circle_radius=0.15,
        circle_omega=0.3,
        approach_duration=3.0
    )
    
    # Create controller
    controller = TaskSpaceController(
        robot_model=robot,
        Kp_pos=200.0,
        Kd_pos=40.0,
        Kp_rot=150.0,
        Kd_rot=30.0,
        k_null=10.0,
        use_manipulability=True
    )
    
    # Run simulation
    duration = 15.0  # 3s approach + ~12s cleaning
    logger = simulate_standalone(robot, controller, trajectory, q_init, duration, dt=0.001)
    
    # Save data
    logger.save('simulation_data.npz')
    
    print("\n" + "=" * 60)
    print("Controller Test Complete")
    print("=" * 60)
    
    return logger


if __name__ == '__main__':
    test_controller()
