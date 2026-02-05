import os
import numpy as np
import pinocchio as pin


class RobotModel:
    """
    Wrapper class for Pinocchio robot model with utility methods.
    """
    
    def __init__(self, urdf_path=None):
        """
        Initialize the robot model from URDF.
        
        Args:
            urdf_path: Path to URDF file. If None, uses default assembly.urdf
        """
        if urdf_path is None:
            urdf_path = os.path.join(os.path.dirname(__file__),  '../urdf/assembly.urdf')
        
        self.urdf_path = os.path.abspath(urdf_path)
        
        # Build Pinocchio model
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        
        # Get frame IDs
        self.ee_frame_id = self.model.getFrameId('ee_link')
        self.tool_tip_frame_id = self.model.getFrameId('tool_tip')
        self.table_frame_id = self.model.getFrameId('table')
        
        # Number of DOFs
        self.nq = self.model.nq  # Should be 8
        self.nv = self.model.nv  # Should be 8
        
        # Joint names for reference
        self.joint_names = [
            'platform_roll', 'platform_pitch',
            'z1_joint1', 'z1_joint2', 'z1_joint3',
            'z1_joint4', 'z1_joint5', 'z1_joint6'
        ]
        
        # Joint limits
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()
        
        # Effort limits (from URDF)
        self.tau_max = np.array([50.0, 50.0,  # platform
                                 33.5, 33.5, 33.5,  # Z1 shoulder/elbow
                                 6.0, 6.0, 6.0])  # Z1 wrist
        
        print(f"[RobotModel] Loaded URDF: {self.urdf_path}")
        print(f"[RobotModel] DOFs: nq={self.nq}, nv={self.nv}")
        print(f"[RobotModel] Tool tip frame ID: {self.tool_tip_frame_id}")
    
    def update_kinematics(self, q, dq=None):
        """
        Update all kinematic quantities for given configuration.
        
        Args:
            q: Joint positions (8,)
            dq: Joint velocities (8,), optional
        """
        if dq is None:
            dq = np.zeros(self.nv)
        
        # Compute all terms (kinematics + dynamics)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
    
    def forward_kinematics(self, q):
        """
        Compute forward kinematics for tool tip.
        
        Args:
            q: Joint positions (8,)
            
        Returns:
            SE3 pose of tool tip in world frame
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tool_tip_frame_id].copy()
    
    def get_tool_pose(self, q):
        """
        Get tool tip pose as position and rotation matrix.
        
        Args:
            q: Joint positions (8,)
            
        Returns:
            position: (3,) numpy array
            rotation: (3,3) rotation matrix
        """
        pose = self.forward_kinematics(q)
        return pose.translation.copy(), pose.rotation.copy()
    
    def get_jacobian(self, q, frame_id=None, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        """
        Compute spatial Jacobian for specified frame.
        
        Args:
            q: Joint positions (8,)
            frame_id: Frame ID (default: tool_tip)
            reference_frame: Pinocchio reference frame
            
        Returns:
            J: (6, 8) Jacobian matrix [linear; angular]
        """
        if frame_id is None:
            frame_id = self.tool_tip_frame_id
        
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        return pin.computeFrameJacobian(self.model, self.data, q, frame_id, reference_frame)
    
    def get_jacobian_derivative(self, q, dq, frame_id=None, 
                                 reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        """
        Compute Jacobian time derivative times qdot (Jdot * qdot).
        
        Args:
            q: Joint positions (8,)
            dq: Joint velocities (8,)
            frame_id: Frame ID (default: tool_tip)
            reference_frame: Pinocchio reference frame
            
        Returns:
            Jdot_qdot: (6,) vector
        """
        if frame_id is None:
            frame_id = self.tool_tip_frame_id
        
        # Compute frame acceleration with zero joint acceleration
        pin.forwardKinematics(self.model, self.data, q, dq, np.zeros(self.nv))
        pin.updateFramePlacements(self.model, self.data)
        
        # Get classical acceleration (Jdot * qdot)
        acc = pin.getFrameClassicalAcceleration(self.model, self.data, frame_id, 
                                                 reference_frame)
        return np.concatenate([acc.linear, acc.angular])
    
    def get_inertia_matrix(self, q):
        """
        Compute joint-space mass/inertia matrix M(q).
        
        Args:
            q: Joint positions (8,)
            
        Returns:
            M: (8, 8) symmetric positive definite mass matrix
        """

        # To save computation time, Pinocchio calculates only the upper triangle of the inertia matrix
        # This is fine since this matrix is symmetric by design (the effect of a joint A over a joint B
        # is the same as seen from joint B to joint A)
        # The final statement is needed to mirror this triangle and complete the M matrix

        M = pin.crba(self.model, self.data, q)
        return np.triu(M) + np.triu(M, 1).T
    
    def get_bias_forces(self, q, dq):
        """
        Compute bias forces of Coriolis + gravity
            h(q, qdot) = C(q,qdot) * qdot + g(q).
        
        Args:
            q: Joint positions (8,)
            dq: Joint velocities (8,)
            
        Returns:
            h: (8,) bias force vector
        """
        return pin.nonLinearEffects(self.model, self.data, q, dq)
    
    def get_gravity(self, q):
        """
        Compute gravity torques g(q).
        
        Args:
            q: Joint positions (8,)
            
        Returns:
            g: (8,) gravity torque vector
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q)
    
    def get_coriolis(self, q, dq):
        """
        Compute Coriolis matrix C(q, qdot).
        
        Args:
            q: Joint positions (8,)
            dq: Joint velocities (8,)
            
        Returns:
            C: (8, 8) Coriolis matrix
        """
        return pin.computeCoriolisMatrix(self.model, self.data, q, dq)
    
    def inverse_dynamics(self, q, dq, ddq):
        """
        Compute inverse dynamics: tau = M*ddq + h.
        
        Args:
            q: Joint positions (8,)
            dq: Joint velocities (8,)
            ddq: Joint accelerations (8,)
            
        Returns:
            tau: (8,) joint torques
        """
        return pin.rnea(self.model, self.data, q, dq, ddq)
    
    def forward_dynamics(self, q, dq, tau):
        """
        Compute forward dynamics: ddq = M^{-1}(tau - h).
        
        Args:
            q: Joint positions (8,)
            dq: Joint velocities (8,)
            tau: Joint torques (8,)
            
        Returns:
            ddq: (8,) joint accelerations
        """
        return pin.aba(self.model, self.data, q, dq, tau)
    
    def compute_yoshikawa_manipulability(self, q, damping_lambda=1e-10):
        """
        Compute manipulability measure w(q) = sqrt(det(J * J^T)).
        
        Args:
            q: Joint positions (8,)
            damping_lambda: Damping parameter for numerical stability
            
        Returns:
            w: Manipulability scalar
        """
        J = self.get_jacobian(q)
        # We're only working with position control.
        # We can use only the linear part of the Jacobian
        J = J[:3, :]

        det_val = np.linalg.det(J @ J.T)
        det_val = np.clip(det_val, damping_lambda, None)

        # w = sqrt(det(J * J^T))
        return np.sqrt(det_val)
    
    def compute_manipulability_gradient(self, q, epsilon=1e-4):
        """
        Compute gradient of manipulability w.r.t. the joint angles using finite differences.
        
        Args:
            q: Joint positions (8,)
            epsilon: Finite difference step size
            
        Returns:
            grad: (8,) gradient vector dw/dq
        """
        grad = np.zeros(self.nq)
        w0 = self.compute_yoshikawa_manipulability(q)

        # Executes the loop for each joint parameter q0, q1, ...., q7
        for i in range(self.nq):
            q_plus = q.copy()

            # The goal is to add a small value to the i-th joint parameter
            # to calculate how the manipulability changes for the entire robot
            q_plus[i] += epsilon
            
            # Clip to joint limits
            q_plus[i] = np.clip(q_plus[i], self.q_min[i], self.q_max[i])
            
            w_plus = self.compute_yoshikawa_manipulability(q_plus)
            grad[i] = (w_plus - w0) / epsilon
        
        return grad

    @staticmethod
    def damped_pseudoinverse(J, damping=1e-3):
        """
        Compute damped pseudoinverse J# = J^T (J J^T + λ^2 I)^{-1}.
        
        Args:
            J: (m, n) Jacobian matrix
            damping: Damping factor λ
            
        Returns:
            J_pinv: (n, m) damped pseudoinverse
        """
        m = J.shape[0]
        JJT = J @ J.T + damping**2 * np.eye(m)
        return J.T @ np.linalg.inv(JJT)

    def null_space_projector(self, J, damping=1e-3):
        """
        Compute null-space projector N = I - J# J.
        
        Args:
            J: (m, n) Jacobian matrix
            damping: Damping factor for pseudoinverse
            
        Returns:
            N: (n, n) null-space projector
        """
        J_pinv = self.damped_pseudoinverse(J, damping)
        return np.eye(self.nq) - J_pinv @ J
    
    def clip_to_limits(self, q):
        """Clip joint positions to limits."""
        return np.clip(q, self.q_min, self.q_max)
    
    def saturate_torques(self, tau):
        """Saturate torques to effort limits."""
        return np.clip(tau, -self.tau_max, self.tau_max)
    
    def check_joint_limits(self, q, margin=0.05):
        """
        Check if the configuration is within joint limits.
        
        Args:
            q: Joint positions (8,)
            margin: Safety margin in radians
            
        Returns:
            within_limits: Boolean
            violations: List of violated joint indices
        """
        violations = []
        for i in range(self.nq):
            if q[i] < self.q_min[i] + margin or q[i] > self.q_max[i] - margin:
                violations.append(i)
        return len(violations) == 0, violations

    @staticmethod
    def get_home_configuration():
        """
        Get a reasonable home configuration for the robot.
        Tool tip pointing down toward table (z-axis = [0,0,-1]).
        
        With URDF tool_tip rpy="3.14159 0 0" (Rx 180°), the tool z-axis 
        points DOWN (-Z world) when the arm is extended horizontally.
        """
        # All joints at zero = arm horizontal, tool pointing DOWN
        # This is a good starting point for IK
        q_home = np.array([
            0.0,      # platform_roll
            0.0,      # platform_pitch  
            0.0,      # z1_joint1 (shoulder yaw) - forward
            0.0,      # z1_joint2 (shoulder pitch) - horizontal
            0.0,      # z1_joint3 (elbow pitch) - straight
            0.0,      # z1_joint4 (wrist yaw)
            0.0,      # z1_joint5 (wrist pitch)
            0.0       # z1_joint6 (wrist roll)
        ])
        return q_home
    
    def compute_ik_for_pose(self, target_pos, target_rot, q_init=None, max_iter=200, 
                             pos_tol=1e-3, rot_tol=0.05):
        """
        Gauss-Newton method for numerical IK to find configuration reaching target pose (position + orientation).
        Uses damped least squares.
        
        Args:
            target_pos: (3,) target position
            target_rot: (3,3) target rotation matrix
            q_init: Initial guess (default: home)
            max_iter: Maximum iterations
            pos_tol: Position tolerance (m)
            rot_tol: Orientation tolerance (rad)
            
        Returns:
            q: Joint configuration
            success: Whether IK converged
        """

        q = self.get_home_configuration() if q_init is None else q_init.copy()

        alpha = 0.3  # Step size
        damping = 0.05
        
        for i in range(max_iter):
            pos, rot = self.get_tool_pose(q)
            
            # Position error
            e_pos = target_pos - pos
            
            # Orientation error using SO(3) log map
            R_error = target_rot @ rot.T
            e_rot = pin.log3(R_error)  # Convert to axis-angle vector (3,)
            
            pos_err_norm = np.linalg.norm(e_pos)
            rot_err_norm = np.linalg.norm(e_rot)
            
            if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
                return q, True
            
            # Full error vector
            error = np.concatenate([e_pos, e_rot])
            
            # Get full Jacobian
            J = self.get_jacobian(q)  # 6x8
            
            # Damped pseudoinverse
            J_pinv = self.damped_pseudoinverse(J, damping)
            
            # Update
            dq = alpha * J_pinv @ error
            q = q + dq
            q = self.clip_to_limits(q)
        
        # Return the best result even if it doesn't fully converge
        return q, False
    
    def compute_ik_for_position(self, target_pos, q_init=None, max_iter=100, tol=1e-4):
        """
        Simple iterative IK to find configuration reaching target position.
        Uses damped least squares (position only, preserves orientation from init).
        
        Args:
            target_pos: (3,) target position
            q_init: Initial guess (default: home)
            max_iter: Maximum iterations
            tol: Position tolerance
            
        Returns:
            q: Joint configuration
            success: Whether IK converged
        """
        q = self.get_home_configuration() if q_init is None else q_init.copy()
        
        alpha = 0.5  # Step size
        damping = 0.1
        
        for i in range(max_iter):
            pos, _ = self.get_tool_pose(q)
            error = target_pos - pos
            
            if np.linalg.norm(error) < tol:
                return q, True
            
            # Get position Jacobian
            J = self.get_jacobian(q)[:3, :]  # 3x8
            
            # Damped pseudoinverse
            J_pinv = self.damped_pseudoinverse(J, damping)
            
            # Update
            dq = alpha * J_pinv @ error
            q = q + dq
            q = self.clip_to_limits(q)
        
        return q, False
