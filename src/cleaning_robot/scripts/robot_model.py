"""
Robot Model Module
==================
Loads the 8-DoF cleaning robot URDF into Pinocchio and provides:
- Forward Kinematics (FK)
- Jacobian computation (spatial and body)
- Jdot * qdot computation
- Mass matrix M(q)
- Bias forces h(q, qdot) = C(q,qdot)*qdot + g(q)
- Manipulability computation and gradient
"""

import os
import numpy as np
import pinocchio as pin


class RobotModel:
    """
    Wrapper class for Pinocchio robot model with utility methods.
    """
    
    def __init__(self, urdf_path=None):
        """
        Initialize robot model from URDF.
        
        Args:
            urdf_path: Path to URDF file. If None, uses default assembly.urdf
        """
        if urdf_path is None:
            urdf_path = os.path.join(os.path.dirname(__file__), 
                                     '../urdf/assembly.urdf')
        
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
        
        J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, reference_frame)
        return J.copy()
    
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
    
    def get_mass_matrix(self, q):
        """
        Compute joint-space mass/inertia matrix M(q).
        
        Args:
            q: Joint positions (8,)
            
        Returns:
            M: (8, 8) symmetric positive definite mass matrix
        """
        M = pin.crba(self.model, self.data, q)
        # Make symmetric (CRBA only fills upper triangle)
        M = np.triu(M) + np.triu(M, 1).T
        return M
    
    def get_bias_forces(self, q, dq):
        """
        Compute bias forces h(q, qdot) = C(q,qdot)*qdot + g(q).
        
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
    
    def compute_manipulability(self, q, use_position_only=True):
        """
        Compute manipulability measure w(q) = sqrt(det(J * J^T)).
        
        Args:
            q: Joint positions (8,)
            use_position_only: If True, use only position Jacobian (3x8)
            
        Returns:
            w: Manipulability scalar
        """
        J = self.get_jacobian(q)
        
        if use_position_only:
            Jp = J[:3, :]  # Position Jacobian (3x8)
        else:
            Jp = J  # Full Jacobian (6x8)
        
        # w = sqrt(det(Jp * Jp^T))
        JJT = Jp @ Jp.T
        det_val = np.linalg.det(JJT)
        
        # Add small damping for numerical stability
        if det_val < 1e-10:
            det_val = 1e-10
        
        return np.sqrt(det_val)
    
    def compute_manipulability_gradient(self, q, epsilon=1e-4, use_position_only=True):
        """
        Compute gradient of manipulability w.r.t. joint angles using finite differences.
        
        Args:
            q: Joint positions (8,)
            epsilon: Finite difference step size
            use_position_only: If True, use only position Jacobian
            
        Returns:
            grad: (8,) gradient vector dw/dq
        """
        grad = np.zeros(self.nq)
        w0 = self.compute_manipulability(q, use_position_only)
        
        for i in range(self.nq):
            q_plus = q.copy()
            q_plus[i] += epsilon
            
            # Clip to joint limits
            q_plus[i] = np.clip(q_plus[i], self.q_min[i], self.q_max[i])
            
            w_plus = self.compute_manipulability(q_plus, use_position_only)
            grad[i] = (w_plus - w0) / epsilon
        
        return grad
    
    def damped_pseudoinverse(self, J, damping=1e-3):
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
        Check if configuration is within joint limits.
        
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
    
    def get_home_configuration(self):
        """
        Get a reasonable home configuration for the robot.
        Tool tip pointing down, arm extended forward.
        """
        # Platform neutral, arm in a good starting pose
        q_home = np.array([
            0.0,    # platform_roll
            0.0,    # platform_pitch
            0.0,    # z1_joint1 (shoulder yaw)
            0.5,    # z1_joint2 (shoulder pitch) - slightly forward
            -1.0,   # z1_joint3 (elbow pitch) - bent
            0.0,    # z1_joint4 (wrist yaw)
            0.5,    # z1_joint5 (wrist pitch) - adjust for perpendicular
            0.0     # z1_joint6 (wrist roll)
        ])
        return q_home


def test_robot_model():
    """Test the robot model functionality."""
    print("=" * 60)
    print("Testing Robot Model")
    print("=" * 60)
    
    # Load model
    robot = RobotModel()
    
    # Test with home configuration
    q = robot.get_home_configuration()
    dq = np.zeros(robot.nv)
    
    print(f"\nHome configuration: {q}")
    
    # Forward kinematics
    pos, rot = robot.get_tool_pose(q)
    print(f"\nTool tip position: {pos}")
    print(f"Tool tip rotation:\n{rot}")
    
    # Jacobian
    J = robot.get_jacobian(q)
    print(f"\nJacobian shape: {J.shape}")
    print(f"Jacobian (position rows):\n{np.round(J[:3, :], 3)}")
    
    # Mass matrix
    M = robot.get_mass_matrix(q)
    print(f"\nMass matrix shape: {M.shape}")
    print(f"Mass matrix diagonal: {np.round(np.diag(M), 4)}")
    
    # Manipulability
    w = robot.compute_manipulability(q)
    print(f"\nManipulability: {w:.6f}")
    
    # Manipulability gradient
    grad = robot.compute_manipulability_gradient(q)
    print(f"Manipulability gradient: {np.round(grad, 6)}")
    
    # Bias forces
    h = robot.get_bias_forces(q, dq)
    print(f"\nBias forces (gravity): {np.round(h, 4)}")
    
    # Test inverse dynamics
    ddq = np.zeros(robot.nv)
    tau = robot.inverse_dynamics(q, dq, ddq)
    print(f"Torques for static hold: {np.round(tau, 4)}")
    
    print("\n" + "=" * 60)
    print("Robot Model Test Complete")
    print("=" * 60)


if __name__ == '__main__':
    test_robot_model()
