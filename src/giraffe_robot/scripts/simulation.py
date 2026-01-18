import numpy as np
import pinocchio as pin

from utils import load_urdf_model, calculate_task_jacobian, calculate_desired_yaw

def calculate_robot_dynamics(model,
                             data,
                             ee_link_id,
                             pos_d,
                             vel_d,
                             acc_d,
                             pitch_d,
                             dt,
                             q,
                             dq,
                             kp,
                             kd,
                             k_null):
    
    ## Firstly, we need to sync the robot with current math
    pin.computeAllTerms(model, data, q, dq)
    pin.updateFramePlacements(model, data)

    current_position = data.oMf[ee_link_id].translation

    # The sum of q2+q4 gives the end effector pitch
    end_pitch = q[1] + q[3]
    
    # Current yaw is just q[0] (joint1_pan)
    current_yaw = q[0]
    
    # Desired yaw: point toward target position
    yaw_d = calculate_desired_yaw(current_position, pos_d)

    # Use 5x5 task Jacobian (position + pitch + yaw)
    j_task = calculate_task_jacobian(model, data, q, ee_link_id, include_yaw=True)

    position_error = pos_d - current_position
    pitch_error = pitch_d - end_pitch
    yaw_error = yaw_d - current_yaw
    
    # Wrap yaw error to [-pi, pi]
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

    error = np.hstack((position_error, pitch_error, yaw_error))

    v_task = np.hstack((vel_d, 0.0, 0.0))
    a_task = np.hstack((acc_d, 0.0, 0.0))

    d_error = v_task - np.dot(j_task, dq)

    # Computing the control (the acceleration)
    a_cmd = a_task + kp * error + kd * d_error

    # Dynamics (calculating torques)
    inv_m = np.linalg.inv(data.M)
    # inertia = (J _dot_ M^-1 _dot_ J^T)^-1
    # Task is now 5D (3 position + 1 pitch + 1 yaw)
    damping_matrix = 1e-4 * np.eye(5)
    inertia = np.linalg.inv(j_task @ inv_m @ j_task.T + damping_matrix)

    # Calculating the forces
    f_task = inertia @ a_cmd
    tau_task = j_task.T @ f_task

    j_bar = inv_m @ j_task.T @ inertia
    n = np.eye(model.nq) - j_bar @ j_task

    q_second = [.0, .0, .0, 1.57, 0]
    q_error = q - q_second
    tau_null = n @ (-k_null * q_error - 2.0 * np.sqrt(k_null) * dq)

    # Calculates Gravity + Coriolis effect
    nle = pin.nonLinearEffects(model, data, q, dq)
    tau_total = tau_task + tau_null + nle

    # Calculating next q parameters
    ddq = pin.aba(model, data, q, dq, tau_total)

    # Euler integration
    dq += ddq * dt
    q  += dq * dt

    return (position_error, pitch_error, yaw_error), (q, dq), (current_position, end_pitch, current_yaw)
    

def simulate_robot_dynamics(model,
                            data,
                            ee_link_id,
                            initial_q,
                            initial_dq,
                            pos_d,
                            pitch_d,

                            kp,
                            kd,
                            k_null,
                            dt=0.001,
                            num_steps=5):
    
    joint_history = {
        'time': [],
        'joints': [],
        'pos_error': [],
        'pitch_error': [],
        'yaw_error': []
    }

    q = initial_q.copy()
    dq = initial_dq.copy()

    # This will implement the control loop
    for i in range(num_steps):
        error, new_parameters, new_positions = calculate_robot_dynamics(model=model,
                                                                        data=data,
                                                                        ee_link_id=ee_link_id,
                                                                        pos_d=pos_d,
                                                                        vel_d=np.zeros(3),
                                                                        acc_d=np.zeros(3),
                                                                        pitch_d=pitch_d,
                                                                        dt=dt,
                                                                        q=q,
                                                                        dq=dq,
                                                                        kp=kp,
                                                                        kd=kd,
                                                                        k_null=k_null)
        
        q = new_parameters[0]
        dq = new_parameters[1]


        # Storing joints historical data
        joint_history['time'].append(i * dt)
        joint_history['joints'].append(new_parameters[0].copy())
        joint_history['pos_error'].append(np.linalg.norm(error[0]))
        joint_history['pitch_error'].append(error[1])
        joint_history['yaw_error'].append(error[2])

        if i % 500 == 0:
            print(f'Time: {i * dt:.2f}s | PosErr: {np.linalg.norm(error[0]):.4f} | PitchErr: {error[1]:.4f} | YawErr: {error[2]:.4f}')

    return new_positions[0], new_positions[1], joint_history

def run_simulation():
    robot_model, robot_data, ee_link_id = load_urdf_model(model_path='../urdf/giraffe.urdf')

    # Simulations parameters
    dt = .001       # The minimum time step  (1ms)
    total_time = 7  # The simulation will last 5s
    num_steps = int(total_time / dt)

    # Initial robot state
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    dq = np.zeros(robot_model.nq)

    # Gains (Tunable)
    kp = 10.0   # Position Stiffness
    kd = 2.0 * np.sqrt(kp) # Critical Damping
    k_null = 1.0 # Null space stiffness (keep robot comfortable)

    # Desired state for the end-effector
    pos_desired = np.array([1.0, 2.0, 1.0])
    # Pitch: +30° means mic body tilts 30° UP from horizontal
    # Mic head in front of person, body extends back toward ceiling
    pitch_desired = np.deg2rad(30)

    return simulate_robot_dynamics(model=robot_model,
                                   data=robot_data,
                                   ee_link_id=ee_link_id,
                                   pos_d=pos_desired,
                                   pitch_d=pitch_desired,
                                    
                                   initial_q=q,
                                   initial_dq=dq,
                                   kd=kd,
                                   kp=kp,
                                   k_null=k_null,
                                    
                                   num_steps=num_steps)

if __name__ == '__main__':
    final_pos, final_pitch, j_history = run_simulation()

    print('Simulation completed.')
    print(f'Final position: {np.round(final_pos, 3)}')
    print(f'Final pitch: {np.round(final_pitch, 3)}')

