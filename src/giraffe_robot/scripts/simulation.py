import numpy as np
import pinocchio as pin

from utils import load_urdf_model, calculate_task_jacobian

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

    j_task = calculate_task_jacobian(model, data, q, ee_link_id)

    position_error = pos_d - current_position
    rotation_error = pitch_d - end_pitch

    error = np.hstack((position_error, rotation_error))

    v_task = np.hstack((vel_d, 0.0))
    a_task = np.hstack((acc_d, 0.0))

    d_error = v_task - np.dot(j_task, dq)

    # Computing the control (the acceleration)
    a_cmd = a_task + kp * error + kd * d_error

    # Dynamics (calculating torques)
    inv_m = np.linalg.inv(data.M)
    # inertia = (J _dot_ M^-1 _dot_ J^T)^-1
    damping_matrix = 1e-4 * np.eye(4)
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

    return (position_error, rotation_error), (q, dq), (current_position, end_pitch)
    

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
        'pitch_error': []
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
        joint_history['pitch_error'].append(error[1])
        joint_history['pos_error'].append(np.linalg.norm(error[0]))

        if i % 500 == 0:
            print(f'Time: {i * dt:.2f}s | PosError: {np.linalg.norm(error[0]):.4f} | PitchError: {error[1]:.4f}')

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

