import sys
# Adjust this path if needed, just like in the previous script
sys.path.append('/opt/ros/noetic/lib/python3.8/site-packages')

import numpy as np
import pinocchio as pin

from utils import load_urdf_model, calculate_task_jacobian

def calculate_robot_dynamics(model,
                             data,
                             ee_link_id,
                             pos_d,
                             pitch_d,
                             dt,
                             q,
                             dq,
                             kp,
                             kd,
                             k_null):
    
    ## Firstly we need to sync the robot with current math
    pin.computeAllTerms(model, data, q, dq)
    pin.updateFramePlacements(model, data)

    current_position = data.oMf[ee_link_id].translation

    # The end effector pitch is given by the sum of q2+q4
    end_pitch = q[1] + q[3]

    j_task = calculate_task_jacobian(model, data, q, ee_link_id)

    position_error = pos_d - current_position
    rotation_error = pitch_d - end_pitch

    error = np.hstack((position_error, rotation_error))
    # j_task _dot_ dq give use the current velocity. Since we want the microphone to be still
    # we need to consider the 0-velocity as target
    d_error = np.zeros(4) - np.dot(j_task, dq)

    # Computing the control (the acceleration)
    a_cmd = kp * error + kd * d_error

    # Dynamics (calculating torques)
    inv_m = np.linalg.inv(data.M)
    # inertia = (J _dot_ M^-1 _dot_ J^T)^-1
    inertia = np.linalg.inv(j_task @ inv_m @ j_task.T)

    # Calculating the forces
    f_task = inertia @ a_cmd
    tau_task = j_task.T @ f_task

    j_bar = inv_m @ j_task.T @ inertia
    n = np.eye(model.nq) - j_task.T @ j_bar.T

    tau_null = n @ (-k_null * (q - np.zeros(5)) - 2.0 * np.sqrt(k_null) * dq)

    tau_total = tau_task + tau_null + data.g

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
    total_time = 5  # The simulation will last 5s
    num_steps = int(total_time / dt)

    # Initial robot state
    q = np.array([.0, .5, 1.0, -.5, .0])
    dq = np.zeros(robot_model.nq)

    # Gains (Tunable)
    kp = 10.0   # Position Stiffness
    kd = 2.0 * np.sqrt(kp) # Critical Damping
    k_null = 1.0 # Null space stiffness (keep robot comfortable)

    # Desired state for the end-effector
    pos_desired = np.array([1.0, 2.0, 2.0])
    pitch_desired = 0.5 # The desired pitch in radiants

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

