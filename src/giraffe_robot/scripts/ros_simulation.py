import sys
import argparse

# Adjust this path if needed, just like in the previous script
sys.path.append('/opt/ros/noetic/lib/python3.8/site-packages')

import numpy as np

import rospy
from sensor_msgs.msg import JointState

from simulation import calculate_robot_dynamics
from utils import (load_urdf_model,
                   publish_chair_markers,
                   generate_trajectory_plan,
                   read_position_from_keyboard,
                   calculate_current_ee_position)

rospy.init_node('giraffe_controller')
pub = rospy.Publisher('joint_states', JointState, queue_size=10)
rate = rospy.Rate(1000)


def run_simulation(args):
    robot_model, robot_data, ee_link_id = load_urdf_model(model_path=args['model_path'])

    settling_time = args['set_time']

    dt = 0.001

    # Settling time is approx 4 / sqrt(kp) ==> kp = 16 / settling_time ** 2
    kp = 16 / settling_time ** 2
    kd = 2.0 * np.sqrt(kp)
    k_null = 1.0
    pitch_desired = 0.5
    q = np.array([0.0, 0.5, 1.0, -0.5, 0.0])
    dq = np.zeros(robot_model.nq)

    new_position = True
    publish_chair_markers()

    current_ee_position = calculate_current_ee_position(robot_model, robot_data, q)

    while not rospy.is_shutdown():

        current_time = rospy.Time.now()

        if new_position:
            start_position = current_ee_position
            pos_desired = read_position_from_keyboard()
            print(f'Moving microphone to: {pos_desired}')

            start_time = current_time
            end_time = current_time + rospy.Duration.from_sec(settling_time)

            new_position = False

        pos_d, vel_d, acc_d = generate_trajectory_plan(current_time,
                                                       start_time,
                                                       end_time,
                                                       start_position,
                                                       pos_desired)

        error, new_parameters, new_positions = calculate_robot_dynamics(model=robot_model,
                                                                        data=robot_data,
                                                                        ee_link_id=ee_link_id,
                                                                        pos_d=pos_d,
                                                                        pitch_d=pitch_desired,
                                                                        dt=dt,
                                                                        q=q,
                                                                        dq=dq,
                                                                        kp=kp,
                                                                        kd=kd,
                                                                        k_null=k_null)

        msg = JointState()
        msg.header.stamp = rospy.Time.now()

        msg.name = ['joint1_pan', 'joint2_tilt', 'joint3_prismatic', 'joint4_pitch', 'joint5_roll']
        msg.position = new_parameters[0].tolist()

        pub.publish(msg)

        rate.sleep()

        current_ee_position = new_positions[0]

        if abs(np.linalg.norm(error[0])) < .0002:
            new_position = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intent and Slot Filling Task")
    parser.add_argument('--urdf', type=str, default='../urdf/giraffe.urdf', help='URDF file path')
    parser.add_argument('--set_time', type=str, default='7s', help='Set time')
    args = parser.parse_args()

    run_simulation(args)
