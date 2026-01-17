import sys
# Adjust this path if needed, just like in the previous script
sys.path.append('/opt/ros/noetic/lib/python3.8/site-packages')

import numpy as np

import rospy
from sensor_msgs.msg import JointState

from simulation import calculate_robot_dynamics
from utils import load_urdf_model, read_position_from_keyboard, publish_chair_markers

rospy.init_node('giraffe_controller')
pub = rospy.Publisher('joint_states', JointState, queue_size=10)
rate = rospy.Rate(1000)

robot_model, robot_data, ee_link_id = load_urdf_model(model_path='../urdf/giraffe.urdf')

dt = 0.001
kp = 10.0
kd = 2.0 * np.sqrt(kp)
k_null = 1.0
pitch_desired = 0.5
q = np.array([0.0, 0.5, 1.0, -0.5, 0.0])
dq = np.zeros(robot_model.nq)

new_position = True
publish_chair_markers()
while not rospy.is_shutdown():

    if new_position:
        pos_desired = read_position_from_keyboard()
        print(f'Moving microphone to: {pos_desired}')

        new_position = False

    error, new_parameters, new_positions = calculate_robot_dynamics(model=robot_model,
                                                                    data=robot_data,
                                                                    ee_link_id=ee_link_id,
                                                                    pos_d=pos_desired,
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

    if abs(np.linalg.norm(error[0])) < .0002:
        new_position = True

