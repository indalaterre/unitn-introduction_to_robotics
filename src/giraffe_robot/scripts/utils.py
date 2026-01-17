import os
import sys
import math
import rospy
import numpy as np
import pinocchio as pin 

from functools import reduce
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

def read_position_from_keyboard():
    is_correct = False
    
    while not is_correct:
        string_pos = input('Enter comma separated chair position: ')
        if string_pos == 'q':
            sys.exit()
            
        positions = string_pos.split(',')
        
        is_correct = len(positions) == 2
        if not is_correct:
            print('\tWrong position...Please insert x,y values.')

    x, y = float(positions[0].strip()), float(positions[1].strip())
    return np.array([x, y, 1.0])

def load_urdf_model(model_path):
    urdf_path = os.path.join(os.path.dirname(__file__), '../urdf/giraffe.urdf')

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    ee_link_name = 'end_effector'
    ee_link_id = model.getFrameId(ee_link_name)

    return model, data, ee_link_id

def calculate_task_jacobian(model, data, q, ee_link_id):
    j_full = pin.computeFrameJacobian(model, 
                                      data, 
                                      q,
                                      ee_link_id, 
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # Now we'll filter the full jacobian extracting only the data required by hour robot
    # We need the linear jacobian (first 3 rows)
    # And the rotational jacobian (only row 4 as requested by the lab exercise)
    j_linear, j_pitch = j_full[:3, :], j_full[4, :]
    return np.vstack((j_linear, j_pitch))

def publish_chair_markers():
    print('Building environment...')
    pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1, latch=True)

    marker_array = MarkerArray()

    rows = 5
    chairs_per_row = 8

    row_spacing, chair_spacing, aisle_gap = 1, 1, 2

    # 2m to for the speech stage, 5m to the left
    x_start, y_start = -2, -4.5

    chair_id = 0
    chairs_scale = .025
    chairs_orientation = calculate_orientation_quaternion(0, 0, 90)

    for r in range(rows):
        current_x = x_start + r * row_spacing
        
        for c in range(chairs_per_row):
            current_y = y_start + c * chair_spacing
            if c >= 4:
                current_y += aisle_gap

            marker = Marker()
            marker.id = chair_id
        
            marker.action = Marker.ADD
            marker.header.frame_id = 'world'
            marker.ns = 'conference_room_chairs'
            marker.header.stamp = rospy.Time.now()

            marker.pose.orientation.x = chairs_orientation[0]
            marker.pose.orientation.y = chairs_orientation[1]
            marker.pose.orientation.z = chairs_orientation[2]
            marker.pose.orientation.w = chairs_orientation[3]

            marker.pose.position.x = current_x
            marker.pose.position.y = current_y
            marker.pose.position.z = 0

            marker.scale.x = chairs_scale
            marker.scale.y = chairs_scale
            marker.scale.z = chairs_scale

            marker.color.r = .6862
            marker.color.g = .6353
            marker.color.b = 1
            marker.color.a = 1.0

            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = 'package://giraffe_robot/models/chair2.stl'

            marker_array.markers.append(marker)
            chair_id += 1

    pub.publish(marker_array)

def calculate_orientation_quaternion(roll, pitch, yaw):
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    return quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)