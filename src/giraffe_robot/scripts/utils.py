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
    print(f'Moving microphone to: {[x, y, 1.0]}')

    ## Fixing coordinates to match the robot ones
    grid_params = get_chairs_grid_parameters()

    target_x = grid_params['x_start'] + (x * grid_params['row_spacing'])
    target_y = grid_params['y_start'] + (y * grid_params['chair_spacing'])
    if y >= 4:
        target_y += grid_params['aisle_gap']
    
    return (x, y), np.array([target_x + .25, target_y, 1.0])

def load_urdf_model(model_path):
    urdf_path = os.path.join(os.path.dirname(__file__), model_path)

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    ee_link_name = 'end_effector'
    ee_link_id = model.getFrameId(ee_link_name)

    return model, data, ee_link_id

def calculate_task_jacobian(model, data, q, ee_link_id, include_yaw=False):
    j_full = pin.computeFrameJacobian(model, 
                                      data, 
                                      q,
                                      ee_link_id, 
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # We need the linear jacobian (first 3 rows)
    # And the rotational jacobian for pitch (row 4) and optionally yaw (row 5)
    j_linear = j_full[:3, :]
    j_pitch = j_full[4, :]  # Rotation about Y (pitch)
    
    if include_yaw:
        j_yaw = j_full[5, :]  # Rotation about Z (yaw)
        return np.vstack((j_linear, j_pitch, j_yaw))
    else:
        return np.vstack((j_linear, j_pitch))


def calculate_desired_yaw(current_pos, target_pos):
    """
    Calculate the yaw angle needed to point from current position toward target.
    Returns angle in radians.
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    return np.arctan2(dy, dx)

def publish_chair_markers(selected_chair_coords=None):
    pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1, latch=True)

    marker_array = MarkerArray()

    rows = 5
    chairs_per_row = 8
    aisle_size = chairs_per_row / 2

    grid_params = get_chairs_grid_parameters()

    chair_id = 0
    chairs_scale = .025
    chairs_orientation = calculate_orientation_quaternion(0, 0, 90)

    std_color = [.6862, .6353, 1.0, 1.0]
    selected_color = [.8824, .7098, .4235, 1.0]

    for r in range(rows):
        current_x = grid_params['x_start'] + r * grid_params['row_spacing']
        
        for c in range(chairs_per_row):
            current_y = grid_params['y_start'] + c * grid_params['chair_spacing']
            if c >= aisle_size:
                current_y += grid_params['aisle_gap']

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

            if selected_chair_coords and r == selected_chair_coords[0] and c == selected_chair_coords[1]:
                marker.color.r = selected_color[0]
                marker.color.g = selected_color[1]
                marker.color.b = selected_color[2]
                marker.color.a = selected_color[3]
            else:
                marker.color.r = std_color[0]
                marker.color.g = std_color[1]
                marker.color.b = std_color[2]
                marker.color.a = std_color[3]

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

def calculate_current_ee_position(model, data, q):

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    return data.oMf[model.getFrameId('end_effector')].translation


def generate_trajectory_plan(t, t_start, t_end, p_start, p_end):
    if t < t_start:
        return p_start, np.zeros_like(p_start), np.zeros_like(p_start)
    elif t > t_end:
        return p_end, np.zeros_like(p_end), np.zeros_like(p_end)
    else:
        T = t_end - t_start
        tau = (t - t_start) / T

        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
        dds = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T ** 2

        # Interpolation
        pos = p_start + s * (p_end - p_start)
        velocity = ds * (p_end - p_start)
        acceleration = dds * (p_end - p_start)

        return pos, velocity, acceleration

def get_chairs_grid_parameters():
    return {
        # 2m to for the speech stage, 5m to the left
        'x_start': -2,
        'y_start': -4.5,

        'row_spacing': 1,
        'chair_spacing': 1,
        'aisle_gap': 2
    }

