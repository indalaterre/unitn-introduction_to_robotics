import sys
sys.path.append('/opt/ros/noetic/lib/python3.8/site-packages')

import os
import numpy as np
import pinocchio as pin

from utils import load_urdf_model, calculate_task_jacobian

model, data, ee_link_id = load_urdf_model(model_path='../urdf/giraffe.urdf')


if __name__ == '__main__':

    q_sample = np.array([0.0, 0.5, 1.0, -0.5, 0.0]) # q1, q2, q3, q4, q5

        # Updates the kinematics with latest q changes
    pin.forwardKinematics(model, data, q_sample)
    pin.updateFramePlacements(model, data)

    j_matrix = calculate_task_jacobian(model,
                                       data,
                                       q_sample,
                                       ee_link_id)

    print("\n--- Joint Configuration q ---")
    print(q_sample)
    
    print("\n--- Full Jacobian (6x5) ---")
    print(np.round(j_matrix, 3))
    
    print("\n--- Task Jacobian (4x5) ---")
    print(np.round(j_matrix, 3))
    
    # Verification check: The Angular Y row (Row 4 of Full) should match Row 3 of Task
    print("\nShape of Task Jacobian:", j_matrix.shape)
