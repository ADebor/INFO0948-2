import numpy as np
from math import atan2, pi, sqrt

def trajectory_smoother(trajectory):
    # Process a given trajectory to remove useless points (keeps only critical parts of the given trajectory, e.g. sharp turns)
    processed_traj = []

    processed_traj.append(trajectory[0])

    angle_0 = atan2(trajectory[0][0] - trajectory[1][0], trajectory[0][1] - trajectory[1][1])
    angle_1 = angle_0

    i = 1

    while i < len(trajectory) - 1:
        angle_2 = atan2(trajectory[i][0] - trajectory[i+1][0], trajectory[i][1] - trajectory[i+1][1])
        if abs(angle_1 - angle_2) < 0.075 and (np.sign(angle_1) == np.sign(angle_2) or np.sign(angle_1) == 0 or np.sign(angle_2) == 0):
            i += 1
        elif i < len(trajectory) - 2 :
            angle_3 = atan2(trajectory[i][0] - trajectory[i+2][0], trajectory[i][1] - trajectory[i+2][1])
            if abs(angle_1 - angle_3) < 0.005 and ((np.sign(angle_1) == np.sign(angle_3) or np.sign(angle_1) == 0 or np.sign(angle_3) == 0)):
                angle_2 = angle_3
                i += 2
            processed_traj.append(trajectory[i])
        else:
            processed_traj.append(trajectory[i])
            i += 1
        angle_1 = angle_2

    processed_traj.append(trajectory[i])

    # The trajectory has been processed to keep only critical steps; still, some redundancy remains and has to be removed.
    i = 0
    while i < len(processed_traj) -1:
        if euclid(processed_traj[i], processed_traj[i+1]) < 3:
            processed_traj.pop(i+1)
        else:
            i += 1


    return np.asarray(processed_traj)

def euclid(point1, point2):
    # Compute the Euclidean distance between two points
    dist = sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return dist
