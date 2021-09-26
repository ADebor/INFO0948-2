# -*- coding: utf-8 -*-
"""
INFO0948-2 - INTELLIGENT ROBOTICS - Project

Antoine DEBOR & Pierre NAVEZ, ULiège 2021

"""

# VREP
import sim as vrep

# Useful import
import time
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import argparse
from robopy import SE2
from math import ceil, atan2, fmod, sqrt
from roboticstoolbox import DXform
from skimage import measure
import cv2
from scipy import ndimage
import scipy.interpolate as inter
from target_seeker import target_seeker
from trajectory_smoother import trajectory_smoother

from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndi
from scipy.special import binom
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import cv2
from utils import xyz2rgb_pnt, xyz2rgb_cloud, img_seg, rgb2xyz_pnt, getTargetPointNormalPar, getTargetPointNormalCyl, getBases, getDepositPoint
from robopy.base.transforms import rotz
from spatialmath.base.transformsNd import homtrans
import math

from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_init import youbot_init
from youbot_drive import youbot_drive
from youbot_hokuyo_init import youbot_hokuyo_init
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
from beacon import beacon_init, youbot_beacon
from utils_sim import angdiff
from ekf import*

def arguments_parsing():
    """
    Argument parser function
    ---
    parameters :

    None
    ---
    return :

    - args : Keyboard passed arguments
    """

    parser = argparse.ArgumentParser(description="Intelligent Robotics - Project - Arg parser")

    parser.add_argument("--mode", type=str, default="exploration mapping mode",
                        help="Action mode followed by the robot, in {exploration mapping mode}")

    parser.add_argument("--ekf", action ="store_true",
        help="When the robot is in exploration mapping mode, determine if ekf method is used or not")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Initiate the connection to the simulator.
    print('IR 2021 - Project - DEBOR & NAVEZ \nProgram started')
    # Use the following line if you had to recompile remoteApi
    # vrep = remApi('remoteApi', 'extApi.h')
    # vrep = remApi('remoteApi')

    # Close the connection in case if a residual connection exists
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1',  19997, True, True, 2000, 5)

    # The time step the simulator is using (your code should run close to it).
    timestep = .05

    # Synchronous mode
    returnCode = vrep.simxSynchronous(clientID, True)

    if clientID < 0:
        sys.exit('Failed connecting to remote API server. Exiting.')

    print('Connection ' + str(clientID) + ' to remote API server open')

    # This will only work in "continuous remote API server service".
    # See http://www.v-rep.eu/helpFiles/en/remoteApiServerSide.htm
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

    # Send a Trigger to the simulator: this will run a time step for the physics engine
    # because of the synchronous mode. Run several iterations to stabilize the simulation
    for i in range(int(1./timestep)):
        vrep.simxSynchronousTrigger(clientID)
        #vrep.simxGetPingTime(clientID)

    # Retrieve all handles, mostly the Hokuyo.
    h = youbot_init(vrep, clientID)
    h = youbot_hokuyo_init(vrep, h)
    beacons_handle = beacon_init(vrep, clientID)

    # Send a Trigger to the simulator: this will run a time step for the physics engine
    # because of the synchronous mode. Run several iterations to stabilize the simulation
    for i in range(int(1./timestep)):
        vrep.simxSynchronousTrigger(clientID)
        #vrep.simxGetPingTime(clientID)



    ######################
    # INITIAL CONDITIONS #
    ######################

    # Define all the variables which will be used through the whole simulation.
    # Important: Set their initial values.

    # Get the position of the beacons in the world coordinate frame (x, y)
    beacons_world_pos = np.zeros((len(beacons_handle), 3))
    for i, beacon in enumerate(beacons_handle):
        res, beacons_world_pos[i] = vrep.simxGetObjectPosition(clientID, beacon, -1,
                                                               vrep.simx_opmode_oneshot_wait)

    # Parameters for controlling the youBot's wheels: at each iteration,
    # those values will be set for the wheels.
    # They are adapted at each iteration by the code.
    forwBackVel = 0  # Move straight ahead.
    rightVel = 0  # Go sideways.
    rotateRightVel = 0  # Rotate.

    args = arguments_parsing()

    # First state of state machine
    if(args.mode == "exploration mapping mode"):
        ### Initialization of the map
        # 3 states :
        # 0 : explored and free, 1 : unexplored, 2 : explored and not free (obstacle)
        resol = 0.1
        sceneSize = 15
        nbPoints = int(sceneSize / resol)

        map = np.ones((nbPoints, nbPoints), dtype=int)

        ### Initialization of the map plot
        X = np.arange(-sceneSize/2, sceneSize/2, resol)
        Y = np.arange(-sceneSize/2, sceneSize/2, resol)

        ### Vector of coordinates to test for free space
        xx, yy = np.meshgrid(X, Y)
        xx, yy = xx.flatten(), yy.flatten()
        points = np.vstack((xx,yy)).T

        ### Initialization of the known-world boundary vector
        boundary = []

        ### Relevant flags
        firstTargetFlag = True
        newTrajFlag = True
        trajFlag = False # True if trajectory exists
        smoothtrajFlag = False # True if smooth trajectory exists
        goingBackFlag = False # True if the robot has to go back to its initial position

        elapsed_plot = []

        fsm = 'rotate_at_start'

    print('Switching to state: ', fsm)

    # Get the initial position of the robot
    res, youbotPosInit = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
    # Get the initial discrete position of the robot
    x_init, y_init = int(( youbotPosInit[0] + (sceneSize / 2) ) / resol), int(( youbotPosInit[1] + (sceneSize / 2) ) / resol)

    # Set the speed of the wheels to 0.
    h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

    if args.ekf:
        ekf_youbot = ekf(youbotPosInit[0:2], beacons_world_pos)

    # Send a Trigger to the simulator: this will run a time step for the physic engine
    # because of the synchronous mode. Run several iterations to stabilize the simulation
    for i in range(int(1./timestep)):
        vrep.simxSynchronousTrigger(clientID)
        #vrep.simxGetPingTime(clientID)

    #####################
    # EXPLORATION PHASE #
    #####################

    # Start the robot.
    while True:
        try:
            t_first = time.time()
            # Check the connection with the simulator
            if vrep.simxGetConnectionId(clientID) == -1:
                sys.exit('Lost connection to remote API.')


            if not args.ekf:
                # Get the current position and orientation of the robot
                res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
                vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error
            else:
                youbotPos = [ekf_youbot.x[0][0],ekf_youbot.x[1][0]]

            res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
            vrchk(vrep, res, True)

            # Get data from the hokuyo - return empty if data is not captured
            full_scanned_points, full_contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)
            vrchk(vrep, res)

            # Downsampling of the data to speed up the computations
            sampling_factor = 10

            scanned_points = full_scanned_points[0:6, ::sampling_factor] # slicing
            contacts = full_contacts[0:2, ::sampling_factor] # slicing

            # -- Transform data from sensor coordinate to absolute coordinate --
            T = np.reshape(SE2(youbotEuler[2], 'rad', youbotPos[0], youbotPos[1]).data, (3,3))

            Tpoints_1 = np.ones((int(scanned_points.shape[0] / 2), scanned_points.shape[1]))
            Tpoints_2 = np.ones((int(scanned_points.shape[0] / 2), scanned_points.shape[1]))

            Tpoints_1[0:2, :] = scanned_points[0:2, :]
            Tpoints_1 = np.round(np.dot(T, Tpoints_1)[:2, :], 2) # scanned data from one side, rounded to fixed resolution

            Tpoints_2[0:2, :] = scanned_points[3:5, :]
            Tpoints_2 = np.round(np.dot(T, Tpoints_2)[:2, :], 2) # scanned data from the other side, rounded to fixed resolution

            tmp = list(zip(Tpoints_1[0], Tpoints_1[1], contacts[0]))
            Tpoints_1 = sorted(set(tmp), key=tmp.index) # unique coordinates extracted from scanned data from one side, zipped as coordinates tuples

            tmp = list(zip(Tpoints_2[0], Tpoints_2[1], contacts[1]))
            Tpoints_2 = sorted(set(tmp), key=tmp.index) # unique coordinates from scanned data from the other side, zipped as coordinates tuples

            Tpoints_1.extend(Tpoints_2)
            Tpoints = Tpoints_1 # list of all scanned points, in abdsolute coordinate

            # -- Update the map --
            # Free space
            Tpoints.append((youbotPos[0], youbotPos[1], False)) # add position of the robot

            vert = np.empty((len(Tpoints), 2))
            vert = [[point[0], point[1]] for point in Tpoints]
            p = Path(vert)  # construct boundary of newly observed space

            grid = p.contains_points(points)
            mask = (grid.reshape(np.shape(map)))
            mask = np.asarray(mask) # construct mask of free space

            obstacle_mask = (map == 2)
            cmp_mask = np.logical_and(mask, obstacle_mask) # construct mask of overlapping between free space and already obstacle space

            mask = ~ mask
            map = map * mask
            map = map + 2 * cmp_mask # update map with new free space while coping with already obstacle space

            # Obstacles
            for x, y, contact in Tpoints:
                if contact == True:
                    y = int(( y + (sceneSize / 2) ) / resol)
                    if abs(y) == sceneSize/resol:
                        y = np.sign(y) * (sceneSize/resol - 1)
                    x = int(( x + (sceneSize / 2) ) / resol)
                    if abs(x) == sceneSize/resol:
                        x = np.sign(x) * (sceneSize/resol - 1)
                    map[y, x] = 2


            # Apply the state machine.
            if fsm == 'rotate_at_start':
                # Rotate until the robot has an angle of -pi/2 (measured with respect to the world's reference frame).
                # Again, use a proportional controller. In case of overshoot, the angle difference will change sign,
                # and the robot will correctly find its way back (e.g.: the angular speed is positive, the robot overshoots,
                # the anguler speed becomes negative).
                # youbotEuler(3) is the rotation around the vertical axis.
                rotateRightVel = angdiff(youbotEuler[2], (-np.pi))

                # Switch to the computation of the target
                # when the robot is at an angle close to -pi.
                if abs(angdiff(youbotEuler[2], (-np.pi))) < .002:
                    rotateRightVel = 0
                    fsm = 'compute_new_target'
                    print('Switching to state: ', fsm)




            elif fsm == 'compute_new_target':
                # Compute a new target for the robot to reach, considering the
                # known-world boundary on a binarized inflated map

                print("\nSeeking for new target...")
                # -- Update the boundary --
                BW_map = (map == 1).astype(np.uint8) # Binarized map s.t. 0 = explored, 1 = unexplored
                boundary = measure.find_contours(BW_map, 0)
                boundary = np.vstack(boundary)

                # -- Inflate the map --
                BW_map = (map >= 2).astype(int) # Binarized map s.t. 0 = free or unexplored, 1 = obstacle
                BW_map = ndimage.binary_dilation(BW_map, iterations=5).astype(BW_map.dtype)

                robot_inflate_map = (map < 0) # 0 everywhere
                x_start, y_start = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)
                robot_inflate_map[y_start, x_start] = 1
                robot_inflate_map = ndimage.binary_dilation(robot_inflate_map, iterations=2).astype(robot_inflate_map.dtype)
                robot_inflate_map = ~ robot_inflate_map

                BW_map = np.logical_and(BW_map, robot_inflate_map).astype(int)

                if(newTrajFlag == True):
                    # -- Seek for new target --
                    not_found = True
                    while not_found == True:
                        if firstTargetFlag == True:
                            target = random.choice(boundary) # random choice along the known-world boundary
                        else:
                            # After first target, seek for the nearest
                            target = target_seeker(boundary, (x_start, y_start), "combined", BW_map, (x_init, y_init))

                        print("new target candidate : {}".format(target))
                        if BW_map[int(target[0]), int(target[1])] == 1:
                            print("candidate rejected : obstacle")
                        else:
                            not_found = False
                            if firstTargetFlag == True:
                                firstTargetFlag = False
                            print("new target locked : {}".format(target))

                if (target ==  np.asarray([y_init, x_init]).astype(int)).all() :
                    # Exploration phase finished, robot has to go back to its initial position
                    print("Going back to initial position")
                    goingBackFlag = True

                fsm = 'compute_new_traj'
                print('Switching to state: ', fsm)

            elif fsm == 'compute_new_traj':
                # Compute a trajectory towards the previously determined target
                # using DXform

                print("\nComputing new trajectory...")

                # Distance transform path planing

                dx = DXform(BW_map)
                dx.plan(goal = np.flip(target.astype(int)))

                traj = dx.query((int(x_start), int(y_start)), animate = False)

                print("\nTrajectory found !")
                trajFlag = True

                for i, element in enumerate(traj):
                    traj[i] = np.flip(element)

                fsm = 'traj_smoothing'
                print('Switching to state: ', fsm)

            elif fsm == 'traj_smoothing':
                # Smoothing and downsampling of the previously computed trajectory

                print("\nSmoothing the trajectory...")
                # Regular downsampling by a factor 2
                traj_pro = traj[::2]
                # Trajectory-specific downsampling
                traj_pro = trajectory_smoother(traj_pro)

                # Find the B-spline representation of the trajectory.

                tmp1 = np.zeros(len(traj_pro))
                tmp2 = np.zeros(len(traj_pro))
                for i, element in enumerate(traj_pro):
                    tmp1[i] = element[0]
                    tmp2[i] = element[1]
                smooth_traj = [tmp1, tmp2]

                smoothtrajFlag = True

                fsm = 'reorientation'
                #This variable allows to better deal with the controller when the kalman filter is used
                locked = False

                print('Switching to state: ', fsm)

            elif fsm == 'reorientation':
                # Reorientating the robot before following the computed trajectory
                if args.ekf:
                    if not locked:
                        x_d, y_d = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)
                        locked = True
                else:
                    x_d, y_d = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)

                target_angle = atan2(smooth_traj[0][1] - y_d, smooth_traj[1][1] - x_d)

                # Change of reference
                if target_angle < 0 :
                    target_angle = 2 * np.pi + target_angle

                head_orientation = youbotEuler[2] - np.pi / 2

                if head_orientation < - np.pi:
                    head_orientation = head_orientation % np.pi

                if head_orientation < 0 :
                    head_orientation = 2 * np.pi + head_orientation

                # Proportional controller
                rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)


                if rotateRightVel < -np.pi or rotateRightVel > np.pi:
                    rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

                # Switch to the trajectory following task when the robot is
                # well oriented
                if abs(rotateRightVel) < .002:
                    rotateRightVel = 0

                    traj_idx = 1    # First trajectory step not considered
                    new_state = True
                    next_state_y = smooth_traj[0][traj_idx]
                    next_state_x = smooth_traj[1][traj_idx]

                    fsm = 'traj_follow'
                    print('Switching to state: ', fsm)

            elif fsm == 'traj_follow':
                # Trajectory following procedure

                # Check if the current step has not been identified as an obstacle
                BW_map = (map >= 2).astype(int) # Binarized map s.t. 0 = free or unexplored, 1 = obstacle
                BW_map = ndimage.binary_dilation(BW_map, iterations=4).astype(BW_map.dtype) # iterations reduced to 4 to avoid the robot to be stuck

                robot_inflate_map = (map < 0) # 0 everywhere
                y_start, x_start = int(( youbotPos[1] + (sceneSize / 2) ) / resol), int(( youbotPos[0] + (sceneSize / 2) ) / resol)
                robot_inflate_map[y_start, x_start] = 1
                robot_inflate_map = ndimage.binary_dilation(robot_inflate_map, iterations=1).astype(robot_inflate_map.dtype)
                robot_inflate_map = ~ robot_inflate_map

                BW_map = np.logical_and(BW_map, robot_inflate_map).astype(int)

                if BW_map[int(next_state_y), int(next_state_x)] == 1:
                    print("Trajectory step turns out to be an obstacle !")
                    forwBackVel = 0.0
                    rotateRightVel = 0.0

                    if traj_idx == len(smooth_traj[0]): # If final target can not be reached, then change target
                        newTrajFlag = True
                    fsm = 'compute_new_target'

                else:
                    if new_state == True:
                        next_state_y = smooth_traj[0][traj_idx]
                        next_state_x = smooth_traj[1][traj_idx]
                        traj_idx += 1

                        x_state, y_state = next_state_x * resol - (sceneSize / 2), next_state_y * resol - (sceneSize / 2)

                        new_state = False
                        rotation_allowed = True
                        rotating = False


                    # -- Forward speed control --
                    x_curr = youbotPos[0]
                    y_curr = youbotPos[1]

                    dist = sqrt((x_curr-x_state)**2 + (y_curr-y_state)**2)
                    forwBackVel = - 3 * dist

                    if traj_idx == len(smooth_traj[0]) - 1: # avoid crashes
                        forwBackVel = - dist


                    # -- Orientation control --
                    y_d, x_d = int(( youbotPos[1] + (sceneSize / 2) ) / resol), int(( youbotPos[0] + (sceneSize / 2) ) / resol)
                    target_angle = atan2(next_state_y - y_d, next_state_x - x_d)


                    if target_angle < 0 :
                        target_angle = 2 * np.pi + target_angle

                    head_orientation = youbotEuler[2] - np.pi / 2

                    if head_orientation < - np.pi:
                        head_orientation = head_orientation % np.pi

                    if head_orientation < 0 :
                        head_orientation = 2 * np.pi + head_orientation

                    # Proportional controller
                    rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

                    if rotateRightVel < - np.pi or rotateRightVel > np.pi:
                        rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

                    if rotating == True :
                        rotateRightVel = rotateRightVel
                        forwBackVel = 0.0
                        if abs(rotateRightVel) < 0.01:
                            rotating = False
                            rotateRightVel = 0.0

                    if abs(rotateRightVel) > 0.1 and rotating == False:
                        # If angle too large, stop the robot and reorienting
                        rotateRightVel = rotateRightVel
                        forwBackVel = 0.0
                        rotating = True
                    elif rotating == False:
                        rotateRightVel = 0.0

                    # -- Trajectory handling --
                    if  dist < .2:
                        new_state = True
                        if traj_idx == len(smooth_traj[0]):
                            # Target reached
                            print("\nTarget reached !")
                            new_state = False
                            forwBackVel = 0.0
                            rotateRightVel = 0.0
                            if goingBackFlag == True:
                                fsm = "exploration end procedure"
                            else:
                                newTrajFlag = True # Need for a new target
                                fsm = "compute_new_target"
                            print("Switching to state : ", fsm)

            elif fsm == 'exploration end procedure':
                print('Exploration phase finished')
                X = np.arange(-sceneSize/2, sceneSize/2 + resol, resol)
                Y = np.arange(-sceneSize/2, sceneSize/2 + resol, resol)
                #fig = plt.pcolormesh(X, Y, map)
                #fig = plt.savefig('explored_map')
                time.sleep(3)
                break

            else:
                sys.exit('Unknown state ' + fsm)

            # -- Update wheel velocities --
            h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)
            res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
            vrchk(vrep, res, True)

            # -- Update the youBot estimate
            if args.ekf:
                velocities = ekf_youbot.compute_velocity_components(h)
                ekf_youbot.predict(velocities, youbotEuler[2])
                beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)
                ekf_youbot.update(beacon_dist)

            # -- Update the plot --

            y_d, x_d = int(( youbotPos[1] + (sceneSize / 2) ) / resol), int(( youbotPos[0] + (sceneSize / 2) ) / resol)


            img = np.zeros((map.shape[0], map.shape[1], 3), dtype = np.uint8)

            DISCOVERED_CLR = (0, 100, 100)
            WALL_CLR = (0, 255, 255)
            ROBOT_CLR = (0, 150, 255)
            TRAJ_CLR = (100, 0, 150)
            SMOOTH_TRAJ_CLR = (100, 0, 255)

            discovered_idx = np.where(map == 0)
            not_discovered_idx = np.where(map == 1)
            walls_idx = np.where(map == 2)
            robot_idx = (np.array([y_d-1, y_d-1, y_d-1, y_d, y_d, y_d, y_d+1, y_d+1, y_d+1]), np.array([x_d-1, x_d, x_d+1, x_d-1, x_d, x_d+1, x_d-1, x_d, x_d+1]))

            img[discovered_idx] = DISCOVERED_CLR
            img[walls_idx] = WALL_CLR

            if trajFlag == True:
                trajectory_idx = (traj[:, 0], traj[:, 1])
                img[trajectory_idx] = TRAJ_CLR

            if smoothtrajFlag == True:
                smooth_trajectory_idx = (smooth_traj[0].astype(int), smooth_traj[1].astype(int))
                img[smooth_trajectory_idx] = SMOOTH_TRAJ_CLR

            img[robot_idx] = ROBOT_CLR

            img = cv2.resize(img, (500, 500))
            cv2.imshow("House map", img)
            cv2.waitKey(1)

            elapsed = time.time() - t_first
            elapsed_plot.append(elapsed)

            # Send a Trigger to the simulator: this will run a time step for the physic engine
            # because of the synchronous mode.
            vrep.simxSynchronousTrigger(clientID)
            #vrep.simxGetPingTime(clientID)
            #print(elapsed)

        except KeyboardInterrupt:
            cleanup_vrep(vrep, clientID)
            sys.exit('Stop simulation')

    # Send a Trigger to the simulator: this will run a time step for the physic engine
    # because of the synchronous mode. Run several iterations to stabilize the simulation
    for i in range(int(1./timestep)):
        vrep.simxSynchronousTrigger(clientID)

    if args.ekf:
        # Grasping phase not implemented for Kalman
        cleanup_vrep(vrep, clientID)
        print('\nSimulation has stopped')
        exit(0)

    #######################
    # MAP POST-PROCESSING #
    #######################

    # Not Implemented.

    # Coordinates of easy table in the training map (hardcoded)
    x_table = -3.0
    y_table = -6.0
    x_deposit = -4.5
    y_deposit = 2.0

    ##################
    # GRASPING PHASE #
    ##################

    # Turn Off the Hokuyo sensor
    res = vrep.simxClearIntegerSignal(clientID, 'handle_xy_sensor', vrep.simx_opmode_oneshot)
    vrchk(vrep, res, True)

    #Rotate rgb/xyz sensor
    vrep.simxSetObjectOrientation(clientID, h['rgbdCasing'], h['ref'], (0.0, 0.0, np.pi/4), vrep.simx_opmode_oneshot)

    res = vrep.simxSetFloatSignal(clientID, 'rgbd_sensor_scan_angle', np.pi/8, vrep.simx_opmode_oneshot_wait)
    vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error.

    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

    res = vrep.simxSetIntegerSignal(clientID, 'handle_xyz_sensor', 1, vrep.simx_opmode_oneshot_wait)
    vrchk(vrep, res, True)

    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

    forwBackVel = 0
    rightVel = 0
    rotateRightVel = 0

    h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

    for i in range(int(1./timestep)):
        vrep.simxSynchronousTrigger(clientID)

    deposit_iter = 0
    bases, traj_bases = getBases(n=24, x_table=x_table, y_table=y_table) # Hardcoded bases around easy table
    print('\nThe hardcoded bases are: \n', bases)
    not_visited = bases.copy()
    traj_not_visited = traj_bases.copy()

    grasp_flag = False
    grasp_flag_docking = False
    drop_flag = False
    cnter_drop = 20
    cnter_ptcloud = 20

    fsm = 'ToEasyTableTarget'
    print('\nSwitching to state ', fsm)
    new_state = True

    while(True):
        res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True)

        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True)

        if fsm == 'NextBase':
            closest = 4*[np.inf]

            if new_state == True:
                not_visited.pop(0)
                traj_not_visited.pop(0)
                new_state = False

            if len(not_visited) == 0:
                fsm = 'GraspingEndProcedure'
                print('\nAll bases have been visited, switching to state ', fsm)
            else:
                cnter_ptcloud -= 1
                print(cnter_ptcloud)
                if cnter_ptcloud == 0:
                    cnter_ptcloud = 20

                    pts = youbot_xyz_sensor(vrep, h, vrep.simx_opmode_oneshot_wait)
                    #pcd = o3d.geometry.PointCloud()
                    #pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
                    #o3d.visualization.draw_geometries([pcd])

                    res = vrep.simxSetIntegerSignal(clientID, 'handle_xyz_sensor', 1, vrep.simx_opmode_oneshot_wait)
                    vrchk(vrep, res)

                    vrep.simxSynchronousTrigger(clientID)

                    # Pre-process point cloud
                    # - Height selection (do not take the table into account)
                    idx = np.where(pts[:,1] >= -0.04)
                    pts = pts[idx]

                    # - Depth selection (only take objects close enough)
                    idx = np.where(pts[:,3] <= 2)
                    pts = pts[idx]

                    # - Find closest point (in order to discard objects different from
                    # the one of interest, and to check if if the object is close enough)
                    if len(pts) == 0:
                        closest = 4*[np.inf]
                    else:
                        closest = pts[np.argmin(pts[:,3])]
                    print("\nClosest point in 3D: ", closest)

                if closest[3] >= 0.6:
                    next_base = traj_not_visited[0]

                    # -- Forward speed control --
                    x_curr = youbotPos[0]
                    y_curr = youbotPos[1]

                    dist = sqrt((x_curr-next_base[0])**2 + (y_curr-next_base[1])**2)
                    forwBackVel = - 0.2 * dist

                    # -- Orientation control --

                    target_angle = atan2(next_base[1] - y_curr, next_base[0] - x_curr)

                    if target_angle < 0 :
                        target_angle = 2 * np.pi + target_angle

                    head_orientation = youbotEuler[2] - np.pi / 2

                    if head_orientation < - np.pi:
                        head_orientation = head_orientation % np.pi

                    if head_orientation < 0 :
                        head_orientation = 2 * np.pi + head_orientation

                    # Proportional controller
                    rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

                    if rotateRightVel < - np.pi or rotateRightVel > np.pi:
                        rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

                    # -- Trajectory handling --
                    if  dist < .2:
                        new_state = True

                else:
                    # The point cloud revealed a point close enough to the robot
                    # to be analyzed in more details.
                    fsm = 'PointCloud'
                    print('The point cloud revealed a point close enough to the robot to be analyzed in more details, switching to state ', fsm)
                    forwBackVel = 0.0
                    rotateRightVel = 0.0
                    rightvel = 0.0
                    cnter_ptcloud = 20



        elif fsm == 'ToEasyTableTarget':
            target_base = not_visited[0]
            x_start, y_start = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)
            x_target, y_target = int(( target_base[0] + (sceneSize / 2) ) / resol), int(( target_base[1] + (sceneSize / 2) ) / resol)
            target = np.array([x_target, y_target])

            # -- Inflate the map --
            BW_map = (map >= 2).astype(int) # Binarized map s.t. 0 = free, 1 = obstacle
            BW_map = ndimage.binary_dilation(BW_map, iterations=3).astype(BW_map.dtype)

            robot_inflate_map = (map < 0) # 0 everywhere
            robot_inflate_map[y_start, x_start] = 1
            robot_inflate_map = ndimage.binary_dilation(robot_inflate_map, iterations=2).astype(robot_inflate_map.dtype)
            robot_inflate_map = ~ robot_inflate_map

            BW_map = np.logical_and(BW_map, robot_inflate_map).astype(int)

            fsm = 'ToEasyTableTraj'
            print('\nSwitching to state ', fsm)

        elif fsm == 'ToEasyTableTraj':
            print('\tComputing new trajectory...')

            # Distance transform path planing

            dx = DXform(BW_map)
            dx.plan(goal = target.astype(int))

            traj = dx.query((int(x_start), int(y_start)), animate = False)

            print("\tTrajectory found !")
            trajFlag = True

            for i, element in enumerate(traj):
                traj[i] = np.flip(element)

            fsm = 'ToEasyTableTrajSmoothing'
            print('\nSwitching to state: ', fsm)

        elif fsm == 'ToEasyTableTrajSmoothing':
            # Smoothing and downsampling of the previously computed trajectory
            print("\tSmoothing the trajectory...")
            # Regular downsampling by a factor 2
            traj_pro = traj[::2]
            # Trajectory-specific downsampling
            traj_pro = trajectory_smoother(traj_pro)

            # Find the B-spline representation of the trajectory.

            tmp1 = np.zeros(len(traj_pro))
            tmp2 = np.zeros(len(traj_pro))
            for i, element in enumerate(traj_pro):
                tmp1[i] = element[0]
                tmp2[i] = element[1]
            smooth_traj = [tmp1, tmp2]

            smoothtrajFlag = True

            fsm = 'ToEasyTableReorientation'
            print('\nSwitching to state: ', fsm)

        elif fsm == 'ToEasyTableReorientation':
            # Reorientating the robot before following the computed trajectory
            x_d, y_d = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)
            a = smooth_traj[0][0] - y_d
            b = smooth_traj[1][0] - x_d
            target_angle = atan2(smooth_traj[0][0] - y_d, smooth_traj[1][0] - x_d)

            # Change of reference
            if target_angle < 0 :
                target_angle = 2 * np.pi + target_angle

            head_orientation = youbotEuler[2] - np.pi / 2

            if head_orientation < - np.pi:
                head_orientation = head_orientation % np.pi

            if head_orientation < 0 :
                head_orientation = 2 * np.pi + head_orientation

            # Proportional controller
            rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

            if rotateRightVel < -np.pi or rotateRightVel > np.pi:
                rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

            # Switch to the trajectory following task when the robot is
            # well oriented
            if abs(rotateRightVel) < .002:
                rotateRightVel = 0

                traj_idx = 0
                new_state = True
                next_state_y = smooth_traj[0][traj_idx]
                next_state_x = smooth_traj[1][traj_idx]

                fsm = 'ToEasyTableTrajFollow'
                print('Switching to state: ', fsm)

        elif fsm == 'ToEasyTableTrajFollow':
            # Trajectory following procedure

            if new_state == True:
                next_state_y = smooth_traj[0][traj_idx]
                next_state_x = smooth_traj[1][traj_idx]
                traj_idx += 1

                x_state, y_state = next_state_x * resol - (sceneSize / 2), next_state_y * resol - (sceneSize / 2)

                new_state = False
                rotation_allowed = True
                rotating = False

            # -- Forward speed control --
            x_curr = youbotPos[0]
            y_curr = youbotPos[1]

            dist = sqrt((x_curr-x_state)**2 + (y_curr-y_state)**2)
            forwBackVel = - dist

            if traj_idx == len(smooth_traj[0]) - 1: # avoid crashes
                forwBackVel = - dist


            # -- Orientation control --
            y_d, x_d = int(( youbotPos[1] + (sceneSize / 2) ) / resol), int(( youbotPos[0] + (sceneSize / 2) ) / resol)

            target_angle = atan2(next_state_y - y_d, next_state_x - x_d)

            if target_angle < 0 :
                target_angle = 2 * np.pi + target_angle

            head_orientation = youbotEuler[2] - np.pi / 2

            if head_orientation < - np.pi:
                head_orientation = head_orientation % np.pi

            if head_orientation < 0 :
                head_orientation = 2 * np.pi + head_orientation

            # Proportional controller
            rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

            if rotateRightVel < - np.pi or rotateRightVel > np.pi:
                rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

            if rotating == True :
                rotateRightVel = rotateRightVel
                forwBackVel = 0.0
                if abs(rotateRightVel) < 0.01:
                    rotating = False
                    rotateRightVel = 0.0

            if abs(rotateRightVel) > 0.1 and rotating == False:
                # If angle too large, stop the robot and reorienting
                rotateRightVel = rotateRightVel
                forwBackVel = 0.0
                rotating = True

            elif rotating == False:
                rotateRightVel = 0.0

            # -- Trajectory handling --
            if  dist < .2:
                new_state = True
                if traj_idx == len(smooth_traj[0]):
                    # Target reached
                    print("\tTarget reached !")
                    new_state = False
                    forwBackVel = 0.0
                    rotateRightVel = 0.0

                    fsm = 'ToEasyTableSelfRotatePhase1'
                    print('\nSwitching to state ', fsm)

        elif fsm == 'ToEasyTableSelfRotatePhase1':
            # Make the robot orient itself towards the centre of the table
            target_angle = atan2(y_table - youbotPos[1], x_table - youbotPos[0])

            if target_angle < 0 :
                target_angle = 2 * np.pi + target_angle

            head_orientation = youbotEuler[2] - np.pi / 2

            if head_orientation < - np.pi:
                head_orientation = head_orientation % np.pi

            if head_orientation < 0 :
                head_orientation = 2 * np.pi + head_orientation

            rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

            if rotateRightVel < -np.pi or rotateRightVel > np.pi:
                rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

            if abs(rotateRightVel) < .002:
                rotateRightVel = 0
                fsm = 'ToEasyTableSelfRotatePhase2'
                angle = youbotEuler[2] - np.pi/2
                print('\nSwitching to state ', fsm)

        elif fsm == 'ToEasyTableSelfRotatePhase2':
            # Make the robot rotate of an angle of -pi/2
            rotateRightVel = angdiff(youbotEuler[2], angle)

            # Switch to the computation of the target
            # when the robot is at an angle close to -pi.
            if abs(angdiff(youbotEuler[2], angle)) < .002:
                rotateRightVel = 0
                fsm = 'ToEasyTablePerpApproach'
                print('\nSwitching to state: ', fsm)

        elif fsm == 'ToEasyTablePerpApproach':
            # Make the robot approach the table in a perpendicular fashion
            dist = math.sqrt((x_table-youbotPos[0])**2 + (y_table-youbotPos[1])**2)
            rightVel = 0.5*dist
            if dist <= 0.75:
                rightVel = 0
                fsm = 'NextBase'
                print('\nSwitching to state ', fsm)

        elif fsm == 'ToDepositSelfRotatePhase1':
            # Make the robot orient itself towards the centre of the table
            target_angle = atan2(y_deposit - youbotPos[1], x_deposit - youbotPos[0])

            if target_angle < 0 :
                target_angle = 2 * np.pi + target_angle

            head_orientation = youbotEuler[2] - np.pi / 2

            if head_orientation < - np.pi:
                head_orientation = head_orientation % np.pi

            if head_orientation < 0 :
                head_orientation = 2 * np.pi + head_orientation

            rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

            if rotateRightVel < -np.pi or rotateRightVel > np.pi:
                rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

            if abs(rotateRightVel) < .002:
                rotateRightVel = 0
                fsm = 'ToDepositSelfRotatePhase2'
                angle = youbotEuler[2] - np.pi/2
                print('\nSwitching to state ', fsm)

        elif fsm == 'ToDepositSelfRotatePhase2':
            # Make the robot rotate of an angle of -pi/2
            rotateRightVel = angdiff(youbotEuler[2], angle)

            # Switch to the computation of the target
            # when the robot is at an angle close to -pi.
            if abs(angdiff(youbotEuler[2], angle)) < .002:
                rotateRightVel = 0
                fsm = 'ToDepositPerpApproach'
                print('\nSwitching to state: ', fsm)

        elif fsm == 'ToDepositPerpApproach':
            # Make the robot approach the table in a perpendicular fashion
            dist = math.sqrt((x_deposit-youbotPos[0])**2 + (y_deposit-youbotPos[1])**2)
            rightVel = 0.5*dist
            if dist <= 0.75:
                rightVel = 0
                fsm = 'GraspingDepositDrop_phase1'
                print('\nSwitching to state ', fsm)

        elif fsm == 'PointCloud':

            pts = youbot_xyz_sensor(vrep, h, vrep.simx_opmode_oneshot_wait)

            # Pre-process point cloud
            # - Height selection (do not take the table into account)
            idx = np.where(pts[:,1] >= -0.04)
            pts = pts[idx]

            # - Depth selection (only take objects close enough)
            idx = np.where(pts[:,3] <= 2)
            pts = pts[idx]

            # - Find closest point (in order to discard objects different from
            # the one of interest, and to check if if the object is close enough)
            if len(pts) == 0:
                fsm = 'NextBase'
                print('\tNo remaining point after cloud post-processing, switching to state ', fsm)

            else:
                closest = pts[np.argmin(pts[:,3])]
                print("\nClosest point in 3D: ", closest)
                if closest[3] >= 0.6:
                    fsm = 'NextBase'
                    print('\tThe closest detected point is too far from the robot, switching to state ', fsm)

                else:
                    # - Delete points too far from the object of interest
                    closest_depth = closest[3]
                    indices = []
                    for i, element in enumerate(pts):
                        if element[3] > closest_depth + 0.1:
                            indices.append(i)
                    pts = np.delete(pts, indices, 0)

                    # Assumption: If less than 20 points of interest are detected, the robot
                    # will not be able to determine an accurate estimation of the target point
                    # to grasp. It should therefore continue its route towards the next base.
                    if len(pts) < 20:
                        fsm = 'NextBase'
                        print("\tPoint cloud not dense enough, switching to state ", fsm)

                    else:
                        # Estimate normals of each point in the point cloud of interest.
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
                        pcd.normalize_normals()
                        pcd.orient_normals_consistent_tangent_plane(k=5)
                        #o3d.visualization.draw_geometries([pcd])

                        # Remove (some of) the noise inherent to the normals estimation
                        normals = np.asarray(pcd.normals)
                        normals = np.around(normals, decimals=3)

                        fsm = 'ShapeDetection'
                        print('\nSwitching to state ', fsm)

        elif fsm == 'ShapeDetection':
            # Compute the cross product between all normals of interest, iot
            # detect if the object is a cylinder or a parallelipiped.
            n_par = 0
            for i, normal in enumerate(normals):
                for j, other in enumerate(np.concatenate((normals[:i, :], normals[i+1:, :]), 0)):
                    cross = np.cross(normal, other)
                    if j>=i:
                        if np.linalg.norm(cross) < 0.25:
                            n_par += 1

            # Compute the fraction of parallel normals.
            frac = n_par/binom(np.shape(normals)[0], 2)*100

            # Assumption: Above a percentage of 30% of parallel normals, the object is considered as a parallelipiped.
            # Below this percentage, the object is assumed to be a cylinder.
            if frac > 30:
                object = 'Parallelipiped'
            else:
                object = 'Cylinder'

            print(f"\n{object} detected !" + f" with {frac} of parallel normals. ")

            fsm = 'GraspingComputeTarget'
            print('\nSwitching to state ', fsm)

        elif fsm == 'GraspingComputeTarget':

            if object == 'Cylinder':
                tp_candidate = getTargetPointNormalCyl(pts)

            elif object == 'Parallelipiped':
                tp_candidate, tp_normal = getTargetPointNormalPar(pts, normals)

            # Check if a valid target has been found
            if tp_candidate is None:
                    fsm = 'NextBase'
                    print('Not enough faces detected, switching to state ', fsm)

            else:
                print("\nTarget candidate (rgbd casing ref): ", tp_candidate)

                # Transformation of target candidates into robot's arm reference
                #1) get position and Euler of rgbd sensor wrt robot's arm
                res, rgbdPos = vrep.simxGetObjectPosition(clientID, h["rgbdCasing"], h["armRef"], vrep.simx_opmode_oneshot_wait)
                vrchk(vrep, res, True)
                res, rgbdEuler = vrep.simxGetObjectOrientation(clientID, h["rgbdCasing"], h["armRef"], vrep.simx_opmode_oneshot_wait)
                vrchk(vrep, res, True)

                #2) transform the targeted point's coordinates into the robot's
                # arm reference
                # ** warning: x and y coordinates inverted from one ref to the other **
                T = np.reshape(SE2(rgbdEuler[2], 'rad', rgbdPos[0], rgbdPos[1]).data, (3,3))

                tp_1 = tp_candidate

                tp_1 = tp_1[:-1] # Get rid of depth information
                tmp = tp_1[2]
                tp_1[2] = tp_1[1] # Invert y and z position
                tp_1[1] = tmp
                tmp = tp_1[0]
                tp_1[0] = tp_1[1] # Invert x and y position
                tp_1[1] = tmp

                if object == 'Parallelipiped':
                    tp_1[0] += 0.01 # Add 'radius' of object
                    tp_1[1] -= 0.015

                trans_tp_1 = homtrans(T, tp_1[:2])
                tp_1 = np.append(trans_tp_1, tp_1[2] + rgbdPos[2])

                print("\nTarget candidate (arm ref): ", tp_1)

                if object == 'Parallelipiped':
                    #3) transform the targeted point's normal components into the
                    # robot's arm reference. (Parallelipiped only)
                    tp_normal_1 = tp_normal

                    tmp = tp_normal_1[2]
                    tp_normal_1[2] = tp_normal_1[1] # Invert y and z position
                    tp_normal_1[1] = tmp
                    tmp = tp_normal_1[0]
                    tp_normal_1[0] = tp_normal_1[1] # Invert x and y position
                    tp_normal_1[1] = tmp
                    norm = np.linalg.norm(tp_normal_1[:-1])
                    tp_normal_1 /= norm
                    tp_normal_1 = rotz(rgbdEuler[2]) * tp_normal_1.reshape(3, 1)

                    print("Associated normal (arm ref): ", tp_normal_1)

                #4) Check if the grasping target is valide regarding the
                # position of the arm and its capabilities.

                #a) Check if the target is not too far from the arm
                if tp_1[0] > 0.5:
                    fsm = 'NextBase'
                    print('Target grasping point too far from the arm, switching to state ', fsm)

                #b) Check if the target is not too close to the arm
                elif tp_1[0] < 0.2:
                    fsm = 'NextBase'
                    print('Target grasping point too close to the arm, switching to state ', fsm)

                #c) Check if the normal associated to the target candidate (Parallelipiped only)
                #if np.abs(np.abs(tp_normal_1[0]) - 1) > 0.2 and object == 'Parallelipiped':
                elif object == 'Parallelipiped':
                    v1 = np.squeeze(tp_normal_1[:-1])
                    v1 = v1/np.linalg.norm(v1)
                    v2 = np.squeeze(np.asarray([tp_1[0], tp_1[1]]))
                    v2 = v2/np.linalg.norm(v2)
                    print("\nv1 : ", v1)
                    print("\nv2 : ", v2)
                    dot_prdct = np.dot(v1, v2)
                    print("\ndot : ", dot_prdct)
                    if abs(abs(dot_prdct) - 1) > 0.05:
                        fsm = 'NextBase'
                        print('\nFace not well aligned, switching to state ', fsm)

                if fsm != 'NextBase':
                    tp = tp_1
                    print('\nTarget locked:  ', tp)
                    fsm = 'VerticalPose'
                    print('\nSwitching to state ', fsm)

        elif fsm == 'VerticalPose':
            # Set robot's arm upright
            if grasp_flag == False and grasp_flag_docking == False:
                base_joint_angle = np.pi - np.arctan2(tp[0], tp[1])
                last_join_angle = 0.0
            elif drop_flag == True:
                base_joint_angle = 0.0
                last_join_angle = 0.0
            elif grasp_flag == True:
                base_joint_angle = np.pi - np.arctan2(tp[0], tp[1])
                last_join_angle = -1.2*np.pi/8
            elif grasp_flag_docking == True:
                base_joint_angle = 0.0
                last_join_angle = -1.2*np.pi/8
            else:
                base_joint_angle = np.pi - np.arctan2(tp[0], tp[1])
                last_join_angle = 0.0
            angle = [base_joint_angle, 0, 0, last_join_angle, 0]
            joint = np.zeros((5,))
            # Only joints 0, 1 and 3 are of interest
            idx_to_move = [0, 1, 3]
            for i in idx_to_move:
                res = vrep.simxSetJointTargetPosition(clientID, h['armJoints'][i], angle[i], vrep.simx_opmode_oneshot)
                res, joint[i] = vrep.simxGetJointPosition(clientID, h['armJoints'][i], vrep.simx_opmode_buffer)

            # Check if the robot's arm is upright
            crit_0 = abs(angdiff(joint[0], base_joint_angle)) < 0.001
            crit_1 = abs(angdiff(joint[1], 0)) < 0.001
            if grasp_flag == True or grasp_flag_docking == True:
                crit_2 = abs(angdiff(joint[3], last_join_angle)) < 0.001
            else:
                crit_2 = abs(angdiff(joint[3], 0)) < 0.001

            if crit_0 & crit_1 & crit_2:
                if grasp_flag_docking == True:
                    grasp_flag_docking = False
                    fsm = 'GraspingDepositDrop_phase1'
                # If the robot's arm is upright, it can start the grasping phase
                elif grasp_flag == False:
                    fsm = 'GraspingStart'
                elif grasp_flag == True:
                    grasp_flag = False
                    fsm = 'GraspingPlace'
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingStart':
            #Rotate rgb/xyz sensor
            vrep.simxSetObjectOrientation(clientID, h['rgbdCasing'], h['ref'], (0.0, 0.0, -np.pi/2), vrep.simx_opmode_oneshot)
            # Get the arm tip position
            res, tpos = vrep.simxGetObjectPosition(clientID, h['ptip'], h['armRef'], vrep.simx_opmode_buffer) # try without that, seems useless
            vrchk(vrep, res, True)
            # Send the order to move the arm
            res = vrep.simxSetIntegerSignal(clientID, 'km_mode', 2, vrep.simx_opmode_oneshot_wait) # Activation of the inverse kinetic mode
            vrchk(vrep, res, True)

            # The robot can now perform the proper grasping phase
            fsm = 'GraspingApproach'
            print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingApproach':
            tp_1 = tp.copy()
            tp_1[2] = tp[2] + 0.1

            # Transformation to the rectangle22 reference
            rot1 = R.from_quat([0., np.sin(-3/8*np.pi), 0., np.cos(-3/8*np.pi)])
            rot2 = R.from_quat([np.sin(-np.pi/4), 0., 0., np.cos(-np.pi/4)])
            quats = (rot1*rot2).as_quat()
            res = vrep.simxSetObjectQuaternion(clientID, h["otarget"], h["r22"], quats, vrep.simx_opmode_oneshot_wait)
            vrchk(vrep, res, True)

            # Set gripper's target position
            res = vrep.simxSetObjectPosition(clientID, h["ptarget"], h["armRef"], tp_1, vrep.simx_opmode_oneshot)
            vrchk(vrep, res, True)

            # Get the gripper position and check whether it is at destination
            res, tpos = vrep.simxGetObjectPosition(clientID, h["ptip"], h["armRef"], vrep.simx_opmode_buffer)
            vrchk(vrep, res, True)
            vrep.simxSynchronousTrigger(clientID)

            # Check if the gripper reached its target
            crit_0 = abs(tpos[0] - tp_1[0]) < 0.001
            crit_1 = abs(tpos[1] - tp_1[1]) < 0.001
            crit_2 = abs(tpos[2] - tp_1[2]) < 0.001

            if crit_0 & crit_1 & crit_2:
                fsm = 'GraspingDocking'
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingDocking':
            # Set gripper's target position
            res = vrep.simxSetObjectPosition(clientID, h["ptarget"], h["armRef"], tp, vrep.simx_opmode_oneshot)
            vrchk(vrep, res, True)

            # Get the gripper position and check whether it is at destination
            res, tpos = vrep.simxGetObjectPosition(clientID, h["ptip"], h["armRef"], vrep.simx_opmode_buffer)
            vrchk(vrep, res, True)
            vrep.simxSynchronousTrigger(clientID)

            # Check if the gripper reached its target
            crit_0 = abs(tpos[0] - tp[0]) < 0.001
            crit_1 = abs(tpos[1] - tp[1]) < 0.001
            crit_2 = abs(tpos[2] - tp[2]) < 0.001

            if crit_0 & crit_1 & crit_2:
                fsm = 'GraspingGrasp'
                cnter = 10
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingGrasp':
            res = vrep.simxSetIntegerSignal(clientID, "gripper_open", 0, vrep.simx_opmode_oneshot_wait)
            vrchk(vrep, res, True)
            cnter -= 1
            if cnter == 0:
                res = vrep.simxClearIntegerSignal(clientID, 'km_mode', vrep.simx_opmode_oneshot_wait) # De-Activation of the inverse kinetic mode
                vrchk(vrep, res, True)
                fsm = 'VerticalPose'
                grasp_flag = True
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingPlace':
            # Set robot's arm along robot's main axis, above deposit area
            base_joint_angle = 0.0
            joint = np.zeros((5,))
            angle = [base_joint_angle, -np.pi/8, 0, -1.8*np.pi/4, 0]
            # Only joints 0, 1 and 3 are of interest
            idx_to_move = [0, 1, 3]
            for i in idx_to_move:
                res = vrep.simxSetJointTargetPosition(clientID, h['armJoints'][i], angle[i], vrep.simx_opmode_oneshot)
                res, joint[i] = vrep.simxGetJointPosition(clientID, h['armJoints'][i], vrep.simx_opmode_buffer)

            # Check if the robot's arm is well set
            crit_0 = abs(angdiff(joint[0], base_joint_angle)) < 0.001
            crit_1 = abs(angdiff(joint[1], -np.pi/8)) < 0.001
            crit_2 = abs(angdiff(joint[3], -1.8*np.pi/4)) < 0.001

            if crit_0 & crit_1 & crit_2:
                if drop_flag == True:
                    drop_flag = False
                    fsm = 'PerpExpulsionDeposit'
                else:
                    deposit_iter += 1
                    fsm = 'GraspingSelfDrop_phase1'
                print('\nSwitching to state: ', fsm)

        elif fsm == 'PerpExpulsionEasyTable':
            # Make the robot go away from the easy table in a perpendicular fashion
            dist = math.sqrt((x_table-youbotPos[0])**2 + (y_table-youbotPos[1])**2)
            rightVel = -1/dist
            if dist >= 0.8 :
                rightVel = 0
                fsm = 'ToDepositTarget'
                print('\nSwitching to state ', fsm)

        elif fsm == 'PerpExpulsionDeposit':
            # Make the robot go away from the deposit table in a perpendicular fashion
            dist = math.sqrt((x_deposit-youbotPos[0])**2 + (y_deposit-youbotPos[1])**2)
            rightVel = -1/dist
            if dist >= 0.8 :
                rightVel = 0
                fsm = 'ToEasyTableTarget'
                print('\nSwitching to state ', fsm)

        elif fsm == 'ToDepositTarget':
            deposit_close_pt, deposit_dock_pt = getDepositPoint(deposit_iter, x_deposit, y_deposit)
            x_start, y_start = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)
            x_target, y_target = int(( deposit_dock_pt[0] + (sceneSize / 2) ) / resol), int(( deposit_dock_pt[1] + (sceneSize / 2) ) / resol)
            target = np.array([x_target, y_target])

            # -- Inflate the map --
            BW_map = (map >= 2).astype(int) # Binarized map s.t. 0 = free, 1 = obstacle
            BW_map = ndimage.binary_dilation(BW_map, iterations=2).astype(BW_map.dtype)

            robot_inflate_map = (map < 0) # 0 everywhere
            robot_inflate_map[y_start, x_start] = 1
            robot_inflate_map = ndimage.binary_dilation(robot_inflate_map, iterations=2).astype(robot_inflate_map.dtype)
            robot_inflate_map = ~ robot_inflate_map

            BW_map = np.logical_and(BW_map, robot_inflate_map).astype(int)
            fsm = 'ToDepositTraj'
            print('Switching to state ', fsm)

        elif fsm == 'ToDepositTraj':
            print('\tComputing new trajectory...')

            # Distance transform path planing

            dx = DXform(BW_map)
            dx.plan(goal = target.astype(int))

            traj = dx.query((int(x_start), int(y_start)), animate = False)

            print("\tTrajectory found !")
            trajFlag = True

            for i, element in enumerate(traj):
                traj[i] = np.flip(element)

            fsm = 'ToDepositTrajSmoothing'
            print('Switching to state ', fsm)

        elif fsm == 'ToDepositTrajSmoothing':
            # Smoothing and downsampling of the previously computed trajectory
            print("\tSmoothing the trajectory...")
            # Regular downsampling by a factor 2
            traj_pro = traj[::2]
            # Trajectory-specific downsampling
            traj_pro = trajectory_smoother(traj_pro)

            # Find the B-spline representation of the trajectory.

            tmp1 = np.zeros(len(traj_pro))
            tmp2 = np.zeros(len(traj_pro))
            for i, element in enumerate(traj_pro):
                tmp1[i] = element[0]
                tmp2[i] = element[1]
            smooth_traj = [tmp1, tmp2]

            smoothtrajFlag = True

            fsm = 'ToDepositReorientation'
            print('\nSwitching to state: ', fsm)

        elif fsm == 'ToDepositReorientation':
            # Reorientating the robot before following the computed trajectory
            x_d, y_d = int(( youbotPos[0] + (sceneSize / 2) ) / resol), int(( youbotPos[1] + (sceneSize / 2) ) / resol)

            target_angle = atan2(smooth_traj[0][1] - y_d, smooth_traj[1][1] - x_d)

            # Change of reference
            if target_angle < 0 :
                target_angle = 2 * np.pi + target_angle

            head_orientation = youbotEuler[2] - np.pi / 2

            if head_orientation < - np.pi:
                head_orientation = head_orientation % np.pi

            if head_orientation < 0 :
                head_orientation = 2 * np.pi + head_orientation

            # Proportional controller
            rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

            if rotateRightVel < -np.pi or rotateRightVel > np.pi:
                rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

            # Switch to the trajectory following task when the robot is
            # well oriented
            if abs(rotateRightVel) < .002:
                rotateRightVel = 0

                traj_idx = 1    # First trajectory step not considered
                new_state = True
                next_state_y = smooth_traj[0][traj_idx]
                next_state_x = smooth_traj[1][traj_idx]

                fsm = 'ToDepositTrajFollow'
                print('Switching to state: ', fsm)

        elif fsm == 'ToDepositTrajFollow':
            # Trajectory following procedure

            if new_state == True:
                next_state_y = smooth_traj[0][traj_idx]
                next_state_x = smooth_traj[1][traj_idx]
                traj_idx += 1

                x_state, y_state = next_state_x * resol - (sceneSize / 2), next_state_y * resol - (sceneSize / 2)

                new_state = False
                rotation_allowed = True
                rotating = False

            # -- Forward speed control --
            x_curr = youbotPos[0]
            y_curr = youbotPos[1]

            dist = sqrt((x_curr-x_state)**2 + (y_curr-y_state)**2)
            forwBackVel = - 3 * dist

            if traj_idx == len(smooth_traj[0]) - 1: # avoid crashes
                forwBackVel = - dist


            # -- Orientation control --
            y_d, x_d = int(( youbotPos[1] + (sceneSize / 2) ) / resol), int(( youbotPos[0] + (sceneSize / 2) ) / resol)

            target_angle = atan2(next_state_y - y_d, next_state_x - x_d)

            if target_angle < 0 :
                target_angle = 2 * np.pi + target_angle

            head_orientation = youbotEuler[2] - np.pi / 2

            if head_orientation < - np.pi:
                head_orientation = head_orientation % np.pi

            if head_orientation < 0 :
                head_orientation = 2 * np.pi + head_orientation

            # Proportional controller
            rotateRightVel = (target_angle - head_orientation) % (2 * np.pi - 0.01)

            if rotateRightVel < - np.pi or rotateRightVel > np.pi:
                rotateRightVel = rotateRightVel-(np.sign(rotateRightVel))*2*np.pi

            if rotating == True :
                rotateRightVel = rotateRightVel
                forwBackVel = 0.0
                if abs(rotateRightVel) < 0.01:
                    rotating = False
                    rotateRightVel = 0.0

            if abs(rotateRightVel) > 0.1 and rotating == False:
                # If angle too large, stop the robot and reorienting
                rotateRightVel = rotateRightVel
                forwBackVel = 0.0
                rotating = True

            elif rotating == False:
                rotateRightVel = 0.0

            # -- Trajectory handling --
            if  dist < .1:
                new_state = True
                if traj_idx == len(smooth_traj[0]):
                    # Target reached
                    print("\tTarget reached !")
                    new_state = False
                    forwBackVel = 0.0
                    rotateRightVel = 0.0

                    fsm = 'ToDepositSelfRotatePhase1'
                    print('Switching to state ', fsm)

        elif fsm == 'GraspingSelfDrop_phase1':

            # Drop object on robot's deposit area
            # Set robot's arm along robot's main axis, above deposit area
            base_joint_angle = 0
            joint = np.zeros((5,))
            angle = [base_joint_angle, np.pi/9, 5*np.pi/8, -1.7*np.pi/4, 0]
            # Only joints 0, 2 and 3 are of interest
            idx_to_move = [0, 2, 3]
            for i in idx_to_move:
                res = vrep.simxSetJointTargetPosition(clientID, h['armJoints'][i], angle[i], vrep.simx_opmode_oneshot)
                res, joint[i] = vrep.simxGetJointPosition(clientID, h['armJoints'][i], vrep.simx_opmode_buffer)

            # Check if the robot's arm is well set
            crit_0 = True
            crit_2 = abs(angdiff(joint[2], 5*np.pi/8)) < 0.001
            crit_3 = abs(angdiff(joint[3], -1.7*np.pi/4)) < 0.001

            if crit_0 & crit_2 & crit_3:
                fsm = "GraspingSelfDrop_phase2"
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingSelfDrop_phase2':

            # Drop object on robot's deposit area
            # Set robot's arm along robot's main axis, above deposit area
            base_joint_angle = 0
            joint = np.zeros((5,))
            angle = [base_joint_angle, 1.1*np.pi/4, np.pi/4, -np.pi, 0]
            # Only joints 1 is of interest
            idx_to_move = [1]
            for i in idx_to_move:
                res = vrep.simxSetJointTargetPosition(clientID, h['armJoints'][i], angle[i], vrep.simx_opmode_oneshot)
                res, joint[i] = vrep.simxGetJointPosition(clientID, h['armJoints'][i], vrep.simx_opmode_buffer)

            crit_1 = abs(angdiff(joint[1], 1.1*np.pi/4)) < 0.001

            if crit_1:
                vrchk(vrep, res, True)
                cnter_drop = 0
                if cnter_drop == 0:
                    cnter = 10
                    fsm = 'PerpExpulsionEasyTable'
                    print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingDepositGrasp':
            res = vrep.simxSetIntegerSignal(clientID, "gripper_open", 0, vrep.simx_opmode_oneshot_wait)
            vrchk(vrep, res, True)
            cnter -= 1
            if cnter == 0:
                res = vrep.simxClearIntegerSignal(clientID, 'km_mode', vrep.simx_opmode_oneshot_wait) # De-Activation of the inverse kinetic mode
                vrchk(vrep, res, True)
                fsm = 'VerticalPose'
                grasp_flag_docking = True
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingDepositDrop_phase2':

            # Set robot's arm along robot's main axis, above deposit area
            base_joint_angle = np.pi/3
            joint = np.zeros((5,))
            angle = [base_joint_angle, np.pi/9, 0.9*np.pi/4, 0, 0]
            # Only joints 2 and 3 are of interest
            idx_to_move = [2, 3]
            for i in idx_to_move:
                res = vrep.simxSetJointTargetPosition(clientID, h['armJoints'][i], angle[i], vrep.simx_opmode_oneshot)
                res, joint[i] = vrep.simxGetJointPosition(clientID, h['armJoints'][i], vrep.simx_opmode_buffer)

            # Check if the robot's arm is well set
            crit_0 = True
            crit_2 = abs(angdiff(joint[2], 0.9*np.pi/4)) < 0.001
            crit_3 = abs(angdiff(joint[3], 0)) < 0.001

            if crit_0 & crit_2 & crit_3:
                cnter_drop -= 1
                if cnter_drop == 0:
                    # Drop the grasped object on the table
                    res = vrep.simxClearIntegerSignal(clientID, "gripper_open", vrep.simx_opmode_oneshot_wait)
                    vrchk(vrep, res, True)
                    cnter = 10
                    fsm = "GraspingPlace"
                    drop_flag = True
                    #Rotate rgb/xyz sensor
                    vrep.simxSetObjectOrientation(clientID, h['rgbdCasing'], h['ref'], (0.0, 0.0, np.pi/4), vrep.simx_opmode_oneshot)
                    print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingDepositDrop_phase1':
            base_joint_angle = np.pi/2
            joint = np.zeros((5,))
            angle = [base_joint_angle, np.pi/4, np.pi/4, -np.pi, 0]
            idx_to_move = [0, 1]
            for i in idx_to_move:
                res = vrep.simxSetJointTargetPosition(clientID, h['armJoints'][i], angle[i], vrep.simx_opmode_oneshot)
                res, joint[i] = vrep.simxGetJointPosition(clientID, h['armJoints'][i], vrep.simx_opmode_buffer)

            crit_0 = abs(angdiff(joint[0], base_joint_angle)) < 0.001
            crit_1 = abs(angdiff(joint[1], np.pi/4)) < 0.001

            if crit_1 and crit_0:
                fsm = "GraspingDepositDrop_phase2"
                cnter_drop = 10
                print('\nSwitching to state: ', fsm)

        elif fsm == 'GraspingEndProcedure':
            print('Grasping phase finished')
            break

        # -- Update wheel velocities --
        h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

        # -- Update the plot --
        y_d, x_d = int(( youbotPos[1] + (sceneSize / 2) ) / resol), int(( youbotPos[0] + (sceneSize / 2) ) / resol)
        img = np.zeros((map.shape[0], map.shape[1], 3), dtype = np.uint8)

        discovered_idx = np.where(map == 0)
        not_discovered_idx = np.where(map == 1)
        walls_idx = np.where(map == 2)
        robot_idx = (np.array([y_d-1, y_d-1, y_d-1, y_d, y_d, y_d, y_d+1, y_d+1, y_d+1]), np.array([x_d-1, x_d, x_d+1, x_d-1, x_d, x_d+1, x_d-1, x_d, x_d+1]))

        img[discovered_idx] = DISCOVERED_CLR
        img[walls_idx] = WALL_CLR

        if trajFlag == True:
            trajectory_idx = (traj[:, 0], traj[:, 1])
            img[trajectory_idx] = TRAJ_CLR

        if smoothtrajFlag == True:
            smooth_trajectory_idx = (smooth_traj[0].astype(int), smooth_traj[1].astype(int))
            img[smooth_trajectory_idx] = SMOOTH_TRAJ_CLR

        img[robot_idx] = ROBOT_CLR

        img = cv2.resize(img, (500, 500))
        cv2.imshow("House map", img)
        cv2.waitKey(1)

        vrep.simxSynchronousTrigger(clientID)

    cleanup_vrep(vrep, clientID)
    """fig2 = plt.plot()
    bins = np.arange(0, 0.5, 0.001) # fixed bin size
    plt.xlim([min(elapsed_plot)-0.005, max(elapsed_plot)+0.005])
    plt.hist(elapsed_plot, bins=bins, alpha=0.5)
    plt.axvline(min(elapsed_plot), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(max(elapsed_plot), color='b', linestyle='dashed', linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Number of loops [-]")
    plt.savefig("elapsed2.pdf")
    plt.show()"""

    print('Simulation has stopped')
