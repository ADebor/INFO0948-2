# -*- coding: utf-8 -*-
"""
INTELLIGENT ROBOTICS - PROJECT 2021

PART 1 - Navigation

Antoine DEBOR & Pierre NAVEZ

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


from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_init import youbot_init
from youbot_drive import youbot_drive
from youbot_hokuyo_init import youbot_hokuyo_init
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
from beacon import beacon_init, youbot_beacon
from utils_sim import angdiff

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

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Initiate the connection to the simulator.
    print('IR 2021 - Part 1 - DEBOR & NAVEZ \nProgram started')
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



    ##############################################################################
    #                                                                            #
    #                          INITIAL CONDITIONS                                #
    #                                                                            #
    ##############################################################################
    # Define all the variables which will be used through the whole simulation.
    # Important: Set their initial values.

    # Get the position of the beacons in the world coordinate frame (x, y)
    beacons_world_pos = np.zeros((len(beacons_handle), 3))
    for i, beacon in enumerate(beacons_handle):
        res, beacons_world_pos[i] = vrep.simxGetObjectPosition(clientID, beacon, -1,
                                                               vrep.simx_opmode_buffer)

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

    # Send a Trigger to the simulator: this will run a time step for the physic engine
    # because of the synchronous mode. Run several iterations to stabilize the simulation
    for i in range(int(1./timestep)):
        vrep.simxSynchronousTrigger(clientID)
        #vrep.simxGetPingTime(clientID)

    # Start the robot.
    while True:
        try:
            t_first = time.time()
            # Check the connection with the simulator
            if vrep.simxGetConnectionId(clientID) == -1:
                sys.exit('Lost connection to remote API.')

            # Get the current position and orientation of the robot
            res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
            vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error
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

                print('Switching to state: ', fsm)

            elif fsm == 'reorientation':
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

            """elapsed = time.time() - t_first
            elapsed_plot.append(elapsed)"""

            # Send a Trigger to the simulator: this will run a time step for the physic engine
            # because of the synchronous mode.
            vrep.simxSynchronousTrigger(clientID)
            #vrep.simxGetPingTime(clientID)

        except KeyboardInterrupt:
            cleanup_vrep(vrep, clientID)
            sys.exit('Stop simulation')

    cleanup_vrep(vrep, clientID)
    """fig2 = plt.plot()
    bins = np.arange(0, 0.1, 0.001) # fixed bin size
    plt.xlim([min(elapsed_plot)-0.005, max(elapsed_plot)+0.005])
    plt.hist(elapsed_plot, bins=bins, alpha=0.5)
    plt.axvline(min(elapsed_plot), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(max(elapsed_plot), color='b', linestyle='dashed', linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Number of loops [-]")
    plt.savefig("elapsed.pdf")"""

    print('Simulation has stopped')
