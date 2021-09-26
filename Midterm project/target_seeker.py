from math import sqrt, isnan
import numpy as np
from roboticstoolbox import DXform


def target_seeker(boundary, bot_pos, norm, bin_map, init_bot_pos):
    # Seek for a target to drive the robot to, according to the "norm" criterion in {Euclidean, DXform, combined}
    target = [init_bot_pos[1], init_bot_pos[0]]
    score_Euclide = []
    score_Dx = []

    if norm == "Euclidean":
        min = np.inf
        for x, y in boundary:
            norm = sqrt((x - bot_pos[1])**2 + (y - bot_pos[0])**2)
            score_Euclide.append(norm)
            if bin_map[int(x), int(y)] == 1:
                continue
            elif norm < min:
                min = norm
                target = [x, y]

    elif norm == "DXform":
        dx = DXform(bin_map)
        dx.plan(goal = np.asarray([bot_pos[0], bot_pos[1]]))
        distance_map = dx.distance_map

        min = np.inf
        for point in boundary:

            x = int(point[0])
            y = int(point[1])
            score_Dx.append(distance_map[x, y])
            if distance_map[x, y] < min:
                min = distance_map[x, y]
                target = [x, y]

    elif norm == "combined":
        for x, y in boundary:
            norm = sqrt((x - bot_pos[1])**2 + (y - bot_pos[0])**2)
            score_Euclide.append(norm)

        dx = DXform(bin_map)
        dx.plan(goal = np.asarray([bot_pos[0], bot_pos[1]]))
        distance_map = dx.distance_map

        for point in boundary:
            x = int(point[0])
            y = int(point[1])
            score_Dx.append(distance_map[x, y])


        min = np.inf
        for i, point in enumerate(boundary):
            if not isnan(distance_map[int(point[0]), int(point[1])]) :
                dist = score_Euclide[i] + score_Dx[i]
                if dist < min:
                    min = dist
                    target = [int(point[0]), int(point[1])]

    return np.asarray(target).astype(int)
