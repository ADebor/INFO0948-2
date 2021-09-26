import numpy as np
import math

# IMAGE/CLOUD PROCESSING CUSTOM FUNCTIONS
#########################################

def xyz2rgb_pnt(point):
    """
    Projects a point from the 3D XYZ frame to the 2D RGB frame
    ---
    Input:
    - point = [x, z, y, depth] : point to project
    ---
    Output:
    - proj = [x, y] : projection
    """
    proj_x = (-(point[0]/point[2])*1260 + 256).astype('int64')
    proj_y = (-(point[1]/point[2])*1260 + 256).astype('int64')
    proj = [proj_x, proj_y]

    return proj

def rgb2xyz_pnt(point, depth):
    """
    Projects a point from the 2D RGB frame to the 3D XYZ frame
    ---
    Input:
    - point = [x, y] : point to project
    - depth : depth of the point in the 3D space
    ---
    Output:
    - proj = [x, z, y, depth] : projection
    """
    x = - (depth-0.05)/1260 * (point[0] - 256)
    z = - (depth-0.05)/1260 * (point[1] - 256)
    proj = [x, z, depth, depth]

    return proj

def xyz2rgb_cloud(cloud):
    """
    Projects a entire point cloud from the XYZ frame to the RGB frame
    ---
    Input:
    - clound : point cloud to project
    ---
    Output:
    - proj : projection
    """
    proj_x = (-((cloud[:, 0])/cloud[:, 2])*1260 + 256)
    proj_y = (-((cloud[:, 1])/cloud[:, 2])*1260 + 256)
    proj = [proj_x, proj_y]

    return proj

def rgb2hsv(r, g, b):
    "gathered from https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/"
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.

    # if cmax and cmin are equal then h = 0
    if cmax == cmin:
        h = 0

    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    # compute v
    v = cmax * 100
    return h, s, v

def rgb2hsv_img(r, g, b):

    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = np.maximum.reduce([r, g, b])    # maximum of r, g, b
    cmin = np.minimum.reduce([r, g, b])    # minimum of r, g, b

    diff = cmax-cmin       # diff of cmax and cmin.

    hues = np.empty(diff.size)
    sat = np.empty(diff.size)
    v = np.empty(diff.size)

    for i, _ in enumerate(diff):
        # if cmax and cmin are equal then h = 0
        if diff[i] == 0:
            h = 0

        # if cmax equal r then compute h
        elif cmax[i] == r[i]:
            h = (60 * ((g[i] - b[i]) / diff[i]) + 360) % 360

        # if cmax equal g then compute h
        elif cmax[i] == g[i]:
            h = (60 * ((b[i] - r[i]) / diff[i]) + 120) % 360

        # if cmax equal b then compute h
        elif cmax[i] == b[i]:
            h = (60 * ((r[i] - g[i]) / diff[i]) + 240) % 360

        if cmax[i] == 0:
            s = 0
        else:
            s = (diff[i] / cmax[i]) * 100

        hues[i] = h
        sat[i] = s

    return hues, sat

def get_strd_hue(hue):
    hsv_strd = {0 : 'red',
                60 : 'yellow',
                120 : 'green',
                180: 'cyan',
                240: 'blue',
                300: 'magenta'}

    # Find closest hue
    hues = np.asarray(list(hsv_strd.keys()))
    idx = np.argmin(np.abs((hues - hue) % 360))
    strd_hue_val = hues[idx]
    strd_hue = hsv_strd[strd_hue_val]
    print('closest standard hue : ', strd_hue)
    print('and val : ', strd_hue_val)

    return strd_hue_val

def img_seg(img, clr):
    """
    Image segmentation to mask non relevant content regarding a given colour
    ---
    Input:
    - img : RGB image to segment
    ---
    Output:
    - seg_img : segmented RGB image
    """
    # Convert rgb to hsv
    r = clr[0]
    g = clr[1]
    b = clr[2]
    hue, sat, val = rgb2hsv(r, g, b)
    strd_hue = get_strd_hue(hue)

    dim0 = np.shape(img)[0]
    dim1 = np.shape(img)[1]
    img = np.reshape(img, (dim0*dim1, 3))
    r = img[:, 0]
    g = img[:, 1]
    b = img[:, 2]
    hues, sats = rgb2hsv_img(r, g, b)
    hues_disc = (np.round(hues/10)*10)%360

    indices_hue = np.where(hues_disc != strd_hue)
    indices_sat = np.where(sats <= 50)

    img[indices_hue, 0] = 0
    img[indices_hue, 1] = 0
    img[indices_hue, 2] = 0

    img[indices_sat, 0] = 0
    img[indices_sat, 1] = 0
    img[indices_sat, 2] = 0


    img = np.reshape(img, (dim0, dim1, 3))

    return img, 1, 1

# TARGETING CUSTOM FUNCTION
###########################

def getTargetPointNormalCyl(pts):
    """
    Given point cloud, estimate center of mass of the
    object and return it.
    (for Parallelipiped only)
    """
    """
    # Bounds
    lb = np.inf
    hb = -np.inf
    rmb = np.inf
    lmb = -np.inf

    lb_pt = None
    hb_pt = None
    rmb_pt = None
    lmb_pt = None

    # Find the lowest/highest/right-most/left-most points to estimate the
    # position of the center of mass.
    for pt in pts:
        if pt[0] > lmb:
            lmb = pt[0]
            lmb_pt = pt.copy()
        if pt[0] < rmb:
            rmb = pt[0]
            rmb_pt = pt.copy()
        if pt[1] > hb:
            hb = pt[1]
            hb_pt = pt.copy()
        if pt[1] < lb:
            lb = pt[1]
            lb_pt = pt.copy()"""
    # Find mean point of the captured point cloud.
    mean_x = np.mean(pts[:, 0])
    mean_y = np.mean(pts[:, 1])
    mean_z = np.mean(pts[:, 2])
    mean_depth = np.mean(pts[:, 3])

    # Account for the depth of the cylinder.
    mean_z += 0.024
    mean_depth += 0.024

    target_pt = [mean_x, mean_y, mean_z, mean_depth]

    return target_pt

def getTargetPointNormalPar(pts, normals):
    """
    Given point cloud and corresponding normals, find centers of detected faces
    and return them.
    (for Parallelipiped only)
    """
    ref_normals = []
    ref_normals.append(normals[0])
    idx_normals = []
    idx_normals.append([0])
    for i, normal in enumerate(normals[1:]):
        for k in range(len(ref_normals)):
            cross = np.cross(normal, ref_normals[k])
            if np.linalg.norm(cross) < 0.8:
                # the normal is // to ref_normals[k] !
                idx_normals[k].append(i+1)
                break
            elif k == len(ref_normals)-1:
                #no match
                ref_normals.append(normal)
                idx_normals.append([i+1])

    # Assumption: If a normal of reference has less than 20 instances, it is
    # considered as an artifact of the point cloud sensor and is not considered
    # as a face identifier.
    tmp_normals = []
    tmp_idx = []
    for i, el in enumerate(idx_normals):
        if len(el) >= 20:
            tmp_normals.append(ref_normals[i].copy())
            tmp_idx.append(idx_normals[i].copy())

    ref_normals = np.asarray(tmp_normals)
    idx_normals = np.asarray(tmp_idx)

    # Assumption: If, due to the previous post-processing, only one face is
    # detected, it should be the one facing the rgbd sensor and should thus be
    # unreachable for the arm. In such a case, the robot has to move to the next
    # base around the table.
    if len(ref_normals) < 2:
        return None, None

    mean_pts = []
    for element in idx_normals:
        face_pts = pts[element] #all points of a same face
        mean_pt = np.mean(face_pts, axis=0) #compute mean point of face
        mean_pts.append(mean_pt)

    mean_pts = np.asarray(mean_pts)
    target_idx = np.argmax(mean_pts[:, 3])

    target_pt = mean_pts[target_idx]
    FoI_normals = normals[idx_normals[target_idx]]
    target_normal = np.mean(FoI_normals, axis=0)

    return target_pt, target_normal

# MISC CUSTOM FUNCTIONS
#######################

def getBases(n=16, x_table=-3.0, y_table=-6.0):
    "Hardcoded positions around the easy table"
    angle = 2*np.pi/n
    d = 0.8
    safe_margin = 0.5
    bases = []
    traj_bases = []
    for i in range(16):
         x = x_table + (d/2 + safe_margin) * math.cos(i * angle)
         y = y_table + (d/2 + safe_margin) * math.sin(i * angle)
         bases.append((x, y))
         x = x_table + (d/2 + safe_margin - 0.2) * math.cos(i * angle)
         y = y_table + (d/2 + safe_margin - 0.2) * math.sin(i * angle)
         traj_bases.append((x, y))

    return bases, traj_bases #bases for approaching, traj_bases for moving around the table

def getDepositPoint(i, x_deposit, y_deposit):
    angle = 2*np.pi/16
    d = 0.8
    safe_margin = 0.5

    x = x_deposit + (d/2 + safe_margin) * math.cos(i * angle)
    y = y_deposit + (d/2 + safe_margin) * math.sin(i * angle)
    dock_pt = (x, y)

    x = x_deposit + (d/2 + safe_margin - 0.2) * math.cos(i * angle)
    y = y_deposit + (d/2 + safe_margin - 0.2) * math.sin(i * angle)
    close_pt = (x, y)

    return close_pt, dock_pt
