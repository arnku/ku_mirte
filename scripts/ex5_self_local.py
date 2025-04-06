import cv2 # Import the OpenCV library
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

from particle_selflocalize import SelfLocalizer

aruco_size = 0.25 # m
box_size = 0.4 # m

mirte = KU_Mirte()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

selflocalizer = SelfLocalizer(1000, [[0,0], [2,2]], [0,1], bounds=(-5, -5, 5, 5))

while cv2.waitKey(4) == -1: # Wait for a key pressed event

    # get reading from the camera

    image = mirte.get_image()

    # find aruco markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow("Detected Markers", image)

    if ids is None:
        continue
    ids = ids.flatten()

    # remove duplicate ids from corners
    corners = [corners[i] for i in range(len(corners)) if ids[i] not in ids[:i]]
    ids = np.unique(ids)

    rvecs, tvecs, rejectedImgPoints = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, mirte.k_matrix, mirte.d_matrix)

    tvecs = [[tvec[0][2], -tvec[0][0], -tvec[0][1]] for tvec in tvecs]
        
    points = []
    for i in range(len(ids)):
        points.append([tvecs[i][0], tvecs[i][1], 0])
    mirte.set_pointcloud('mirte', points)

    # find distance and angle to the markers

    distances = []
    angles = []
    for i in range(len(ids)):
        distance = np.linalg.norm(tvecs[i])
        angle = np.arctan2(tvecs[i][1], tvecs[i][0])
        distances.append(distance)
        angles.append(angle)

    print(f"Distances: {distances} and Angles: {angles}")

    # update particle filter

    
    #selflocalizer.add_uncertainty()
    selflocalizer.update_image(ids,distances, angles)
    #selflocalizer.random_positions(50)
    # plot the particles with pointcloud

    points = []
    colors = []
    for p in selflocalizer.particle_filter.particles.positions:
        points.append([p[0], p[1], 0])
    for w in selflocalizer.particle_filter.particles.weights:
        colors.append([0,0,255,min(255, 255 * w * 50)])
    
    #estimate = selflocalizer.particle_filter.estimate_pose()
    #points.append([estimate.x, estimate.y, 0])
    #colors.append([255,255,255,255])

    mirte.set_pointcloud('world', points, colors)

    #time.sleep(1)

    # plot the best particle


cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node