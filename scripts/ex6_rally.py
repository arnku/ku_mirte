import cv2 # Import the OpenCV library
import sys
import os
import numpy as np
import time
import math


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

from particle_selflocalize import SelfLocalizer
import rrt

CANVASS = True

if CANVASS:
    import tkinter as tk
    CANVAS_WIDTH = 1200
    CANVAS_HEIGHT = 1200
    window = tk.Tk()
    canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)


def lidar_positions(measured_lidar: np.ndarray, lidar_diff_theshold: float = 5, lidar_max_dist: float = 20, object_size: tuple = (0.17,0.3)) -> np.ndarray:
    measured_lidar = np.array(measured_lidar, dtype=np.float64)
    measured_lidar[measured_lidar == np.inf] = lidar_max_dist
    
    # Mask out robot components
    measured_lidar[29:36] = lidar_max_dist # TODO: This can be done smarter
    measured_lidar[324:331] = lidar_max_dist
    
    # Segment lidar into objects based on difference threshold
    lidar_diffs = np.diff(measured_lidar, axis=0) # calculating lidar differentials
    lidar_diffs[abs(lidar_diffs) < lidar_diff_theshold] = 0 # filtering out small differences
    segment_indices = np.where(lidar_diffs)[0] # Segment the lidar data into objects
    segments = np.split(measured_lidar, segment_indices + 1)
    
    # Analyze segments
    min_segment_dists = np.array([np.min(segment) for segment in segments]) # Find the minimum distance in each segment
    index_widths = np.diff(segment_indices, prepend=0, append=measured_lidar.shape[0]) # Find the angle width of each segment
    angle_widths = np.deg2rad(index_widths) # since there are 360 angles, the index width is the same as the angle width
    object_widths = min_segment_dists * np.tan(angle_widths / 2) # Calculate the actual width of each object

    # Identify boxes
    box_indices = np.where((object_widths > object_size[0]) & (object_widths < object_size[1]))[0] # get the index of objects that are around the size of a box
    if not np.any(box_indices):
        return None, None
    
    # Get the distances and angles of the objects
    object_dists = min_segment_dists[box_indices]
    object_location = np.where(np.isin(measured_lidar, object_dists))[0]
    object_angles = np.deg2rad(object_location) # Here again the indicies can be directly used as angles

    return object_dists, object_angles

aruco_size = 0.25 # m
box_size = 0.4 # m
path_res = 0.1

mirte = KU_Mirte()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

selflocalizer = SelfLocalizer(200, [[0,0], [0,4], [4,0], [4,4]], [0,1,2,3], bounds=(-6, -6, 6, 6))
selflocalizer.particle_filter.weights_used = "aruco"

robot = rrt.PointMassModel(ctrl_range=[-path_res, path_res])
occ_map = rrt.GridOccupancyMap(low=(-6, -6), high=(6, 6), res=path_res)
path_find = rrt.RRT(robot, occ_map, expand_dis=3 * path_res, path_resolution=path_res, max_iter=200)


while cv2.waitKey(4) == -1: # Wait for a key pressed event
    # get reading from the camera
    image = mirte.get_image()
    lidar = mirte.get_lidar_ranges()
    # find aruco markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    #cv2.imshow("Detected Markers", image)

    # Aruco marker positions
    distances = []
    angles = []
    points = []
    colors = []
    if ids is not None:
        # Estimate marker box positions
        ids = ids.flatten()
        corners = [corners[i] for i in range(len(corners)) if ids[i] not in ids[:i]] # Remove duplicates from corners while preserving order
        ids = np.array(list(dict.fromkeys(ids))) # Remove duplicates from ids while preserving order
        rvecs, tvecs, rejectedImgPoints = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, mirte.k_matrix, mirte.d_matrix)
        tvecs = [[tvec[0][2], -tvec[0][0], -tvec[0][1]] for tvec in tvecs]
        
        for i in range(len(ids)):
            # Draw the markers
            points.append([tvecs[i][0], tvecs[i][1], 0])
            if ids[i] in  [0,1,2,3]:
                colors.append([0,255,0,255])
            else:
                colors.append([255,0,0,255])

            # Get box positions
            distance = np.linalg.norm(tvecs[i])
            angle = np.arctan2(tvecs[i][1], tvecs[i][0])
            distances.append(distance)
            angles.append(angle)
    
    # Lidar box positions
    lidar_dists, lidar_angles = lidar_positions(lidar)
    for i in range(len(lidar_dists)): # Find the positions of the boxes in relation to the robot
        angle = lidar_angles[i]
        distance = lidar_dists[i]
        x = distance * -np.cos(angle)
        y = distance * -np.sin(angle)
        points.append([x, y, 0])
        colors.append([255, 255, 0, 255])
    # Visualize the boxes
    mirte.set_pointcloud('mirte', points, colors)

    # Particle filter
    selflocalizer.add_uncertainty()
    selflocalizer.update_image(ids,distances, angles, lidar)
    selflocalizer.particle_filter.resample_particles()
    selflocalizer.random_positions(1)

    # Draw the particles
    points = []
    colors = []
    for p in selflocalizer.particle_filter.particles.positions:
        points.append([p[0], p[1], 0])
    for w in selflocalizer.particle_filter.particles.weights:
        colors.append([0,0,255,min(255, 255 * w * selflocalizer.particle_filter.particles.num_particles)])
    
    estimate = selflocalizer.particle_filter.estimate_pose()
    points.append([estimate.x, estimate.y, 0])
    colors.append([0,255,255,255])
    mirte.set_pointcloud('world', points, colors)
    print(f"Estimated position: {estimate.x }, {estimate.y }")
    print(f"Estimated direction (degrees): {np.rad2deg(estimate.theta)}")

    marker_list = []
    for i in range(len(lidar_dists)):
        angle = lidar_angles[i]
        distance = lidar_dists[i]
        marker_x = -distance * np.cos(angle)
        marker_y = distance * np.sin(angle)
        marker_list.append([marker_x, marker_y])
    
    print(f"Markers: {marker_list}")
    occ_map.populate(marker_list, [box_size] * len(marker_list))
    mirte.set_occupancy_grid(occ_map.grid, occ_map.resolution)

    # Goal position
    goal_global_position = [4, 4]
    goal_position_vector = np.array(goal_global_position) - np.array(estimate.position)

    goal_position_angle = estimate.theta - np.arctan2(goal_position_vector[1], goal_position_vector[0])

    goal_position_x = np.cos(goal_position_angle) * np.linalg.norm(goal_position_vector)
    goal_position_y = np.sin(goal_position_angle) * np.linalg.norm(goal_position_vector)
    
    print(f"Goal position: {goal_position_x}, {goal_position_y}")   

    edges = []
    colours = []
    path = path_find.do_rrt((0,0), (goal_position_x, goal_position_y))
    print(f"Path: {path}")
    if path is not None:
        for i in range(len(path) - 1):
            edges.append(((path[i][0],-path[i][1]), (path[i + 1][0], -path[i + 1][1])))
            colours.append((255, 0, 0, 255))
    mirte.set_tree('mirte', edges, colours)

    if CANVASS:
        canvas.delete("all")
        occ_map.canvas_draw(canvas)
        path_find.canvas_draw(path, canvas, low=(-6, -6), high=(6, 6))
        canvas.pack()
        window.update()



#cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node