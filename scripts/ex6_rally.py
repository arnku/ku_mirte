import cv2 # Import the OpenCV library
import sys
import os
import numpy as np
import time
import tkinter as tk
import math


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

from particle_selflocalize import SelfLocalizer
import rrt

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_RES = 10 # Resolution of the grid
window = tk.Tk()
canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

aruco_size = 0.25 # m
box_size = 0.4 # m

mirte = KU_Mirte()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

selflocalizer = SelfLocalizer(200, [[0,0], [0,4], [4,0], [4,4]], [0,1,2,3], bounds=(-4, -4, 8, 8))
selflocalizer.particle_filter.weights_used = "aruco"

robot = rrt.PointMassModel(ctrl_range=[-CANVAS_RES, CANVAS_RES])
occ_map = rrt.GridOccupancyMap(low=(0, 0), high=(CANVAS_WIDTH, CANVAS_HEIGHT), res=CANVAS_RES)
path_find = rrt.RRT(robot, occ_map, expand_dis=3 * CANVAS_RES, path_resolution=CANVAS_RES)

def lidar_positions(measured_lidar: np.ndarray, lidar_diff_theshold: float = 5, lidar_max_dist: float = 20, object_size: tuple = (0.17,0.3)) -> np.ndarray:
    """
    
    """
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


while cv2.waitKey(4) == -1: # Wait for a key pressed event
    # get reading from the camera
    image = mirte.get_image()
    lidar = mirte.get_lidar_ranges()
    # find aruco markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow("Detected Markers", image)

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

    #
    marker_list = []
    for i in range(len(lidar_dists)):
        angle = lidar_angles[i]
        distance = lidar_dists[i]
        marker_x = distance * np.cos(angle) * 100
        marker_y = distance * np.sin(angle) * 100

        canvas_x = int(-marker_x + CANVAS_WIDTH // 2)
        canvas_y = int(marker_y + CANVAS_HEIGHT // 2)
        radius = int((1.41 * box_size * 100) // 2)

        marker_list.append([canvas_x, canvas_y])
    
    occ_map.populate(marker_list, [radius] * len(marker_list))

    # North direction from the robot
    print(f"Estimated position: {estimate.x * 100}, {estimate.y * 100}")
    print(f"Estimated direction (degrees): {np.rad2deg(estimate.theta)}")
    robot_angle = estimate.theta
    canvas.create_line(
        CANVAS_WIDTH // 2,
        CANVAS_HEIGHT // 2,
        CANVAS_WIDTH // 2 + 50 * np.cos(robot_angle),
        CANVAS_HEIGHT // 2 + 50 * np.sin(robot_angle),
        fill="#000000",
        width=5,
    )    

    # Goal position
    goal_position = [400, 400]
    goal_position_delta = np.array(goal_position) - np.array(estimate.position) * 100
    goal_position_angle = robot_angle - np.arctan2(goal_position_delta[1], goal_position_delta[0])

    goal_canvas_position_x = np.cos(goal_position_angle) * np.linalg.norm(goal_position_delta)
    goal_canvas_position_y = np.sin(goal_position_angle) * np.linalg.norm(goal_position_delta)
    canvas_goal_x = int(goal_canvas_position_x + CANVAS_WIDTH // 2)
    canvas_goal_y = int(goal_canvas_position_y + CANVAS_HEIGHT // 2)
    
    path = rrt.do_rrt([CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2], [canvas_goal_x,canvas_goal_y])

    occ_map.canvas_draw(canvas)
    rrt.canvas_draw(path, canvas)

    canvas.pack()
    window.update()
    canvas.delete("all")



cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node