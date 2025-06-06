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

def lidar_positions(measured_lidar: np.ndarray, lidar_diff_theshold: float = 5, lidar_max_dist: float = 20, object_size: tuple = (0.17,0.3)) -> np.ndarray:
    measured_lidar = np.array(measured_lidar, dtype=np.float64)
    measured_lidar[measured_lidar == np.inf] = lidar_max_dist
    
    # Mask out robot components
    # 29 to 36
    measured_lidar[29:32] = measured_lidar[28]
    measured_lidar[33:36] = measured_lidar[37]
    # 324 to 331
    measured_lidar[324:328] = measured_lidar[323]
    measured_lidar[329:331] = measured_lidar[332]
    
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
box_size = 0.35 # m
path_res = 0.1
goal_order = [[4,4], [0,4], [4,0], [0,0]]
iteration_time = time.time()
drive_speed = 0.5 # m/s
turn_speed = 0.5 # rad/s
goal_margin = 1.2 # m

mirte = KU_Mirte()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

selflocalizer = SelfLocalizer(200, [[0,0], [0,4], [4,0], [4,4]], [0,1,2,3], bounds=(-6, -6, 6, 6), starting_area=(-1, -1, 1, 1))
selflocalizer.particle_filter.weights_used = "aruco"

robot = rrt.PointMassModel(ctrl_range=[-path_res, path_res])
occ_map = rrt.GridOccupancyMap(low=(-6, -6), high=(6, 6), res=path_res)
path_find = rrt.RRT(robot, occ_map, expand_dis=3 * path_res, path_resolution=path_res, goal_radius=1.0, max_iter=200)


def process_aruco_markers(image, aruco_dict, parameters, mirte, aruco_size):
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    distances = []
    angles = []
    points = []
    colors = []
    if ids is not None:
        ids = ids.flatten()
        corners = [corners[i] for i in range(len(corners)) if ids[i] not in ids[:i]]
        ids = np.array(list(dict.fromkeys(ids)))
        rvecs, tvecs, rejectedImgPoints = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, mirte.k_matrix, mirte.d_matrix)
        tvecs = [[tvec[0][2], -tvec[0][0], -tvec[0][1]] for tvec in tvecs]
        for i in range(len(ids)):
            points.append([tvecs[i][0], tvecs[i][1], 0])
            if ids[i] in [0,1,2,3]:
                colors.append([0,255,0,255])
            else:
                colors.append([255,0,0,255])
            distance = np.linalg.norm(tvecs[i])
            angle = np.arctan2(tvecs[i][1], tvecs[i][0])
            distances.append(distance)
            angles.append(angle)
    return ids, distances, angles, points, colors

def process_lidar_boxes(lidar, lidar_positions):
    lidar_dists, lidar_angles = lidar_positions(lidar)

    if lidar_dists is None:
        return None, None, None, None

    points = []
    colors = []
    for i in range(len(lidar_dists)):
        angle = lidar_angles[i]
        distance = lidar_dists[i]
        x = distance * -np.cos(angle)
        y = distance * -np.sin(angle)
        points.append([x, y, 0])
        colors.append([255, 255, 0, 255])
    return lidar_dists, lidar_angles, points, colors

def update_occupancy_map_and_markers(lidar_dists, lidar_angles, box_size, occ_map):
    marker_list = []
    for i in range(len(lidar_dists)):
        angle = lidar_angles[i]
        distance = lidar_dists[i]
        marker_x = -distance * np.cos(angle)
        marker_y = distance * np.sin(angle)
        marker_list.append([marker_x, marker_y])
    occ_map.populate(marker_list, [box_size] * len(marker_list))
    return marker_list

def calculate_goal_and_path(estimate, goal_global_position, path_find):
    goal_position_vector = np.array(goal_global_position) - np.array(estimate.position)
    goal_position_angle = estimate.theta - np.arctan2(goal_position_vector[1], goal_position_vector[0])
    goal_position_x = np.cos(goal_position_angle) * np.linalg.norm(goal_position_vector)
    goal_position_y = np.sin(goal_position_angle) * np.linalg.norm(goal_position_vector)
    edges = []
    colours = []
    path = path_find.do_rrt((0,0), (goal_position_x, goal_position_y))
    if path is not None:
        for i in range(len(path) - 1):
            edges.append(((path[i][0],-path[i][1]), (path[i + 1][0], -path[i + 1][1])))
            colours.append((255, 0, 0, 255))
    return edges, colours, path

def draw_canvas(canvas, occ_map, path_find, path, window):
    canvas.delete("all")
    occ_map.canvas_draw(canvas)
    path_find.canvas_draw(path, canvas, low=(-6, -6), high=(6, 6))
    canvas.pack()
    window.update()

def path_to_instructions(path, init_angle):
    current_angle = init_angle
    instructions = []
    for i in range(len(path) - 1):
        target_angle = np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
        angle_diff = target_angle - current_angle
        current_angle = target_angle
        instructions.append((angle_diff, math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])))
    return instructions

instructions = None

def update_sensors():
    image = mirte.get_image()
    lidar = mirte.get_lidar_ranges()
    ids, distances, angles, aruco_points, aruco_colors = process_aruco_markers(
        image, aruco_dict, parameters, mirte, aruco_size
    )
    lidar_dists, lidar_angles, lidar_points, lidar_colors = process_lidar_boxes(
        lidar, lidar_positions
    )

    # Combine and visualize
    points = aruco_points + lidar_points
    colors = aruco_colors + lidar_colors
    mirte.set_pointcloud('mirte', points, colors)

    return ids, distances, angles, lidar, lidar_dists, lidar_angles


def update_localization(ids, distances, angles, lidar):
    selflocalizer.update_image(ids, distances, angles, lidar)
    selflocalizer.particle_filter.resample_particles()
    
    # Visualize particles
    points, colors = [], []
    for p in selflocalizer.particle_filter.particles.positions:
        points.append([p[0], p[1], 0])
    for w in selflocalizer.particle_filter.particles.weights:
        colors.append([0, 0, 255, min(255, 255 * w * selflocalizer.particle_filter.particles.num_particles)])
    
    estimate = selflocalizer.particle_filter.estimate_pose()
    points.append([estimate.x, estimate.y, 0])
    colors.append([0, 255, 255, 255])
    mirte.set_pointcloud('world', points, colors)

    return estimate


def turn_in_place_until_localized():
    mirte.drive(0, 0.5, 4 * math.pi, blocking=False)
    while selflocalizer.particle_filter.variance > 0.0002 and mirte.is_driving:
        iteration_time = time.time()
        time.sleep(0.1)
        ids, distances, angles, lidar, _, _ = update_sensors()
        selflocalizer.random_positions(1)
        update_localization(ids, distances, angles, lidar)
        delta_time = time.time() - iteration_time
        selflocalizer.update_turn(direction * delta_time * turn_speed)
    mirte.stop()


def update_path_and_instructions(estimate):
    edges, colours, path = calculate_goal_and_path(estimate, goal_order[0], path_find)
    mirte.set_tree('mirte', edges, colours)
    return path_to_instructions(path, 0) if path else None


def drive_instruction():
    global iteration_time, direction
    angle_instr, distance_instr = instructions[0]
    direction = -1 if angle_instr > 0 else 1
    mirte.drive(0, direction * 0.5, abs(angle_instr) * 2, blocking=False)

    while mirte.is_driving:
        iteration_time = time.time()
        time.sleep(0.1)
        ids, distances, angles, lidar, _, _ = update_sensors()
        update_localization(ids, distances, angles, lidar)
        selflocalizer.update_turn(direction * (time.time() - iteration_time) * turn_speed)

    mirte.drive(0.5, 0, distance_instr * 5, blocking=False)
    iteration_time = time.time()


# Main control loop
while cv2.waitKey(4) == -1:

    ids, distances, angles, lidar, lidar_dists, lidar_angles = update_sensors()
    estimate = update_localization(ids, distances, angles, lidar)

    # Driving update
    if mirte.is_driving:
        delta_time = time.time() - iteration_time
        selflocalizer.update_drive(delta_time * drive_speed)
        iteration_time = time.time()

    # Check goal
    print(f"Variance: {selflocalizer.particle_filter.variance:.4f}")
    if selflocalizer.particle_filter.variance < 0.0002:
        if np.linalg.norm(np.array(estimate.position) - np.array(goal_order[0])) < goal_margin:
            print("\n!!!!!!!!!!!!!!!At goal!!!!!!!!!!!!!!!!!!!!!\n")
            goal_order.pop(0)
            if not goal_order:
                print("All goals reached")
                break
            print(f"New goal: {goal_order[0]}")
            instructions = None

    time.sleep(0.1)

    if instructions:
        if not mirte.is_driving:
            if selflocalizer.particle_filter.variance < 0.0001:
                print("Low variance, updating occupancy map and path")
                update_occupancy_map_and_markers(lidar_dists, lidar_angles, box_size, occ_map)
                instructions = update_path_and_instructions(estimate) or instructions[1:]

            elif selflocalizer.particle_filter.variance < 0.0002:
                print("High variance, keeping the current plan")
                instructions.pop(0)

            else:
                print("Too high variance, turning in place")
                turn_in_place_until_localized()
                lidar = mirte.get_lidar_ranges()
                lidar_dists, lidar_angles, _, _ = process_lidar_boxes(lidar, lidar_positions)
                instructions = update_path_and_instructions(estimate)
        else:
            print("Driving, keeping the current plan")
    else:
        print("Path does not exist")
        update_occupancy_map_and_markers(lidar_dists, lidar_angles, box_size, occ_map)
        instructions = update_path_and_instructions(estimate)

    if not mirte.is_driving and instructions:
        drive_instruction()
    

#cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node