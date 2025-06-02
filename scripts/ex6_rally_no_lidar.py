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

aruco_size = 0.15 # m
path_res = 0.05
goal_order = [[2.6,2.6], [0,2.3], [2.5,0], [-0.1,-0.1]]
landmarks = [[0,0], [0,2.6], [2.6,0], [2.6,2.6]]
landmark_ids = [10, 2, 6, 3] # Same order as landmarks
iteration_time = time.time()
drive_speed = 0.3 # m/s
turn_speed = 0.5 # rad/s
goal_margin = 0.8 # m

mirte = KU_Mirte()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

selflocalizer = SelfLocalizer(75, landmarks, landmark_ids, bounds=(0, 0, 2.5, 2.5), starting_area=(0, 0, 0.5, 0.5))
selflocalizer.particle_filter.weights_used = "aruco"

robot = rrt.PointMassModel(ctrl_range=[-path_res, path_res])
occ_map = rrt.GridOccupancyMap(low=(-6, -6), high=(6, 6), res=path_res)
path_find = rrt.RRT(robot, occ_map, expand_dis=6*path_res, path_resolution=path_res, goal_radius=1.0, max_iter=200)


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
            #points.append([tvecs[i][0], tvecs[i][1], 0])
            #if ids[i] in [0,1,2,3]:
            #    colors.append([0,255,0,255])
            #else:
            #    colors.append([255,0,0,255])
            distance = np.linalg.norm(tvecs[i])
            angle = np.arctan2(tvecs[i][1], tvecs[i][0])
            distances.append(distance)
            angles.append(angle)
    print(f"Detected IDs: {ids}")
    return ids, distances, angles, points, colors

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    ids, distances, angles, aruco_points, aruco_colors = process_aruco_markers(
        image, aruco_dict, parameters, mirte, aruco_size
    )

    # Combine and visualize
    #points = aruco_points 
    #colors = aruco_colors 
    #mirte.set_pointcloud('mirte', points, colors)

    return ids, distances, angles


def update_localization(ids, distances, angles):
    lidar = None
    selflocalizer.update_image(ids, distances, angles, lidar)
    selflocalizer.particle_filter.resample_particles()
    
    # Visualize particles
    #points, colors = [], []
    #for p in selflocalizer.particle_filter.particles.positions:
    #    points.append([p[0], p[1], 0])
    #for w in selflocalizer.particle_filter.particles.weights:
    #    colors.append([0, 0, 255, min(255, 255 * w * selflocalizer.particle_filter.particles.num_particles)])
    
    estimate = selflocalizer.particle_filter.estimate_pose()
    #points.append([estimate.x, estimate.y, 0])
    #colors.append([0, 255, 255, 255])
    #mirte.set_pointcloud('world', points, colors)

    return estimate


def turn_in_place_until_localized():
    mirte.drive(0, 0.5, 4 * math.pi, blocking=False)
    while selflocalizer.particle_filter.variance > 0.0002 and mirte.is_driving:
        turn_iteration_time = time.time()
        time.sleep(0.1)
        ids, distances, angles = update_sensors()
        selflocalizer.random_positions(1)
        update_localization(ids, distances, angles)
        delta_time = time.time() - turn_iteration_time
        selflocalizer.update_turn(direction * delta_time * turn_speed)
    mirte.stop()
    global iteration_time
    iteration_time = time.time()


def update_path_and_instructions(estimate):
    edges, colours, path = calculate_goal_and_path(estimate, goal_order[0], path_find)
    #mirte.set_tree('mirte', edges, colours)
    return path_to_instructions(path, 0) if path else None


def drive_instruction():
    global iteration_time, direction
    angle_instr, distance_instr = instructions[0]
    print(f"Driving instruction: angle={angle_instr}, distance={distance_instr}")
    direction = -1 if angle_instr > 0 else 1
    mirte.drive(0, direction * turn_speed, abs(angle_instr) / turn_speed, blocking=False)

    time_delta = time.time() 

    while mirte.is_driving:
        turn_iteration_time = time.time()
        time.sleep(0.1)
        ids, distances, angles = update_sensors()
        update_localization(ids, distances, angles)
        selflocalizer.update_turn(direction * (time.time() - turn_iteration_time) * turn_speed)

    time_delta = time.time() - time_delta
    print(f"Turn completed in {time_delta:.2f} seconds")

    iteration_time += time_delta

    mirte.drive(drive_speed, 0, distance_instr / drive_speed, blocking=False)

for _ in range(3):
    ids, distances, angles = update_sensors()
    print(f"Detected IDs: {ids} and distances: {distances}")
    estimate = update_localization(ids, distances, angles)

# Main control loop
while cv2.waitKey(4) == -1:
    print("")
    ids, distances, angles = update_sensors()
    print(f"Detected IDs: {ids} and distances: {distances}")
    estimate = update_localization(ids, distances, angles)
    print(f"Estimate: {estimate.position}")
    print("Distance from goal:", np.linalg.norm(np.array(estimate.position) - np.array(goal_order[0])))



    # Check goal
    #print(f"Variance: {selflocalizer.particle_filter.variance:.4f}")
    if selflocalizer.particle_filter.variance < 0.0002:
        if np.linalg.norm(np.array(estimate.position) - np.array(goal_order[0])) < goal_margin:
            print(f"Estimate: {estimate.position}")
            print("\n!!!!!!!!!!!!!!!At goal!!!!!!!!!!!!!!!!!!!!!\n")
            print(f"distance from goal: {np.linalg.norm(np.array(estimate.position) - np.array(goal_order[0]))}")
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
                instructions = update_path_and_instructions(estimate) or instructions[1:]

            elif selflocalizer.particle_filter.variance < 0.001:
                print("High variance, keeping the current plan")
                instructions.pop(0)

            else:
                print("Too high variance, turning in place")
                turn_in_place_until_localized()
                instructions = update_path_and_instructions(estimate)
        else:
            print("Driving, keeping the current plan")
    else:
        print("Path does not exist")
        instructions = update_path_and_instructions(estimate)

    # Driving update
    print(f"Driving status: {mirte.is_driving}")
    if mirte.is_driving:
        print(time.time())
        delta_time = time.time() - iteration_time
        print(f"Delta time: {delta_time:.2f} seconds, moving particles {delta_time * drive_speed:.2f} m")
        selflocalizer.update_drive(delta_time * drive_speed, sigma_x=0.2, sigma_y=0.2, sigma_theta=0.01)

    iteration_time = time.time()

    if not mirte.is_driving and instructions:
        drive_instruction()
    

#cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node
