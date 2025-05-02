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

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 500
PATH_FOUND = False

aruco_size = 0.25 # m
box_size = 0.4 # m

mirte = KU_Mirte()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

selflocalizer = SelfLocalizer(2000, [[0,0], [0,4], [4,0], [4,4]], [0,1,2,3], bounds=(-4, -4, 8, 8))
selflocalizer.particle_filter.weights_used = "aruco"



# Create GUI canvas
window = tk.Tk()
window.title("Path Planning Visualization")
window.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}")
canvas = tk.Canvas(window, height=CANVAS_HEIGHT, width=CANVAS_WIDTH, bg="#F9ECCC")
canvas.pack()

class PointMassModel():
    # Note Arlo is differential driven and may be simpler to avoid Dubins car model by rotating in-place to direct and executing piecewise straight path
    def __init__(self, ctrl_range) -> None:
        # record the range of action that can be used to integrate the state
        self.ctrl_range = ctrl_range
    
    def forward_dyn(self, x, u, T):
        path = [x]
        # note u must have T ctrl to apply
        for i in range(T):
            x_new = path[-1] + u[i]  # u is velocity command here
            path.append(x_new)

        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        # for point mass, the path is just a straight line by taking full ctrl_range at each step

        if type(x_goal) == list:
            x = np.array(x)
            x_goal = np.array(x_goal)

        dir = (x_goal - x) / np.linalg.norm(x_goal - x)
        u = np.array([dir * self.ctrl_range[1] for _ in range(T)])

        return self.forward_dyn(x, u, T)


class GridOccupancyMap(object):
    """ """

    def __init__(self, low=(0, 0), high=(2, 2), res=0.05) -> None:
        self.map_area = [low, high]  # a rectangular area
        self.map_size = np.array([high[0] - low[0], high[1] - low[1]])
        self.resolution = res

        self.n_grids = [int(s // res) for s in self.map_size]

        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

        self.extent = [
            self.map_area[0][0],
            self.map_area[1][0],
            self.map_area[0][1],
            self.map_area[1][1],
        ]

    def in_collision(self, pos):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        indices = [
            int((pos[i] - self.map_area[0][i]) // self.resolution) for i in range(2)
        ]
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return 1

        return self.grid[indices[0], indices[1]]

    def populate(self, origins=[None], radius=[25]):
        """
        generate a grid map with some circle shaped obstacles
        """

        # fill the grids by checking if the grid centroid is in any of the circle
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array(
                    [
                        self.map_area[0][0] + self.resolution * (i + 0.5),
                        self.map_area[0][1] + self.resolution * (j + 0.5),
                    ]
                )
                for o, r in zip(origins, radius):
                    if np.linalg.norm(centroid - o) <= r:
                        self.grid[i, j] = 1
                        break


class RRT:
    class Node:
        """Class representing a node in the RRT graph."""

        def __init__(self, pos):
            self.pos = pos                # Node position (e.g., 2D coordinates)
            self.path = []               # List of intermediate positions from parent to this node
            self.parent = None           # Reference to parent node

        def calc_distance_to(self, other):
            """Calculate Euclidean distance to another node."""
            return np.linalg.norm(np.array(self.pos) - np.array(other.pos))

    def __init__(
        self,
        start,
        goal,
        robot_model,
        map,
        canvas,
        expand_dis=0.2,
        path_resolution=0.05,
        goal_sample_rate=5,
        max_iter=500,
    ):
        """
        Initialize the RRT planner.

        Parameters:
            start (list): Start position
            goal (list): Goal position
            robot_model: Robot model with inverse_dyn method
            map: Map object with in_collision method
            canvas: Canvas for drawing
            expand_dis (float): Maximum distance to expand tree
            path_resolution (float): Path resolution for edge creation
            goal_sample_rate (int): Goal sampling rate in percentage
            max_iter (int): Maximum iterations to try
        """
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.robot = robot_model
        self.map = map
        self.canvas = canvas

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        self.min_rand = map.map_area[0]
        self.max_rand = map.map_area[1]
        self.node_list = [self.start]

    def planning(self):
        """Perform RRT path planning. Returns the path if found, else None."""

        def angle_between(p1, p2):
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if (
                self.check_collision_free(new_node)
                and 220 < new_node.pos[1] < 300
                and abs(angle_between(nearest_node.pos, new_node.pos)) < 2 * np.pi / 3
            ):
                self.node_list.append(new_node)
            else:
                self.draw_node(new_node.pos, color="red")

            # Check if path to goal is feasible
            if self.end_close(new_node):
                return self.generate_final_course(len(self.node_list) - 1)

            # Attempt to connect directly to goal
            if new_node.calc_distance_to(self.end) <= self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if self.check_collision_free(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Generate a new node by moving from one node towards another."""
        new_node = self.Node(from_node.pos)
        d = new_node.calc_distance_to(to_node)
        extend_length = min(extend_length, d)
        n_expand = int(extend_length // self.path_resolution)

        if n_expand > 0:
            path = self.robot.inverse_dyn(from_node.pos, to_node.pos, n_expand)
            new_node.pos = path[-1]
            new_node.path = path

        if new_node.calc_distance_to(to_node) <= self.path_resolution:
            new_node.path.append(to_node.pos)
            new_node.pos = to_node.pos.copy()

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_index):
        """Backtrack from goal to start to build the path."""
        path = [self.end.pos]
        node = self.node_list[goal_index]
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return path

    def get_random_node(self):
        """Sample a random node, with some probability of returning the goal."""
        if np.random.randint(0, 100) > self.goal_sample_rate:
            return self.Node(np.random.uniform(self.min_rand, self.max_rand))
        return self.Node(self.end.pos)

    def get_nearest_node_index(self, rnd_node):
        """Return the index of the nearest node to a given node."""
        return min(range(len(self.node_list)), key=lambda i: self.node_list[i].calc_distance_to(rnd_node))

    def check_collision_free(self, node):
        """Check that all points in a node's path are collision-free."""
        return node is not None and all(not self.map.in_collision(np.array(p)) for p in node.path)

    def end_close(self, node):
        """Determine if a node is close enough to the goal to consider connection."""
        radius = node.calc_distance_to(self.end) - 30
        return any(np.linalg.norm(np.array(p) - np.array(self.end.pos)) < radius for p in node.path)

    def draw_graph(self):
        """Draw the current RRT graph on the canvas."""
        for node in self.node_list:
            if node.parent:
                path = np.array(node.path)
                for i in range(1, len(path)):
                    self.canvas.create_line(
                        path[i - 1][1], path[i - 1][0], path[i][1], path[i][0], fill="black"
                    )
        print(f"Drawing {len(self.node_list)} nodes")

    def draw_node(self, pos, color="red"):
        """Draw a single node as a colored oval on the canvas."""
        self.canvas.create_oval(
            pos[1] - 5, pos[0] - 5, pos[1] + 5, pos[0] + 5, fill=color
        )
        self.canvas.pack()

def circle_cord(x, y, r):
    return x - r, y - r, x + r, y + r

def is_collided(x1, y1, x2, y2, r):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < r

def angle_between_nodes(a, b):
    return np.arctan2(b[1] - a[1], b[0] - a[0])

def distance_between_nodes(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def path_to_instructions(path, init_angle):
    current_angle = init_angle
    angle_instr, distance_instr = [], []
    for i in range(len(path) - 1):
        target_angle = angle_between_nodes(path[i], path[i + 1])
        angle_diff = target_angle - current_angle
        angle_instr.append(angle_diff)
        current_angle = target_angle
        distance_instr.append(distance_between_nodes(path[i], path[i + 1]))
    return angle_instr, distance_instr

def do_instructions_on_arlo(angle_instr, distance_instr):
    for angle, dist in zip(angle_instr, distance_instr):
        dist = (dist / 100) * 2
        print(f"Turning {angle:.2f} radians")
        direction = 1 if angle > 0 else -1
        mirte.drive(0,  direction * 0.2, abs(angle) * 5, blocking=True)
        print(f"Moving {dist:.2f} units")
        mirte.drive(0.2, 0, dist * 5, blocking=True)

def do_rrt(marker_positions, start_pos, goal_pos):
    global PATH_FOUND
    path_res = 30

    map = GridOccupancyMap(low=(0, 0), high=(CANVAS_WIDTH, CANVAS_HEIGHT), res=path_res)
    print("Occupancy map created")

    # Flip x and y for marker positions
    marker_positions[:, 0], marker_positions[:, 1] = marker_positions[:, 1], marker_positions[:, 0].copy()

    map.populate(origins=marker_positions, radius=[50] * len(marker_positions))
    print("Map populated with obstacles")

    robot = PointMassModel(ctrl_range=[-path_res, path_res])
    rrt = RRT(
        start=start_pos,
        goal=goal_pos,
        robot_model=robot,
        map=map,
        canvas=canvas,
        expand_dis=80,
        path_resolution=path_res,
        max_iter=100,
    )

    path = rrt.planning()
    canvas.create_oval(circle_cord(250, 300, 10), fill="#587D71")
    rrt.draw_graph()

    if path is None:
        print("Path not found")
        return None
    else:
        PATH_FOUND = True
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            canvas.create_line(y1, x1, y2, x2, width=5, fill="#754668")
        return path



def lidar_positions(measured_lidar: np.ndarray, lidar_diff_theshold: float = 5, lidar_max_dist: float = 20, object_size: tuple = (0.2,0.3)) -> np.ndarray:
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
    selflocalizer.random_positions(5)

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

    window.update()

    if PATH_FOUND:
        PATH_FOUND = False
        input("Press Enter to take a picture...")
    
    

cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node