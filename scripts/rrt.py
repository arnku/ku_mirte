import numpy as np
import math

class PointMassModel:
    """
    Simple point mass model for robot motion.
    For Arlo (differential drive), rotates in-place and executes piecewise straight paths.
    """
    def __init__(self, ctrl_range):
        """
        Initialize the model with control range.
        Initialize the model with control range.
        ctrl_range: [min, max] velocity command.
        """
        self.ctrl_range = ctrl_range

    def forward_dyn(self, x, u, T):
        """
        Simulate forward dynamics for T steps.
        x: initial state (np.array)
        u: list of velocity commands (np.array)
        T: number of steps
        Returns: list of states
        """
        path = [np.array(x)]
        for i in range(T):
            x_new = path[-1] + u[i]
            path.append(x_new)
        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        """
        Generate a straight line path from x to x_goal in T steps.
        Returns: list of states along the path.
        """
        x = np.array(x)
        x_goal = np.array(x_goal)
        direction = (x_goal - x) / np.linalg.norm(x_goal - x)
        u = np.array([direction * self.ctrl_range[1]] * T)
        return self.forward_dyn(x, u, T)

class GridOccupancyMap:
    """
    2D grid occupancy map for collision checking.
    """
    def __init__(self, low, high, res):
        """
        low: lower bound of map (tuple)
        high: upper bound of map (tuple)
        res: resolution of the map
        """
        self.map_area = [low, high]
        self.map_size = np.array([high[0] - low[0], high[1] - low[1]])
        self.resolution = res
        self.n_grids = [int(s // res) for s in self.map_size]
        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

    def in_collision(self, pos):
        """
        Check if the position is occupied or outside the map.
        Returns True if in collision or out of bounds, else False.
        """
        indices = [
            int((pos[0] - self.map_area[0][0]) // self.resolution),
            int((pos[1] - self.map_area[0][1]) // self.resolution),
        ]
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return True
        return self.grid[indices[0], indices[1]] == 1

    def populate(self, origins, radii):
        """
        Populate the map with circular obstacles.
        origins: list of circle centers
        radii: list of radii
        """
        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8) # Reset grid
        
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array([
                    self.map_area[0][0] + self.resolution * (i + 0.5),
                    self.map_area[0][1] + self.resolution * (j + 0.5),
                ])
                for o, r in zip(origins, radii):
                    if np.linalg.norm(centroid - o) <= r:
                        self.grid[i, j] = 1
                        break
    
    def canvas_draw(self, canvas):
        """
        Draw the occupancy map on the canvas.
        """
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                if self.grid[i, j] == 1:
                    x1 = int(self.map_area[0][0] + self.resolution * (i + 0.5))
                    y1 = int(self.map_area[0][1] + self.resolution * (j + 0.5))
                    x2 = int(x1 + self.resolution)
                    y2 = int(y1 + self.resolution)
                    canvas.create_rectangle(x1, y1, x2, y2, fill="black")

class RRT:
    """
    Rapidly-exploring Random Tree (RRT) planner.
    """
    def __init__(self, robot_model, occ_map, expand_dis, path_resolution, max_iter=200):
        self.robot = robot_model
        self.map = occ_map
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        
        self.start = None
        self.end = None
        self.node_list = None
    
    class Node:
        """Node in the RRT graph."""
        def __init__(self, pos):
            self.pos = np.array(pos)
            self.path = []
            self.parent = None

        def calc_distance_to(self, other):
            return np.linalg.norm(self.pos - np.array(other.pos))


    def get_random_node(self):
        """
        Sample a random node
        """
        return self.Node(np.random.uniform(self.map.map_area[0], self.map.map_area[1]))

    def get_nearest_node(self, node):
        """
        Return the index of the nearest node to a given node.
        """
        return self.node_list[min(range(len(self.node_list)), key=lambda i: self.node_list[i].calc_distance_to(node))]

    def check_collision_free(self, node):
        """
        Check that all points in a node's path are collision-free.
        """
        if node is None:
            raise ValueError("Node is None")
        return all(not self.map.in_collision(p) for p in node.path)

    def is_goal_reached(self, node):
        """
        Determine if a node is close enough to the goal.
        """
        return node.calc_distance_to(self.end) < self.expand_dis
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Move from from_node towards to_node by at most extend_length.
        Returns: new node.
        """
        # How far to extend
        new_node = self.Node(from_node.pos)
        extend_length = min(extend_length, new_node.calc_distance_to(to_node))
        n_expand = int(extend_length // self.path_resolution)
        
        # Put the new node at the end of the path towards to_node
        if n_expand > 0:
            path = self.robot.inverse_dyn(from_node.pos, to_node.pos, n_expand)
            new_node.pos = path[-1] # last point
            new_node.path = path
        
        # If the new node is close enough to to_node, add it to the path
        if new_node.calc_distance_to(to_node) <= self.path_resolution:
            new_node.path.append(to_node.pos)
            new_node.pos = np.array(to_node.pos)
        
        new_node.parent = from_node
        return new_node
    
    def generate_final_course(self, goal_index):
        """
        Backtrack from goal to start to build the path.
        Returns: list of positions from start to goal.
        """
        path = [self.end.pos]
        node = self.node_list[goal_index]
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return path[::-1]
    
    def do_rrt(self, start_pos, goal_pos):
        """
        Perform RRT* path planning.
        Returns the path if found, else None.
        """

        self.start = self.Node(start_pos)
        self.end = self.Node(goal_pos)
        self.node_list = [self.start]

        for _ in range(self.max_iter):
            # Try to reach the goal
            end_node = self.Node(self.end.pos)
            nearest_end_node = self.get_nearest_node(end_node) # nearest node to random location
            new_node = self.steer(nearest_end_node, end_node, extend_length=self.expand_dis) # Line from nearest node towards random location
            
            if not self.check_collision_free(new_node):    
                # Sample a random node
                rnd_node = self.get_random_node() # random location
                nearest_node = self.get_nearest_node(rnd_node) # nearest node to random location
                new_node = self.steer(nearest_node, rnd_node, self.expand_dis) # Line from nearest node towards random location

            if self.check_collision_free(new_node):
                self.node_list.append(new_node)

            if self.is_goal_reached(new_node):
                return self.generate_final_course(len(self.node_list) - 1)
        return None

    def canvas_draw(self, path, canvas, point_size=5, line_width=2):
        """

        """
        # Draw start and goal positions
        canvas.create_oval((self.start.pos[0] - point_size, self.start.pos[1] - point_size, self.start.pos[0] + point_size), self.start.pos[1] + point_size, fill="green")
        canvas.create_oval((self.end.pos[0] - point_size, self.end.pos[1] - point_size, self.end.pos[0] + point_size), self.end.pos[1] + point_size, fill="blue")
        
        # Draw the tree
        for node in self.node_list:
            if node.parent:
                tree = np.array([node.parent.pos] + node.path)
                for i in range(1, len(tree)):
                    canvas.create_line(
                        tree[i - 1][0], tree[i - 1][1], tree[i][0], tree[i][1], 
                        width=line_width,
                        fill="black"
                    )
        print(f"Drawing {len(self.node_list)} nodes")

        # Draw the path
        if path is not None:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                canvas.create_line(x1, y1, x2, y2, width=line_width, fill="green")


if __name__ == "__main__":
    # Example usage
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 600
    # Example marker positions (obstacles)
    marker_positions = np.array([[100, 200], [200, 200], [400, 300]])
    marker_radius = 50
    
    # Start and goal positions
    start_pos = [300, 100]
    goal_pos = [500, 500]

    path_res = 10 # Resolution of the grid

    # Create a Tkinter canvas for visualization
    import tkinter as tk
    window = tk.Tk()
    canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    
    robot = PointMassModel(ctrl_range=[-path_res, path_res])

    occ_map = GridOccupancyMap(low=(0, 0), high=(CANVAS_WIDTH, CANVAS_HEIGHT), res=path_res)
    occ_map.populate(origins=marker_positions, radii=[marker_radius] * len(marker_positions))
    occ_map.canvas_draw(canvas)

    rrt = RRT(robot, occ_map, expand_dis=3 * path_res, path_resolution=path_res, max_iter=100)
    path = rrt.do_rrt(start_pos, goal_pos)
    rrt.canvas_draw(path, canvas)

    # Run RRT
    canvas.pack()
    input("Press Enter to exit...")
    
    #window.mainloop()
    
    