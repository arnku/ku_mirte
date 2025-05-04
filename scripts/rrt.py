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
    def __init__(self, low, high, res=0.05):
        """
        low: lower bound of map (tuple)
        high: upper bound of map (tuple)
        res: grid resolution
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
    class Node:
        """Node in the RRT graph."""
        def __init__(self, pos):
            self.pos = np.array(pos)
            self.path = []
            self.parent = None

        def calc_distance_to(self, other):
            return np.linalg.norm(self.pos - np.array(other.pos))

    def __init__(self, start, goal, robot_model, occ_map, canvas, expand_dis=0.2, path_resolution=0.05, goal_sample_rate=5, max_iter=500):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.robot = robot_model
        self.map = occ_map
        self.canvas = canvas
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        
        self.min_rand = occ_map.map_area[0]
        self.max_rand = occ_map.map_area[1]
        self.node_list = [self.start]

    def get_random_node(self, goal_sample_rate):
        """
        Sample a random node, with some probability of returning the goal.
        """
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rand = np.random.uniform(self.min_rand, self.max_rand)
            return self.Node(rand)
        return self.Node(self.end.pos)

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
            print("Node is None")
            return False
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
    
    def planning(self):
        """
        Perform RRT path planning.
        Returns the path if found, else None.
        """
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node(self.goal_sample_rate) # random location
            nearest_node = self.get_nearest_node(rnd_node) # nearest node to random location
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis) # Line from nearest node towards random location

            if self.check_collision_free(new_node):
                self.node_list.append(new_node)
            else:
                self.draw_node(new_node.pos, color="red")

            if self.is_goal_reached(new_node):
                return self.generate_final_course(len(self.node_list) - 1)
        return None


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


    def draw_graph(self):
        """
        Draw the current RRT graph on the canvas.
        """
        for node in self.node_list:
            if node.parent:
                path = np.array(node.path)
                for i in range(1, len(path)):
                    self.canvas.create_line(
                        path[i - 1][1], path[i - 1][0], path[i][1], path[i][0], fill="black"
                    )
        print(f"Drawing {len(self.node_list)} nodes")

    def draw_node(self, pos, color="red"):
        """
        Draw a single node as a colored oval on the canvas.
        """
        self.canvas.create_oval(
            pos[1] - 5, pos[0] - 5, pos[1] + 5, pos[0] + 5, fill=color
        )
        self.canvas.pack()

# --- Utility functions ---

def circle_cord(x, y, r):
    """Return bounding box coordinates for a circle."""
    return x - r, y - r, x + r, y + r

def do_rrt(marker_positions, start_pos, goal_pos):
    """
    Run RRT planner and visualize the result.
    marker_positions: numpy array of obstacle centers
    start_pos: start position (x, y)
    goal_pos: goal position (x, y)
    Returns: path if found, else None
    """
    path_res = 10

    occ_map = GridOccupancyMap(low=(0, 0), high=(CANVAS_WIDTH, CANVAS_HEIGHT), res=path_res)
    print("Occupancy map created")

    # Flip x and y for marker positions
    #rev_marker_positions[:, 0], rev_marker_positions[:, 1] = marker_positions[:, 1], marker_positions[:, 0]

    occ_map.populate(origins=marker_positions, radii=[50] * len(marker_positions))
    occ_map.canvas_draw(canvas)
    print("Map populated with obstacles")

    robot = PointMassModel(ctrl_range=[-path_res, path_res])
    rrt = RRT(
        start=start_pos,
        goal=goal_pos,
        robot_model=robot,
        occ_map=occ_map,
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
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            canvas.create_line(y1, x1, y2, x2, width=5, fill="#754668")
        return path

if __name__ == "__main__":
    # Example usage
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 600

    # Create a Tkinter canvas for visualization
    import tkinter as tk
    window = tk.Tk()
    canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    
    # Example marker positions (obstacles)
    marker_positions = np.array([[100, 200], [300, 400], [500, 100]])
    
    # Start and goal positions
    start_pos = [250, 300]
    goal_pos = [700, 500]
    
    # Draw the occupancy map
    occ_map = GridOccupancyMap(low=(0, 0), high=(CANVAS_WIDTH, CANVAS_HEIGHT), res=10)
    occ_map.populate(origins=marker_positions, radii=[50] * len(marker_positions))
    occ_map.canvas_draw(canvas)
    canvas.pack()
    input("Press Enter to continue...")
    canvas.delete("all")

    # Run RRT
    do_rrt(marker_positions, start_pos, goal_pos)
    input("Press Enter to exit...")
    
    #window.mainloop()
    
    