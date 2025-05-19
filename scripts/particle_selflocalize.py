import numpy as np
from copy import deepcopy
from scipy.stats import norm

class Particle:
    def __init__(self):
        self.x : float = 0
        self.y : float = 0
        self._theta : float = 0
        self.weight : float = 0

    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, value):
        self._theta = np.mod(value, 2.0 * np.pi)  # Ensure theta is within [0, 2π)
    
    @property
    def position(self):
        return np.array([self.x, self.y])
    
    @position.setter
    def position(self, value):
        self.x, self.y = value
    
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}, weight={self.weight})"

class Particles:
    def __init__(self, num_particles: int):
        self.particles = [Particle() for _ in range(num_particles)]
        self.num_particles = num_particles

    @property
    def weights(self):
        return np.array([p.weight for p in self.particles])
    @weights.setter
    def weights(self, value : np.ndarray):
        # Normalize the weights
        value /= np.sum(value)
        # Set the weights for each particle
        for p, w in zip(self.particles, value):
            p.weight = w
    
    @property
    def positions(self):
        return np.array([[p.x, p.y] for p in self.particles])
    @positions.setter
    def positions(self, postitions : np.ndarray):
        # Set the positions for each particle
        for p, pos in zip(self.particles, postitions):
            p.x, p.y = pos
    
    @property
    def angles(self):
        return np.array([p.theta for p in self.particles])
    @angles.setter
    def angles(self, angles : np.ndarray):
        # Set the angles for each particle
        for p, angle in zip(self.particles, angles):
            p.theta = angle

    def rearrange(self, indices: np.ndarray):
        """
        Rearrange particles based on the given indices.
        This is useful for resampling particles.
        """
        self.particles = [deepcopy(self.particles[i]) for i in indices]
        self.num_particles = len(self.particles)

    def __len__(self):
        self.num_particles = len(self.particles)
        return self.num_particles

    def __iter__(self):
        return iter(self.particles)

class ParticleFilter:
    def __init__(self, area: tuple, num_particles: int = 100, starting_area: tuple = None):
        self.num_particles = num_particles
        self.particles = Particles(num_particles)
        
        # Initialize particles uniformly in a square area
        self.area = area
        self.particles.positions = np.random.uniform(low=[area[0], area[1]], high=[area[2], area[3]], size=(num_particles, 2))
        self.particles.angles = np.random.uniform(low=0, high=2 * np.pi, size=num_particles)
        if starting_area is not None:
            self.particles.positions = np.random.uniform(low=[starting_area[0], starting_area[1]], high=[starting_area[2], starting_area[3]], size=(num_particles, 2))

        self.weights_used = "all"

    @property
    def variance(self) -> float:
        """
        Calculate the variance of the particles' positions.
        """
        return np.prod(np.var(self.particles.positions, axis=0))

    def _precompute_landmarks(self, landmarks: np.ndarray) -> tuple:
        """
        Precompute the distances and angles to landmarks for each particle.
        This is a helper function to avoid redundant calculations.
        """
        # Calculate the unit vector to each landmark from each particle
        diff = (landmarks[np.newaxis, :, :] - self.particles.positions[:, np.newaxis, :]) # Shape of diff: [n_particles, n_landmarks, 2]
        dists = np.linalg.norm(diff, axis=2)  # Shape: [n_particles, n_landmarks]
        e_ls = (diff / dists[..., np.newaxis])  # Unit vectors to landmarks, shape: [n_particles, n_landmarks, 2]

        # Calculate the orientation vectors and their orthogonals for each particle
        e_thetas = np.stack([np.cos(self.particles.angles), np.sin(self.particles.angles)], axis=1)  # Shape: [n_particles, 2]
        e_ortho_thetas = np.stack([-e_thetas[:, 1], e_thetas[:, 0]], axis=1)  # Orthogonal vectors, shape: [n_particles, 2]

        # Calculate theoretical angles using the dot product
        dot_products = np.einsum("ijk,ik->ij", e_ls, e_ortho_thetas)  # Dot products, shape: [n_particles, n_landmarks]
        angle_signs = np.sign(dot_products)  # Shape: [n_particles, n_landmarks]
        angles = angle_signs * np.arccos(np.einsum("ijk,ik->ij", e_ls, e_thetas))  # Shape: [n_particles, n_landmarks]

        return dists, angles
        

    def position_probabilities(self, measured_dists: list, landmark_distances: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Input:
        - measured_dists: list of measured distances to landmarks [d1, d2, ...]
        - landmark_distances: 2D array of distances to landmarks [[d1_l1, d1_l2, ...], [d2_l1, d2_l2, ...], ...]
        - sigma: standard deviation of the distance measurement

        Output:
        - probs: 2D array of probabilities for each particle and landmark [[p1_l1, p1_l2, ...], [p2_l1, p2_l2, ...], ...]
        """
        measured_dists = np.array(measured_dists, dtype=np.float64)
        probs = norm.pdf(measured_dists, landmark_distances, sigma) # Calculate the probabilities (shape: [n_particles, n_landmarks])
        probs[:, np.isnan(measured_dists)] = 1 # Replace probabilities for non-measured distances (NaN) with 1

        return probs

    def angle_probabilities(self, measured_angles: dict, landmark_angles: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Input:
        - measured_angles: list of measured angles to landmarks [a1, a2, ...]
        - landmark_angles: 2D array of angles to landmarks [[a1_l1, a1_l2, ...], [a2_l1, a2_l2, ...], ...]
        - sigma: standard deviation of the angle measurement

        Output:
        - probs: 2D array of probabilities for each particle and landmark [[p1_l1, p1_l2, ...], [p2_l1, p2_l2, ...], ...]
        """
        measured_angles = np.array(measured_angles, dtype=np.float64)
        probs = norm.pdf(measured_angles, landmark_angles, sigma) # Calculate probabilities for each particle and landmark based on angles
        probs[:, np.isnan(measured_angles)] = 1 # Replace probabilities for non-measured angles (NaN) with 1

        return probs

    def lidar_probabilities(self, measured_lidar: np.ndarray, landmark_distances: np.ndarray, landmark_angles: np.ndarray, lidar_diff_theshold: float = 5, lidar_max_dist: float = 20, object_size: tuple = (0.2,0.3), sigma_dist: float = 0.2, sigma_angle: float = 1 ) -> np.ndarray:
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
            return np.ones(self.num_particles)
        
        # Get the distances and angles of the objects
        object_dists = min_segment_dists[box_indices]
        object_location = np.where(np.isin(measured_lidar, object_dists))[0]
        object_angles = np.deg2rad(object_location) # Here again the indicies can be directly used as angles
        
        # Filter the angles to only include those that are close to the object angles
        filtered_angles = np.array([
            row[np.argsort(np.min(np.abs(row[:, None] - object_angles[None, :]), axis=1))[:len(object_angles)]]
            for row in landmark_angles
        ])
        # Filter the distances 
        filtered_distances = np.array([
            row[np.argsort(np.min(np.abs(row[:, None] - object_dists[None, :]), axis=1))[:len(object_dists)]]
            for row in landmark_distances
        ])

        # Compute probabilities
        angle_probs = (norm.pdf(object_angles, filtered_angles, sigma_angle)) # TODO: Angle are acting strange
        dist_probs = (norm.pdf(object_dists, filtered_distances, sigma_dist))
        #combined_probs = angle_probs * dist_probs
        combined_probs = dist_probs
        combined_probs = np.prod(combined_probs, axis=1)
        
        return combined_probs
        
    
    def particle_weights(self, measured_dists:list, measured_angles:list, measured_lidar:list, landmarks:list) -> None:
        dists, angles = self._precompute_landmarks(landmarks) # Precompute distances and angles to landmarks for each particle

        if self.weights_used == "all":
            position_probs = self.position_probabilities(measured_dists, dists)
            angle_probs = self.angle_probabilities(measured_angles, angles)
            lidar_probs = self.lidar_probabilities(measured_lidar, dists, angles)
            
            aruco_angle_dist = np.prod([position_probs, angle_probs], axis=0) # Combine the probabilities for each particle and landmark
            aruco_angle_dist = np.prod(aruco_angle_dist, axis=1)
            combined_probs = np.prod([aruco_angle_dist, lidar_probs], axis=0) # Combine the probabilities with the lidar probabilities

            self.particles.weights = combined_probs
            return
        if self.weights_used == "lidar": 
            lidar_probs = self.lidar_probabilities(measured_lidar, dists, angles)
            self.particles.weights = lidar_probs
            return
        if self.weights_used == "aruco":
            position_probs = self.position_probabilities(measured_dists, dists)
            angle_probs = self.angle_probabilities(measured_angles, angles)
            aruco_angle_dist = np.prod([position_probs, angle_probs], axis=0)
            aruco_angle_dist = np.prod(aruco_angle_dist, axis=1)
            self.particles.weights = aruco_angle_dist
            return
        
    def resample_particles(self) -> None:
        normalized_weights = self.particles.weights / np.sum(self.particles.weights)
        new_particle_indecies = np.random.choice(self.num_particles, self.num_particles, p=normalized_weights)
        self.particles.rearrange(new_particle_indecies)
        
    def estimate_pose(self) -> Particle:
        x_avg, y_avg = np.mean(self.particles.positions, axis=0)
        theta_avg = np.mean(self.particles.angles)

        p = Particle()
        p.x = x_avg
        p.y = y_avg
        p.theta = theta_avg

        return p
    

class SelfLocalizer:
    def __init__(self, num_particles: int, landmarks: list, landmark_ids: list, bounds: tuple, starting_area: tuple = None):
        self.landmarks = np.array(landmarks)
        self.landmark_id_index = {landmark_id: i for i, landmark_id in enumerate(landmark_ids)}

        self.num_particles = num_particles
        self.bounds = bounds
        self.particle_filter = ParticleFilter(bounds, num_particles=num_particles, starting_area=starting_area)

    def update_drive(self, dist: float, sigma_x=0.1, sigma_y=0.1, sigma_theta=0.02) -> None:
        # Get angles of all particles as a vector

        # Calculate dx and dy vectors
        dx = dist * np.cos(self.particle_filter.particles.angles)
        dy = dist * np.sin(self.particle_filter.particles.angles)

        # Move each particle using vectorized dx, dy
        self.particle_filter.particles.positions += np.column_stack((dx, dy))

        self.add_uncertainty(abs(sigma_x * dist), abs(sigma_y * dist), abs(sigma_theta * dist))

    def update_turn(self, angle: float, sigma_x=0.001, sigma_y=0.001, sigma_theta=0.4) -> None:
        self.particle_filter.particles.angles += angle

        self.add_uncertainty(abs(sigma_x * angle), abs(sigma_y * angle), abs(sigma_theta * angle))

    def update_image(self, ids:list, distances: list, angles: list, lidars: list) -> None:
        distance_list = [None] * len(self.landmarks)
        angle_list = [None] * len(self.landmarks)
        if ids is not None:
            for distance, angle, landmark_id in zip(distances, angles, ids):
                if landmark_id in self.landmark_id_index:
                    distance_list[self.landmark_id_index[landmark_id]] = distance
                    angle_list[self.landmark_id_index[landmark_id]] = angle
                else:
                    #print(f"Landmark ID {landmark_id} not found in landmark index.")
                    continue
            
        self.particle_filter.particle_weights(distance_list, angle_list, lidars, self.landmarks)
        #self.particle_filter.resample_particles()

    def add_uncertainty(self, sigma_x=0.02, sigma_y=0.02, sigma_theta=0.3) -> None:
        # Create uncertainty arrays
        x_uncertainty = np.random.normal(0, sigma_x, self.num_particles)
        y_uncertainty = np.random.normal(0, sigma_y, self.num_particles)
        theta_uncertainty = np.random.normal(0, sigma_theta, self.num_particles) * np.random.choice([1, -1], self.num_particles)

        # Add uncertainty to each particle
        self.particle_filter.particles.positions += np.column_stack((x_uncertainty, y_uncertainty))
        self.particle_filter.particles.angles += theta_uncertainty
    
    def random_positions(self, num_particles: int) -> None:
        """
        Randomly move some particles within the bounds of the area.
        """
        # Generate random positions within the bounds
        random_positions = np.random.uniform(low=[self.bounds[0], self.bounds[1]], high=[self.bounds[2], self.bounds[3]], size=(num_particles, 2))

        # get the indices of the particles with lowest weights
        lowest_weight_indices = np.argsort(self.particle_filter.particles.weights)[:num_particles]
        # Update the positions of the particles with lowest weights
        for random_pos, index in zip(random_positions, lowest_weight_indices):
            self.particle_filter.particles.particles[index].x = random_pos[0]
            self.particle_filter.particles.particles[index].y = random_pos[1]

def main():
    # Example usage
    area = (0, 0, 10, 10)  # Define the area as (x_min, y_min, x_max, y_max)
    num_particles = 10
    landmarks = np.array([[1, 1], [2, 2], [3, 3]])  # Example landmarks
    landmark_ids = [1, 2, 4]

    localizer = SelfLocalizer(num_particles, landmarks, landmark_ids, area)

    # Simulate some movements and updates
    localizer.update_drive(1.0)
    localizer.update_turn(np.pi / 4)
    lidar_data = np.array([np.inf] * 360)
    lidar_data[100:130] = np.random.uniform(9, 10, 30)  # Simulate some lidar data
    lidar_data[210:230] = np.random.uniform(14, 15, 20)  # Simulate some lidar data
    localizer.update_image([1, 4], [1.5, 2.5], [0.5, 0.7], lidar_data)

    estimated_pose = localizer.particle_filter.estimate_pose()
    print(f"Estimated Pose: {estimated_pose}")
