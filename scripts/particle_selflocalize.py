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
    def __init__(self, num_particles: int = 100, area: tuple = (0, 0, 10, 10)):
        self.num_particles = num_particles
        self.particles = Particles(num_particles)
        
        # Initialize particles uniformly in a square area
        self.area = area
        self.particles.positions = np.random.uniform(low=[area[0], area[1]], high=[area[2], area[3]], size=(num_particles, 2))
        self.particles.angles = np.random.uniform(low=0, high=2 * np.pi, size=num_particles)

    def position_probabilities(self, measured_dists: list, landmarks: np.ndarray, sigma: float = 17.0) -> np.ndarray:
        """
        Input:
        - measured_dists: list of measured distances to landmarks [d1, d2, ...]
        - sigma: standard deviation of the distance measurement

        Output:
        - probs: 2D array of probabilities for each particle and landmark [[p1_l1, p1_l2, ...], [p2_l1, p2_l2, ...], ...]
        """
        # Expand dimensions to calculate distances from each particle to each landmark
        # Calculate the squared differences along each coordinate (shape: [n_particles, n_landmarks, 2])
        diff = (self.particles.positions[:, np.newaxis, :] - landmarks[np.newaxis, :, :])
        dists = np.linalg.norm(diff, axis=2)  # Shape: [n_particles, n_landmarks]

        # Calculate the probabilities (shape: [n_particles, n_landmarks])
        probs = norm.pdf(measured_dists, dists, sigma)

        return probs

    def angle_probabilities(self, measured_angles: dict, landmarks: np.ndarray, sigma: float = 0.4) -> np.ndarray:
        """
        Input:
        - measured_angles: list of measured angles to landmarks [a1, a2, ...]
        - sigma: standard deviation of the angle measurement

        Output:
        - probs: 2D array of probabilities for each particle and landmark [[p1_l1, p1_l2, ...], [p2_l1, p2_l2, ...], ...]
        """
        # Calculate the unit vector to each landmark from each particle
        # Shape of diff: [n_particles, n_landmarks, 2]
        diff = (landmarks[np.newaxis, :, :] - self.particles.positions[:, np.newaxis, :])
        dists = np.linalg.norm(diff, axis=2)  # Shape: [n_particles, n_landmarks]
        e_ls = (diff / dists[..., np.newaxis])  # Unit vectors to landmarks, shape: [n_particles, n_landmarks, 2]

        # Calculate the orientation vectors and their orthogonals for each particle
        e_thetas = np.stack([np.cos(self.particles.angles), np.sin(self.particles.angles)], axis=1)  # Shape: [n_particles, 2]
        e_ortho_thetas = np.stack([-e_thetas[:, 1], e_thetas[:, 0]], axis=1)  # Orthogonal vectors, shape: [n_particles, 2]

        # Calculate theoretical angles using the dot product
        dot_products = np.einsum("ijk,ik->ij", e_ls, e_ortho_thetas)  # Dot products, shape: [n_particles, n_landmarks]
        angle_signs = np.sign(dot_products)  # Shape: [n_particles, n_landmarks]
        angles = angle_signs * np.arccos(np.einsum("ijk,ik->ij", e_ls, e_thetas))  # Shape: [n_particles, n_landmarks]

        # Calculate probabilities for each particle and landmark based on angles
        probs = norm.pdf(measured_angles, angles, sigma)

        return probs
    
    def particle_weights(self, measured_dists:list, measured_angles:list, landmarks:list) -> None:
        position_probs = self.position_probabilities(measured_dists, landmarks)
        angle_probs = self.angle_probabilities(measured_angles, landmarks)
        
        pos_angle_probs = np.prod([position_probs, angle_probs], axis=0)
        combined_probs = np.prod(pos_angle_probs, axis=1)

        # Normalize the combined probabilities
        combined_probs /= np.sum(combined_probs)

        self.particles.weights = combined_probs
        
    def resample_particles(self) -> None:
        new_particle_indecies = np.random.choice(self.num_particles, self.num_particles, p=self.particles.weights)
        self.particles.rearrange(new_particle_indecies)
        # normalize the weights after resampling
        self.particles.weights /= np.sum(self.particles.weights)
        
    def estimate_pose(self) -> Particle:
        x_avg, y_avg = np.mean(self.particles.positions, axis=0)
        theta_avg = np.mean(self.particles.angles)

        p = Particle()
        p.x = x_avg
        p.y = y_avg
        p.theta = theta_avg

        return p
    

class SelfLocalizer:
    def __init__(self, num_particles: int, landmarks: list, bounds: tuple):
        self.landmarks = landmarks
        self.num_particles = num_particles
        self.particle_filter = ParticleFilter(num_particles, area=bounds)

    def update_drive(self, dist: float) -> None:
        # Get angles of all particles as a vector

        # Calculate dx and dy vectors
        dx = dist * np.cos(self.particle_filter.particles.angles)
        dy = dist * np.sin(self.particle_filter.particles.angles)

        # Move each particle using vectorized dx, dy
        self.particle_filter.particles.positions += np.column_stack((dx, dy))

        #self.add_uncertainty()

    def update_turn(self, angle: float) -> None:
        self.particle_filter.particles.angles += angle

        #self.add_uncertainty()

    def update_image(self, distances: list, angles: list, iterations=3) -> None:
        for _ in range(iterations):
            self.particle_filter.particle_weights(distances, angles, self.landmarks)
            self.particle_filter.resample_particles()

    def add_uncertainty(self, sigma_x=1, sigma_y=1, sigma_theta=0.1):
        # Create uncertainty arrays
        x_uncertainty = np.random.normal(0, sigma_x, self.num_particles)
        y_uncertainty = np.random.normal(0, sigma_y, self.num_particles)
        theta_uncertainty = np.random.normal(0, sigma_theta, self.num_particles) * np.random.choice([1, -1], self.num_particles)

        # Add uncertainty to each particle
        self.particle_filter.particles.positions += np.column_stack((x_uncertainty, y_uncertainty))
        self.particle_filter.particles.angles += theta_uncertainty


def main():
    particle = Particle()
    particle.x = 1
    particle.y = 2
    particle.theta = 3 * np.pi

    assert(particle.x == 1)
    assert(particle.y == 2)
    assert(particle.theta == np.pi)
    del particle

    num_particles = 1000
    particle_filter = ParticleFilter(num_particles=num_particles, area=(0, 0, 10, 10))
    
    assert(particle_filter.particles.positions.shape == (num_particles, 2))
    assert(particle_filter.particles.angles.shape == (num_particles, ))
    assert(particle_filter.particles.weights.shape == (num_particles, ))

    particle_filter.particle_weights(
        measured_dists=[1, 2, 3],
        measured_angles=[0.1, 0.2, 0.3],
        landmarks=np.array([[1, 1], [2, 2], [3, 3]])
    )
    
    assert(particle_filter.particles.positions.shape == (num_particles, 2))
    assert(particle_filter.particles.angles.shape == (num_particles, ))
    assert(particle_filter.particles.weights.shape == (num_particles, ))
    assert(not np.all(particle_filter.particles.positions == particle_filter.particles.positions[0]))
    assert(np.sum(particle_filter.particles.weights) == 1.0)

    particle_filter.resample_particles()

    assert(particle_filter.particles.positions.shape == (num_particles, 2))
    assert(particle_filter.particles.angles.shape == (num_particles, ))
    assert(particle_filter.particles.weights.shape == (num_particles, ))
    assert(not np.all(particle_filter.particles.positions == particle_filter.particles.positions[0]))
    assert(np.sum(particle_filter.particles.weights) == 1.0)

    particle_filter.resample_particles()

    estimated_pose = particle_filter.estimate_pose()
    assert(estimated_pose.x >= 0 and estimated_pose.x <= 10)
    assert(estimated_pose.y >= 0 and estimated_pose.y <= 10)
    assert(estimated_pose.theta >= 0 and estimated_pose.theta <= 2 * np.pi)

    self_localizer = SelfLocalizer(
        num_particles=num_particles,
        landmarks=np.array([[1, 1], [2, 2], [3, 3]]),
        bounds=(0, 0, 10, 10)
    )
    self_localizer.update_drive(1.0)
    self_localizer.update_turn(np.pi / 4)
    self_localizer.update_image([1, 2, 3], [0.1, 0.2, 0.3])
    estimated_pose = self_localizer.particle_filter.estimate_pose()


if __name__ == "__main__":
    main()