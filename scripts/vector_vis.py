import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Header
import numpy as np

class OdometryPublisher(Node):
    def __init__(self, topic_name, frame_id='odom', rate=0.1):
        super().__init__('odometry_publisher')

        # Create a publisher for the Odometry message
        self.publisher_ = self.create_publisher(Odometry, topic_name, 10)
        self.frame_id = frame_id

        self.current_position = Point(x=0.0, y=0.0, z=0.0)
        self.current_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Timer to publish Odometry data at 10 Hz
        self.timer = self.create_timer(rate, self.publish_odometry)

    def odometry(self, position, orientation):
        """Update the Odometry data dynamically."""
        self.current_position = position
        self.current_orientation = orientation

    def odemetry_numpy(self, position: np.array, orientation: np.array):
        """
        Update the Odometry data dynamically using numpy arrays.
        Args: 
            position (np.array): The position of the robot in 3D space (x, y, z).
            orientation (np.array): The orientation of the robot in 3D space (x, y, z, w).
        """
        self.current_position = Point(x=position[0], y=position[1], z=position[2])
        norm = np.linalg.norm(orientation)
        self.current_orientation = Quaternion(x=orientation[0] / norm, y=orientation[1] / norm, z=orientation[2] / norm, w=0.0)

    def publish_odometry(self):
        # Create Odometry message
        odometry_msg = Odometry()

        # Set the header
        odometry_msg.header = Header()
        odometry_msg.header.stamp = self.get_clock().now().to_msg()
        odometry_msg.header.frame_id = self.frame_id

        odometry_msg.pose.pose.position = self.current_position
        odometry_msg.pose.pose.orientation = self.current_orientation

        # Publish the message
        self.publisher_.publish(odometry_msg)
        # self.get_logger().info('Publishing Odometry data at (0, 0, 0)')


def main():
    rclpy.init()
    node = OdometryPublisher('test_odometry')

    try:
        while rclpy.ok():
            # Update the Odometry data
            node.odometry(Point(x=0.0, y=0.0, z=0.0), Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

            # Spin the node to publish the Odometry message
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
