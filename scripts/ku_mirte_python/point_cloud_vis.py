import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import struct
import numpy as np
import time

class PointCloudPublisher(Node):
    def __init__(self, topic_name, frame_id='odom', rate=0.1):
        super().__init__('pointcloud_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, topic_name, 10)
        self.frame_id = frame_id
        self.current_points = np.empty((0, 3))  # Default empty point cloud
        self.current_colors = np.empty((0, 4))  # Default empty color array
        self.timer = self.create_timer(rate, self.publish_pointcloud)

    def points(self, new_points, new_colors=None):
        """Update the point cloud data dynamically, with optional colors."""
        self.current_points = np.array(new_points)

        # If colors are not provided, set default white (RGBA = 255,255,255,255)
        if new_colors is None:
            self.current_colors = np.full((len(new_points), 4), 255, dtype=np.uint8)
        else:
            self.current_colors = np.array(new_colors, dtype=np.uint8)

        # self.get_logger().info(f"Updated point cloud with {len(self.current_points)} points.")

    def publish_pointcloud(self):
        if self.current_points.size == 0:
            self.get_logger().warn("No points set! Call node.points(points, colors) to set data.")
            return

        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = self.frame_id  

        cloud_msg.height = 1
        cloud_msg.width = len(self.current_points)
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.FLOAT32, count=1)  # Color field
        ]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, rgba)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True

        # Pack the data into the PointCloud2 message
        data = []
        for point, color in zip(self.current_points, self.current_colors):
            r, g, b, a = color  # Extract RGBA
            rgba = struct.unpack('f', struct.pack('BBBB', b, g, r, a))[0]  # Pack color into float32
            data.append(struct.pack('ffff', point[0], point[1], point[2], rgba))  # Pack all fields

        cloud_msg.data = b''.join(data)

        self.publisher_.publish(cloud_msg)
        # self.get_logger().info(f'Published PointCloud2 with {len(self.current_points)} points')

def main():
    rclpy.init()
    node = PointCloudPublisher('test_pointcloud')

    try:
        while rclpy.ok():
            # Generate new random points
            new_points = np.random.rand(100, 3) * 10

            # Generate random colors (RGBA format, values 0-255)
            new_colors = np.random.randint(0, 256, (100, 4), dtype=np.uint8)

            node.points(new_points, new_colors)  # Update the point cloud with colors
            rclpy.spin_once(node, timeout_sec=1.0)  # Process callbacks
            time.sleep(0.05)  # Wait before updating again

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
