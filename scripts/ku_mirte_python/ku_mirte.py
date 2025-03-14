import rclpy

from robot_pos_sub import PositionSubscriber
from camera_sub import CameraSubscriber
from vector_vis import OdometryPublisher
from point_cloud_vis import PointCloudPublisher

class KU_Mirte:
    def __init__(self, odometry_topic='test_odometry', point_cloud_topic='test_point_cloud'):
        rclpy.init()
        self.robot_pos_sub = PositionSubscriber()
        self.camera_sub = CameraSubscriber()
        self.odometry_pub = OdometryPublisher(odometry_topic)
        self.point_cloud_pub = PointCloudPublisher(point_cloud_topic)

    def __del__(self):
        """Ensure that nodes are properly destroyed when the object is deleted."""
        if self.robot_pos_sub:
            self.robot_pos_sub.destroy_node()
        if self.camera_sub:
            self.camera_sub.destroy_node()
        if self.odometry_pub:
            self.odometry_pub.destroy_node()
        if self.point_cloud_pub:
            self.point_cloud_pub.destroy_node()
        rclpy.shutdown()
    
    def get_position(self):
        """Returns the most recent position of the robot."""
        rclpy.spin_once(self.robot_pos_sub)
        return self.robot_pos_sub.get_position()
    
    def get_rotation(self):
        """Returns the most recent rotation of the robot."""
        rclpy.spin_once(self.robot_pos_sub)
        return self.robot_pos_sub.get_rotation()

    @property
    def position(self):
        """Property that updates and returns the position."""
        position = self.get_position()
        if position is None:
            print("Position not initialized...")
        return position
    
    @property
    def rotation(self):
        """Property that updates and returns the rotation."""
        rotation = self.get_rotation()
        if rotation is None:
            print("Rotation not initialized...")
        return rotation

    def get_image(self):
        """Returns the most recent image from the camera."""
        rclpy.spin_once(self.camera_sub)
        return self.camera_sub.image()

    def set_odometry(self, position, rotation):
        """Sets the odometry position and rotation."""
        self.odometry_pub.odemetry_numpy(position, rotation)
        rclpy.spin_once(self.odometry_pub)
    
    def set_point_cloud(self, points, colors=None):
        """Sets the point cloud data."""
        self.point_cloud_pub.points(points, colors)
        rclpy.spin_once(self.point_cloud_pub)

def main():
    mirte = KU_Mirte()
    while rclpy.ok():
        try:
            position = mirte.position
            rotation = mirte.rotation
            image = mirte.get_image()

            if position is not None and rotation is not None:
                print(f"position: x={position.x}, y={position.y}, z={position.z}")
                print(f"rotation: x={rotation.x}, y={rotation.y}, z={rotation.z}, w={rotation.w}")
            print(f"image: {image.shape}")

            #mirte.set_odometry(position, rotation)
            #mirte.set_point_cloud(position)

        except KeyboardInterrupt:
            break
    
    del mirte 

if __name__ == '__main__':
    main()