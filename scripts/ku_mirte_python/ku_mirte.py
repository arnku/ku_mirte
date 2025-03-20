import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

import time

from robot_pos_sub import PositionSubscriber
from camera_sub import CameraSubscriber
from vector_vis import OdometryPublisher
from point_cloud_vis import PointCloudPublisher
from drive_pub import MovementPublisher

class KU_Mirte:
    def __init__(self, odometry_topic='test_odometry', point_cloud_topic='test_point_cloud'):
        rclpy.init()
        self.robot_pos_sub = PositionSubscriber()
        self.camera_sub = CameraSubscriber()
        self.odometry_pub = OdometryPublisher(odometry_topic)
        self.point_cloud_pub = PointCloudPublisher(point_cloud_topic)
        self.movement_pub = MovementPublisher()

        self.drive_thread = None
        self.drive_thread_executor = None
        self._start_drive_thread()
        

    def __del__(self):
        """Ensure that nodes are properly destroyed when the object is deleted."""
        
        self._stop_drive_thread()

        if self.robot_pos_sub:
            self.robot_pos_sub.destroy_node()
        if self.camera_sub:
            self.camera_sub.destroy_node()
        if self.odometry_pub:
            self.odometry_pub.destroy_node()
        if self.point_cloud_pub:
            self.point_cloud_pub.destroy_node()
        if self.movement_pub:
            self.movement_pub.destroy_node()
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
    
    def _start_drive_thread(self):
        self.drive_thread_executor = MultiThreadedExecutor()
        self.drive_thread_executor.add_node(self.movement_pub)
        self.drive_thread = threading.Thread(target=self.drive_thread_executor.spin_until_future_complete, args=(self.movement_pub.stop_future,))
        self.drive_thread.start()

    def _stop_drive_thread(self):
        if self.drive_thread:
            self.movement_pub.stop()
            self.movement_pub.shutdown()
            self.drive_thread.join()
            self.drive_thread_executor.shutdown()
            self.drive_thread_executor = None
            self.drive_thread = None

    def drive(self, lin_speed, ang_speed, duration, blocking=True):
        """
        Drive the robot with a given speed and direction for a given duration.
        If duration is `0.0`, the robot will drive forever.
        Blocking will wait for the drive to finish before continuing. 
        Disabling blocking will allow for the robot to drive while the script continues.
        The Robot can be stopped at any time by calling `stop()`.

        Parameters:
            lin_speed (float): The linear velocity (m/s) of the robot. Positive values drive forward, negative values drive backward.
            ang_speed (float): The angular velocity (rad/s) of the robot. Positive values turn left, negative values turn right.
            duration (float): The duration (seconds) of the drive. If `0.0`, the robot will drive forever.
            blocking (bool): If `True`, the function will wait for the drive to finish before continuing. If `False`, the function will return immediately.
        """
        self.movement_pub.drive(lin_speed, ang_speed, duration)
        
        while blocking and self.movement_pub.timer is not None: # Blocking
            time.sleep(0.01) # Robot can stop during this sleep, it is only to prevent the function from returning before the drive is finished.
    
    def stop(self):
        """Stops the robot."""
        self.movement_pub.stop()

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
