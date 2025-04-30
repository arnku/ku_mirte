import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

import time

from robot_pos_sub import PositionSubscriber
from camera_sub import CameraSubscriber
from lidar_sub import LidarSubscriber
from vector_vis import OdometryPublisher
from point_cloud_vis import PointCloudPublisher
from drive_pub import MovementPublisher

class KU_Mirte:
    def __init__(self):
        rclpy.init()
        self.robot_pos_sub = PositionSubscriber()
        self.camera_sub = CameraSubscriber()
        self.lidar_sub = LidarSubscriber()
        self.odometry_pub_mirte = OdometryPublisher('odometry_mirte', 'base_link')
        self.odometry_pub_world = OdometryPublisher('odometry_world', 'odom')
        self.pointcloud_pub_mirte = PointCloudPublisher('pointcloud_mirte', 'base_link')
        self.pointcloud_pub_world = PointCloudPublisher('pointcloud_world', 'odom')
        self.movement_pub = MovementPublisher()

        self.executor_thread = None
        self.executor = None
        self._start_executor_thread()

        rclpy.spin_once(self.robot_pos_sub)
        rclpy.spin_once(self.lidar_sub)

        self.k_matrix = self.camera_sub.k_matrix # Camera intrinsic matrix
        self.d_matrix = self.camera_sub.d_matrix # Camera distortion
        self.p_matrix = self.camera_sub.p_matrix # Camera projection matrix

        print("KU Mirte initialized.")

    def __del__(self):
        """Ensure that nodes are properly destroyed when the object is deleted."""
        
        self._stop_executor_thread()

        if hasattr(self, 'robot_pos_sub') and self.robot_pos_sub:
            self.robot_pos_sub.destroy_node()
        if hasattr(self, 'camera_sub') and self.camera_sub:
            self.camera_sub.destroy_node()
        if hasattr(self, 'lidar_sub') and self.lidar_sub:
            self.lidar_sub.destroy_node()
        if hasattr(self, 'odometry_pub_mirte') and self.odometry_pub_mirte:
            self.odometry_pub_mirte.destroy_node()
        if hasattr(self, 'odometry_pub_world') and self.odometry_pub_world:
            self.odometry_pub_world.destroy_node()
        if hasattr(self, 'pointcloud_pub_mirte') and self.pointcloud_pub_mirte:
            self.pointcloud_pub_mirte.destroy_node()
        if hasattr(self, 'pointcloud_pub_world') and self.pointcloud_pub_world:
            self.pointcloud_pub_world.destroy_node()
        if hasattr(self, 'movement_pub') and self.movement_pub:
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
    
    def _start_executor_thread(self):
        print("Starting executor thread...")
        self.executor = MultiThreadedExecutor()

        self.executor.add_node(self.camera_sub)
        self.executor.add_node(self.movement_pub)
        self.executor.add_node(self.lidar_sub)
        self.executor.add_node(self.pointcloud_pub_mirte)
        self.executor.add_node(self.pointcloud_pub_world)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        print("Executor thread started.")
    
    def _stop_executor_thread(self):
        if hasattr(self, 'executor_thread') and self.executor_thread:
            self.executor.shutdown()
            self.executor_thread.join()
            self.executor = None
            self.executor_thread = None

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
        self.movement_pub.drive(float(lin_speed), float(ang_speed), duration)
        
        while blocking and self.movement_pub.timer is not None: # Blocking
            time.sleep(0.01) # Robot can stop during this sleep, it is only to prevent the function from returning before the drive is finished.
    
    def tank_drive(self, left_speed, right_speed, duration, blocking=True):
        """
        Drive the robot with a given left and right speed for a given duration.
        If duration is `0.0`, the robot will drive forever.
        Blocking will wait for the drive to finish before continuing. 
        Disabling blocking will allow for the robot to drive while the script continues.
        The Robot can be stopped at any time by calling `stop()`.

        Parameters:
            left_speed (float): The left wheel speed (m/s) of the robot. Positive values drive forward, negative values drive backward.
            right_speed (float): The right wheel speed (m/s) of the robot. Positive values drive forward, negative values drive backward.
            duration (float): The duration (seconds) of the drive. If `0.0`, the robot will drive forever.
            blocking (bool): If `True`, the function will wait for the drive to finish before continuing. If `False`, the function will return immediately.
        """
        self.movement_pub.tank_drive(left_speed, right_speed, duration)
        
        while blocking and self.movement_pub.timer is not None:
            time.sleep(0.01)
    
    def stop(self):
        """Stops the robot."""
        self.movement_pub.stop()

    def get_image(self):
        """Returns the most recent image from the camera."""
        #rclpy.spin_once(self.camera_sub)
        return self.camera_sub.image()
    
    def get_lidar_ranges(self):
        """Returns the most recent lidar ranges."""
        #rclpy.spin_once(self.lidar_sub)
        return self.lidar_sub.ranges()
    
    def get_lidar_section(self, start_angle, end_angle):
        """
        Returns the lidar ranges for a given angle section.
        The start_angle and end_angle are in radians from -pi to pi. 
        Here 0 is straight ahead, pi/2 is to the left, and -pi/2 is to the right.

        Parameters:
            start_angle (float): The start angle of the section in radians.
            end_angle (float): The end angle of the section in radians.
        """
        #rclpy.spin_once(self.lidar_sub)
        return self.lidar_sub.angle_section(start_angle, end_angle)

    def set_odometry(self, reference, position, rotation):
        """Sets the odometry position and rotation."""
        if reference == 'mirte':
            self.odometry_pub_mirte.set_position(position, rotation)
            rclpy.spin_once(self.odometry_pub_mirte)
        elif reference == 'world':
            self.odometry_pub_world.set_position(position, rotation)
            rclpy.spin_once(self.odometry_pub_world)
        else:
            raise ValueError("Reference must be 'mirte' or 'world'.")
        
    def set_pointcloud(self, reference, points, colors=None):
        """Sets the point cloud data."""
        if reference == 'mirte':
            self.pointcloud_pub_mirte.set_points(points, colors)
            #rclpy.spin_once(self.pointcloud_pub_mirte)
        elif reference == 'world':
            self.pointcloud_pub_world.set_points(points, colors)
            #rclpy.spin_once(self.pointcloud_pub_world)
        else:
            raise ValueError("Reference must be 'mirte' or 'world'.")
    
