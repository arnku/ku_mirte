import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

# Import necessary ROS2 modules
from rclpy.node import Node

from robotPos_sub import PositionSubscriber
from camera_sub import CameraSubscriber
from vector_vis import OdometryPublisher

rclpy.init()

robot_pos_sub = PositionSubscriber()
camera_sub = CameraSubscriber()
odometry_pub = OdometryPublisher('test_odometry')

def get_image():
    rclpy.spin_once(camera_sub)
    return camera_sub.image()

def get_position_and_rotation():
    rclpy.spin_once(robot_pos_sub)
    position = robot_pos_sub.get_position()
    rotation = robot_pos_sub.get_rotation()
    return position, rotation

def set_odometry(position, rotation):
    print(f"Setting vector position: {position}")
    print(f"Setting vector rotation: {rotation}")
    odometry_pub.odemetry_numpy(position * 10, rotation)
    rclpy.spin_once(odometry_pub)

camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# Aruco marker dectection
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()
while rclpy.ok():
    image = get_image()
    position, rotation = get_position_and_rotation()
    if image is not None:
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        print(f"Detected {len(corners)} markers")
        if ids is not None:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.1)
                set_odometry(tvec[i][-1], rvec[i][-1])
        cv2.imshow("Aruco Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

rclpy.shutdown()
cv2.destroyAllWindows()