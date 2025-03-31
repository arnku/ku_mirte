import cv2 # Import the OpenCV library
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte


aruco_size = 0.25 # m
box_size = 0.4 # m


mirte = KU_Mirte()

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

mirte.odometry_pub_mirte.target_frame = 'base_link'

colors = [
    [255, 0, 0, 255], # Red
    [0, 255, 0, 255], # Green
    [0, 0, 255, 255], # Blue
    [255, 255, 0, 255], # Yellow
    [255, 0, 255, 255], # Magenta
    [0, 255, 255, 255], # Cyan
    [255, 255, 255, 255], # White
    [0, 0, 0, 255], # Black
]

while cv2.waitKey(4) == -1: # Wait for a key pressed event

    image = mirte.get_image()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow("Detected Markers", image)

    if ids is  None:
        continue


    rvecs, tvecs, rejectedImgPoints = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, mirte.k_matrix, mirte.d_matrix)

    tvecs = [[tvec[0][2], -tvec[0][0], -tvec[0][1]] for tvec in tvecs]
    
    c = [colors[id] for id in ids.flatten()]
    mirte.set_pointcloud('mirte', tvecs, c)



cv2.destroyAllWindows() # Close the OpenCV windows
del mirte # Clean up the ROS node