import cv2 # Import the OpenCV library
import time
import math
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

imageSize = (640, 480)
FOV = 70

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calc_box_size(corner):
    """
    Calculate the size of the box in the image in pixels
    """
    left_bar =  (corner[0][0][0] - corner[0][1][0])**2 + (corner[0][0][1] - corner[0][1][1])**2
    right_bar = (corner[0][2][0] - corner[0][3][0])**2 + (corner[0][2][1] - corner[0][3][1])**2

    avg = (left_bar + right_bar) / 2
    return math.sqrt(avg)

def calc_box_x_position(corner):
    # TODO: maths can be simplified
    x_center = imageSize[0] / 2
    x_marker = (corner[0][0][0] + corner[0][2][0]) / 2
    x_diff = x_center - x_marker
    x_diff_percent = x_diff / x_center
    x_diff_degree = x_diff_percent * FOV / 2

    return x_diff_degree

frame = 0
fps = 0
while cv2.waitKey(4) == -1: # Wait for a key pressed event

    time_s = time.time()

    image = mirte.get_image()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    
    cv2.imshow("Detected Markers", image)

    time_e = time.time()
    if frame > fps: # print once pr second
        frame = 0
        fps = 1 / (time_e - time_s)
        print(f"{fps=}")
        print(f"{ids=}")
        if mirte.position is None:
            continue
        print(f"robot position: ({mirte.position.x}, {mirte.position.y})")
        for corner in corners:
            print(f"{calc_box_size(corner)=}")
            print(f"{calc_box_x_position(corner)=}")
            print(f"distance: {distance((mirte.position.x, mirte.position.y), (box_size/2, 0))} m") # TODO: Add the angle to the calculation. Right now it only works along the x-axis
            print(f"focal length: {calc_box_size(corner) * (distance((mirte.position.x, mirte.position.y), (box_size/2, 0)) / aruco_size)}")


    frame += 1
