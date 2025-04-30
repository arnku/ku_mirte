import os
import sys
from math import pi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

mirte = KU_Mirte()

while True:
    left_lidar = (mirte.get_lidar_section(-0.25*pi, -0.1*pi))
    right_lidar = (mirte.get_lidar_section(0.1*pi, 0.25*pi))

    print("R", end = '')
    for l in left_lidar:
        if l == float('inf'):
            print(".", end = '')
        else:
            print("#", end = '')
    print("   L", end = '')
    for r in right_lidar:
        if r == float('inf'):
            print(".", end = '')
        else:
            print("#", end = '')
    print("")

    left_dist = min(left_lidar)
    right_dist = min(right_lidar)

    diff = left_dist - right_dist
    if diff > 0.1:
        print("turning left")
        mirte.tank_drive(0.5, 0.0, None, blocking = False)
    elif diff < -0.1:
        print("turning right")
        mirte.tank_drive(0.0, 0.5, None, blocking = False)
    else:
        print("going straight")
        mirte.tank_drive(0.5, 0.5, None, blocking = False)


del mirte