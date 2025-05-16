import os
import sys
from math import pi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

mirte = KU_Mirte()

while True:
    sonar = mirte.get_sonar_ranges()
    left_sonar, right_sonar = sonar['front_left'], sonar['front_right']

    print(f"L: {left_sonar:.2f}   R: {right_sonar:.2f}  ", end = '')

    diff = left_sonar - right_sonar
    if diff > 0.1:
        print("turning left")
        mirte.drive(0.5, 2.0, None, blocking = False)
    elif diff < -0.1:
        print("turning right")
        mirte.drive(0.5, -2.0, None, blocking = False)
    else:
        print("going straight")
        mirte.drive(0.5, 0.5, None, blocking = False)


del mirte