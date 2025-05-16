import os
import sys
from math import pi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

mirte = KU_Mirte()

for _ in range(4):
    print("Moving forward")
    mirte.drive(0.5, 0.0, 1.0) # 0.5 m forward
    print("Turning right")
    mirte.drive(0.0, 2.0, pi/2) # 90 degrees to the left

del mirte