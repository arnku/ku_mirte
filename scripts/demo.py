import os
import sys
from math import pi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

mirte = KU_Mirte()


mirte.drive(0.5, 0.0, 1.0) # 0.5 m forward
mirte.drive([0.0,1.0], 0.0, 1.0) # 0.5 m left
mirte.drive(0.0, 1.0, pi) # 180 degrees to the left
mirte.drive(1.0, pi, 2.0) # turn back
mirte.drive([-1.0,1.0], 0.0, 3.0) # straife back
mirte.drive(0.0, 2.0, pi) # 360 degrees to the left

del mirte