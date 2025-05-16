import os
import sys
from math import pi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ku_mirte_python')))
from ku_mirte import KU_Mirte

mirte = KU_Mirte()

# Drive in a figure 8 pattern
print("drive forward")
mirte.tank_drive(0.5, 0.5, 1.0)
print("drive left") 
mirte.tank_drive(1.5, 0.05, 5.0)
print("drive forward") 
mirte.tank_drive(0.5, 0.5, 1.0)
print("drive right") 
mirte.tank_drive(0.05, 1.5, 5.0) 

del mirte