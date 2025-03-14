# Examples

Run the gazebo enviorenment
```sh
ros2 launch ku_mirte ku_mirte_qr.launch.xml
```

Run a python script
```sh
python3 src/ku_mirte/scripts/ex3_focal_length.py
```

Control the robot
```sh
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/mirte_base_controller/cmd_vel_unstamped
```

