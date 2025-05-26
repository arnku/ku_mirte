import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor

class MovementPublisher(Node):
    """
    This class is used to publish movement commands to the robot.
    Publishes Twist messages to the `/mirte_base_controller/cmd_vel_unstamped` topic.
    """
    def __init__(self, speed_modifier=1.0, rotation_modifier=1.0):
        super().__init__('movement_publisher')
        self.publisher_mirte = self.create_publisher(Twist, '/mirte_base_controller/cmd_vel', 10)
        self.publisher_gazebo = self.create_publisher(Twist, '/mirte_base_controller/cmd_vel_unstamped', 10)

        self.lin_speed: float = 0.0
        self.ang_speed: float = 0.0
        self.duration: float | None = 0.0

        # Multiplier for the speed of the robot between the Gazebo and Mirte controller.
        self.speed_modifier: float = speed_modifier  # Speed modifier to adjust the speed of the robot
        self.rotation_modifier: float = rotation_modifier  # Rotation modifier to adjust the rotation speed of the robot

        self.start_drive_time: float = time.time()
        self.driving : bool = False

        self.timer = self.create_timer(0.05, self._publish_volicity) 

    def drive(self, lin_speed: float, ang_speed: float, duration: float, interrupt:bool=True):
        """
        Drive the robot with a given speed and direction for a given duration.
        If duration is None, the robot will drive forever.
        If interrupt is `True`, the current drive will be interrupted. 
        If it however is `False`, the current drive will not be interrupted.

        Parameters:
            lin_speed (float): The linear velocity (m/s) of the robot. Positive values drive forward, negative values drive backward.
            ang_speed (float): The angular velocity (rad/s) of the robot. Positive values turn left, negative values turn right.
            duration (float): The duration (seconds) of the drive. If `0.0`, the robot will drive forever.
            interrupt (bool): If `True`, the current drive will be interrupted. If `False`, the current drive will not be interrupted.
        """
        if self.driving:
            if not interrupt: # The robot is already driving and dont want to interrupt
                return # Did not change the drive of the robot
        
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.duration = duration
        self.start_drive_time = time.time()        
    
    def tank_drive(self, left_speed: float, right_speed: float, duration: float, interrupt:bool=True):
        """
        Drive the robot with a given left and right speed for a given duration.
        If duration is `None`, the robot will drive forever.

        Parameters:
            left_speed (float): The left wheel speed (m/s) of the robot. Positive values drive forward, negative values drive backward.
            right_speed (float): The right wheel speed (m/s) of the robot. Positive values drive forward, negative values drive backward.
            duration (float): The duration (seconds) of the drive. If `None`, the robot will drive forever.
        """
        self.drive((left_speed + right_speed) / 2, (right_speed - left_speed) / 2, duration, interrupt) 

    def stop(self):
        """
        Stops the robot.
        """
        # Stop the robot by publishing zero velocities
        self.driving = False
        self.duration = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_mirte.publish(twist)
        self.publisher_gazebo.publish(twist)

    def _publish_volicity(self):
        """
        Always running. 
        """
        if self.duration is not None:
            if self.duration == 0.0:
                self.stop()
                return
            if time.time() - self.start_drive_time > self.duration:
                self.stop()
                return

        self.driving = True
        twist = Twist()
        twist.linear.x = self.lin_speed
        twist.angular.z = self.ang_speed
        self.publisher_gazebo.publish(twist)
        twist.linear.x = self.lin_speed * self.speed_modifier
        twist.angular.z = self.ang_speed * self.rotation_modifier
        self.publisher_mirte.publish(twist)
        

def main():
    print("Initializing")
    rclpy.init()
    node = MovementPublisher()
    
    print("Starting thread")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Run the executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    node.drive(-0.5, 0.0, 1.0)
    
    time.sleep(0.5) 

    node.drive(0.5, 0.0, 3.0)

    time.sleep(1)

    print("Stoppig")
    node.stop()
    executor_thread.join()  # Wait for the thread to finish before proceeding
    
    # Ensure proper shutdown
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
