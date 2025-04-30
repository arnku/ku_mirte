import time
import threading
from concurrent.futures import Future
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor

class MovementPublisher(Node):
    """
    This class is used to publish movement commands to the robot.
    Publishes Twist messages to the `/mirte_base_controller/cmd_vel_unstamped` topic.
    """
    def __init__(self):
        super().__init__('movement_publisher')
        self.publisher_ = self.create_publisher(Twist, '/mirte_base_controller/cmd_vel_unstamped', 10)

        self.lin_speed: float = 0.0
        self.ang_speed: float = 0.0
        self.duration: float = 0.0

        self.timer = None
        self.start_drive_time: float | None = None

        self.stop_future = Future()  # Future to signal when to stop

    def drive(self, lin_speed: float, ang_speed: float, duration: float, interrupt:bool=True):
        """
        Drive the robot with a given speed and direction for a given duration.
        If duration is `0.0`, the robot will drive forever.
        If interrupt is `True`, the current drive will be interrupted. 
        If it however is `False`, the current drive will not be interrupted.

        Parameters:
            lin_speed (float): The linear velocity (m/s) of the robot. Positive values drive forward, negative values drive backward.
            ang_speed (float): The angular velocity (rad/s) of the robot. Positive values turn left, negative values turn right.
            duration (float): The duration (seconds) of the drive. If `0.0`, the robot will drive forever.
            interrupt (bool): If `True`, the current drive will be interrupted. If `False`, the current drive will not be interrupted.
        """
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.duration = duration
        self.start_drive_time = time.time()

        if self.timer is not None:
            if not interrupt: # The robot is already driving and dont want to interrupt
                return # Did not change the drive of the robot
            if interrupt: # The robot is already driving and want to interrupt
                self.timer.cancel() # Stop the current drive
        
        self.timer = self.create_timer(0.01, self._publish_volicity) # Create a timer to publish the velocity every 0.1 seconds
    
    def tank_drive(self, left_speed: float, right_speed: float, duration: float):
        """
        Drive the robot with a given left and right speed for a given duration.
        If duration is `None`, the robot will drive forever.

        Parameters:
            left_speed (float): The left wheel speed (m/s) of the robot. Positive values drive forward, negative values drive backward.
            right_speed (float): The right wheel speed (m/s) of the robot. Positive values drive forward, negative values drive backward.
            duration (float): The duration (seconds) of the drive. If `None`, the robot will drive forever.
        """
        self.drive((left_speed + right_speed) / 2, (right_speed - left_speed) / 2, duration)

    def stop(self):
        """
        Stops the robot.
        """
        # Stop the robot by publishing zero velocities
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def shutdown(self):
        """
        Stops the robot, and signals the node to stop spinning if running in a separate thread.
        Use `drive_thread_executor.spin_until_future_complete` and wait for `stop_future` to complete.
        """
        self.stop()
        # Signal the node to stop spinning
        self.stop_future.set_result(True)

    def _publish_volicity(self):
        twist = Twist()
        twist.linear.x = self.lin_speed
        twist.angular.z = self.ang_speed
        self.publisher_.publish(twist)

        if self.duration is None:
            return # The robot should drive forever
        if self.duration == 0.0:
            self.stop()
        if time.time() - self.start_drive_time > self.duration:
            self.stop()
            
        

def main():
    rclpy.init()
    node = MovementPublisher()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Run the executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin_until_future_complete, args=(node.stop_future,))
    executor_thread.start()

    node.drive(-0.5, 0.0, 1.0)
    
    time.sleep(0.5) 

    node.drive(0.5, 0.0, 3.0)

    time.sleep(1)

    node.stop()

    time.sleep(2)

    node.shutdown()

    executor_thread.join()  # Wait for the thread to finish before proceeding
    
    # Ensure proper shutdown
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
