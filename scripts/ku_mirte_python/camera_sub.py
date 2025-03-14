import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        # Create a subscriber to the 'camera/image_raw' topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10  # Queue size
        )
        self.subscription  # prevent unused variable warning
        
        # Initialize CvBridge to convert ROS image messages to OpenCV
        self.bridge = CvBridge()

        # Initialize a variable to store the latest image
        self.latest_image = None

    def listener_callback(self, msg):
        """Callback function to process incoming camera images."""
        try:
            # Convert the ROS Image message to a CV2 image (BGR format)
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def image(self):
        """Returns the most recent image received."""
        return self.latest_image

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()

    # Spin the node in a separate thread so we can update the image continuously
    rclpy.spin_once(camera_subscriber)

    # Loop to display the latest image
    while rclpy.ok():
        # Check if a new image is available
        img = camera_subscriber.image()

        if img is not None:
            # Display the image using OpenCV
            cv2.imshow("Camera Image", img)
        
        # Wait for 1 millisecond to update the OpenCV window
        key = cv2.waitKey(1)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

        # Spin the node to allow ROS2 callbacks to be processed
        rclpy.spin_once(camera_subscriber)

    # Clean up
    camera_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
