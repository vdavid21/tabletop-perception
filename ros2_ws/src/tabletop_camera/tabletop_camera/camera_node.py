import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('sensor_mode', 3)
        self.declare_parameter('capture_width', 1640)
        self.declare_parameter('capture_height', 1232)
        self.declare_parameter('output_width', 820)
        self.declare_parameter('output_height', 616)
        self.declare_parameter('fps', 30)
        self.declare_parameter('frame_id', 'camera_optical_frame')

        self.frame_id = self.get_parameter('frame_id').value
        self.fps = self.get_parameter('fps').value

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, 'image_raw', qos_profile_sensor_data)

        self.cap = cv2.VideoCapture(self.build_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open camera pipeline')

        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)

    def build_pipeline(self) -> str:
        def p(name):
            return self.get_parameter(name).value
        return (
            f"nvarguscamerasrc sensor-id={p('sensor_id')} sensor-mode={p('sensor_mode')} ! "
            f"video/x-raw(memory:NVMM),width={p('capture_width')},height={p('capture_height')},"
            f"format=NV12,framerate={p('fps')}/1 ! "
            "nvvidconv ! "
            f"video/x-raw,width={p('output_width')},height={p('output_height')},format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1"
        )

    def publish_frame(self):
        ok, frame = self.cap.read()
        stamp = self.get_clock().now().to_msg()
        if not ok:
            self.get_logger().warn(
                'Frame read failed', throttle_duration_sec=5.0)
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self):
        self.cap.release()
        return super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
