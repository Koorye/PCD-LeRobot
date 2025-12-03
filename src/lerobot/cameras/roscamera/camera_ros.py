import numpy as np
import time
from typing import Any

from lerobot.cameras.camera import Camera


class ROSCamera(Camera):
    """
    Dummy camera implementation for testing purposes.
    This camera returns random rgb images instead of capturing from hardware.

    Example:
        ```python
        config = DummyCameraConfig(fps=30, width=640, height=480)
        camera = DummyCamera(config)
        camera.connect()

        # frame: np.ndrray of shape (height, width, 3) with random values
        frame = camera.read()

        camera.disconnect()
        ```
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._is_connected = False
        self.config = config
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        rospy.init_node(self.config.node_name, anonymous=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.config.topic, Image, self.callback)
        self._is_connected = True
        time.sleep(1)  # Give some time to establish connection
    
    def callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("DummyCamera does not support method find_cameras")

    def read(self) -> np.ndarray:
        return self.image

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self.image

    def disconnect(self) -> None:
        import rospy
        rospy.signal_shutdown('Camera disconnected')
        self._is_connected = False