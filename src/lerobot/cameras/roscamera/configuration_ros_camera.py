from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("ros_camera")
@dataclass
class ROSCameraConfig(CameraConfig):
    """
    Configuration class for the DummyCamera.

    Attributes:
        fps: Frames per second for the dummy camera.
        width: Width of the dummy camera frames.
        height: Height of the dummy camera frames.
    """
    node_name: str = "ros_camera"
    topic: str = "/camera/image_raw"