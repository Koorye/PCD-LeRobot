"""
Base Robot class with joint control
"""

import numpy as np
from typing import Any, Dict

from lerobot.errors import DeviceNotConnectedError

from .config_base_leader import BaseLeaderConfig
from .units_transform import UnitsTransform
from ..teleoperator import Teleoperator


class BaseLeader(Teleoperator):
    """
    Base class for robot implementations with joint control.
    Subclasses should implement hardware-specific communication methods.
    Supports:
    1. Joint & End-Effector control
    2. Visualization support
    3. Unified unit management
    4. Absolute & Delta action support
    Params:
    - config: BaseRobotConfig
    e.g.
    ```python
    from lerobot.robots.base_robot import BaseRobot, BaseRobotConfig

    config = BaseRobotConfig(
        joint_names=['joint1_pos', 'joint2_pos', 'joint3_pos', 'gripper'],
        init_type='joint',
        init_state=[0.0, 0.0, 0.0, 0.0],
        joint_units=['radian', 'radian', 'radian', 'meter'],
        pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
        model_joint_units=['radian', 'radian', 'radian', 'meter'],
        cameras={
            'front_camera': OpenCVCameraConfig(
                width=640,
                height=480,
                fps=30,
            ),
        },
    )
    robot = BaseRobot(config)
    robot.connect()
    observation = robot.get_observation()
    action = {'joint1_pos': 0.1, 'joint2_pos': 0.2, 'joint3_pos': 0.3}
    robot.send_action(action)
    robot.disconnect()
    ```
    """

    config_class = BaseLeaderConfig
    name = "base_leader"

    def __init__(self, config: BaseLeaderConfig) -> None:
        """Initialize the robot with configuration settings"""

        super().__init__(config)
        self._check_dependency()

        self.config = config
        self.arm = None
        
        self.joint_transform = UnitsTransform(config.joint_units)
        self.pose_transform = UnitsTransform(config.pose_units)
        self.model_joint_transform = UnitsTransform(config.model_joint_units)

        self._init_state = None
        self._current_state = None
    
    def _check_dependency(self) -> None:
        """
        Check for required dependencies and libraries.
        Should be implemented by subclasses 
        to verify necessary hardware libraries are available.
        """
        return
    
    def _connect_arm(self):
        """
        Establish connection to the robot arm hardware.
        This method must be implemented by subclasses 
        to handle hardware-specific connection logic.
        """
        raise NotImplementedError
    
    def _disconnect_arm(self):
        """
        Disconnect from the robot arm hardware.
        This method must be implemented by subclasses 
        to handle hardware-specific disconnection logic.
        """
        raise NotImplementedError
    
    def _get_joint_state(self) -> np.ndarray:
        """
        Get joint positions from hardware.
        This method must be implemented by subclasses 
        to retrieve joint states from the physical robot.
        Returns:
        - state: Joint positions in robot-specific units
        """
        raise NotImplementedError
    
    def _get_ee_state(self) -> np.ndarray:
        """
        Get end-effector pose from hardware.
        This method must be implemented by subclasses
        to retrieve end-effector states from the physical robot.
        Returns:
        - state: End-effector pose in robot-specific units
        """
        raise NotImplementedError

    def get_joint_state(self) -> np.ndarray:
        """
        Get joint positions with automatic unit conversion.
        Retrieves joint positions from hardware and converts from robot-specific units to standard units.
        Returns:
        - state: Joint positions in standard units
        """
        state = self._get_joint_state()
        return self.joint_transform.input_transform(state) # joint -> standard
    
    def get_ee_state(self) -> np.ndarray:
        """
        Get end-effector pose with automatic unit conversion.
        Retrieves end-effector pose from hardware and converts from robot-specific units to standard units.
        Returns:
        - state: End-effector pose in standard units
        """
        state = self._get_ee_state()
        return self.pose_transform.input_transform(state) # end_effector -> standard
    
    def connect(self) -> None:
        """
        Connect to the robot and initialize components.
        1. Connects to all cameras.
        2. Connects to the robot arm.
        3. Warms up the cameras by capturing initial frames.
        4. Sets the robot to the initial state based on configuration.
        """
        self._connect_arm()

    def disconnect(self) -> None:
        """
        Disconnect from the robot and clean up resources.
        1. Disconnects all cameras.
        2. Disconnects from the robot arm.
        """
        self._disconnect_arm()
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get observation from robot including joint states and camera images.
        Retrieves joint states and camera images, applies unit conversions,
        and returns a combined observation dictionary.
        Returns:
        - obs_dict: Dictionary containing joint positions and camera images
                    in model units
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self.get_joint_state()
        state_to_send = self.model_joint_transform.output_transform(state) # standard -> model
        obs_dict = {k: v for k, v in zip(self._motors_ft.keys(), state_to_send)}

        self._current_state = state

        return obs_dict
    
    @property
    def _motors_ft(self) -> Dict[str, Any]:
        """
        Motor joint features dictionary.
        Returns:
        - dict mapping joint names to float types
        """
        return {
            f'{each}_pos': float for each in self.config.joint_names
        }

    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Action features dictionary.
        Returns:
        - dict mapping joint names to float types
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot and all cameras are connected.
        Returns:
        - bool indicating connection status
        """
        raise NotImplementedError
    
    def is_calibrated(self) -> bool:
        """
        Check if the robot is calibrated.
        Returns:
        - bool indicating calibration status, True by default
        """
        return True
    
    def calibrate(self) -> None:
        """
        Calibrate the robot, doing nothing by default.
        """
        pass

    def configure(self) -> None:
        """
        Configure the robot, doing nothing by default.
        """
        pass
    
    @property
    def feedback_features(self) -> Dict[str, Any]:
        """
        Feedback features dictionary.
        Returns:
        - dict mapping feedback names to float types
        """
        return {}
    
    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError