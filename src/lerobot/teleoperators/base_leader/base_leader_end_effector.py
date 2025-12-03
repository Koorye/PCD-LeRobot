"""
Base class for robots controlled via end-effector (EE) commands.
"""

import numpy as np
from typing import Any, Dict

from lerobot.errors import DeviceNotConnectedError

from .base_leader import BaseLeader
from .config_base_leader import BaseLeaderEndEffectorConfig
from .units_transform import UnitsTransform


class BaseLeaderEndEffector(BaseLeader):
    """
    Base class for robots controlled via end-effector (EE) commands.
    Extends the BaseRobot class to provide end-effector level control.
    Handles unit conversions and action preparation specific to EE control.
    Params:
    - config: Configuration object for the end-effector robot
    e.g.
    ```python
    from lerobot.robots.base_robot.base_robot_end_effector import BaseRobotEndEffector
    from lerobot.robots.base_robot.configuration_base_robot import BaseRobotEndEffectorConfig

    config = BaseRobotEndEffectorConfig(
        pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
        model_pose_units=['meter', 'meter', 'meter', 'radian', 'radian', 'radian', 'meter'],
    )
    robot = BaseRobotEndEffector(config)
    robot.connect()
    obs = robot.get_observation()
    action = {'x': 0.1, 'y': 0.0, 'z': 0.2, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper': 0.05}
    robot.send_action(action)
    robot.disconnect()
    ```
    """

    config_class = BaseLeaderEndEffectorConfig
    name = "base_leader_end_effector"

    def __init__(self, config: BaseLeaderEndEffectorConfig) -> None:
        """
        Initialize the end-effector controlled robot.
        """
        super().__init__(config)
        self.model_pose_transform = UnitsTransform(config.model_pose_units)
    
    def connect(self) -> None:
        """
        Connect to the robot and initialize the initial end-effector state.
        """
        super().connect()
        self._init_state = self.get_ee_state()
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get current observation and update EE state.
        Calls the base class method and updates current EE state.
        Returns:
        - obs_dict: Dictionary of current observations
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        state = self.get_ee_state()
        state_to_send = self.model_joint_transform.output_transform(state) # standard -> model
        obs_dict = {k: v for k, v in zip(self.action_features, state_to_send)}

        self._current_state = state

        return obs_dict

    @property
    def action_features(self) -> Dict[str, Any]:
        """
        Define the action features for end-effector control.
        Returns:
        - Dictionary mapping action feature names to their types
        """
        return {
            each: float for each in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        }