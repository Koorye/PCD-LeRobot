"""
Piper end-effector robot class implementation.
"""

from .piper import Piper
from .configuration_piper import PiperEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class PiperEndEffector(Piper, BaseRobotEndEffector):
    """
    Piper robot class implementation with end effector.
    Params:
    - config: PiperEndEffectorConfig
    """
    def __init__(self, config: PiperEndEffectorConfig) -> None:
        super().__init__(config)
    
    def get_observation(self):
        state = self.get_ee_state()
        state_to_send = self.model_pose_transform.output_transform(state) # standard -> model
        obs_dict = {k: v for k, v in zip(self.action_features.keys(), state_to_send)}

        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            obs_dict[cam_key] = outputs

        self._current_state = state

        return obs_dict