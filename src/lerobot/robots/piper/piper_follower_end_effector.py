"""
Piper end-effector robot class implementation.
"""

from .piper_follower import PiperFollower
from .configuration_piper import PiperEndEffectorConfig
from ..base_robot import BaseRobotEndEffector


class PiperFollowerEndEffector(PiperFollower, BaseRobotEndEffector):
    """
    Piper robot class implementation with end effector.
    Params:
    - config: PiperEndEffectorConfig
    """
    def __init__(self, config: PiperEndEffectorConfig) -> None:
        super().__init__(config)