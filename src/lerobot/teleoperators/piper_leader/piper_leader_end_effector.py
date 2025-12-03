"""
Piper end-effector robot class implementation.
"""

from .piper_leader import PiperLeader
from .config_piper_leader import PiperLeaderEndEffectorConfig
from ..base_leader import BaseLeaderEndEffector


class PiperLeaderEndEffector(PiperLeader, BaseLeaderEndEffector):
    """
    Piper robot class implementation with end effector.
    Params:
    - config: PiperEndEffectorConfig
    """
    def __init__(self, config: PiperLeaderEndEffectorConfig) -> None:
        super().__init__(config)