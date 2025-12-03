from dataclasses import dataclass, field
from typing import List

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("base_leader")
@dataclass
class BaseLeaderConfig(TeleoperatorConfig):
    # list of joint names, including gripper
    joint_names: List[str] = field(default_factory=lambda: [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper',
    ])
    # units for robot joints, for sdk control
    joint_units: List[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])
    # units for end effector pose, for sdk control
    pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])
    # units for model joints, for model input/output
    model_joint_units: List[str] = field(default_factory=lambda: [
        'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'radian', 'm',
    ])


@TeleoperatorConfig.register_subclass("base_leader_end_effector")
@dataclass
class BaseLeaderEndEffectorConfig(BaseLeaderConfig):
    base_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    model_pose_units: List[str] = field(default_factory=lambda: [
        'm', 'm', 'm', 'radian', 'radian', 'radian', 'm',
    ])