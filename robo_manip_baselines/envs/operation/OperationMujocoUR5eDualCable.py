import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import (
    GraspPhaseBase,
    ReachPhaseBase,
    get_pose_from_se3,
)


def get_target_se3(op, pos_z):
    left_target_pose = get_pose_from_se3(
        op.motion_manager.body_manager_list[0].current_se3
    )

    right_target_pos = op.env.unwrapped.get_body_pose("cable_end")[0:3]
    right_target_pos[2] = pos_z
    right_target_pose = get_pose_from_se3(
        pin.SE3(pin.rpy.rpyToMatrix(0.0, np.pi, np.pi), right_target_pos)
    )

    return np.concatenate([left_target_pose, right_target_pose])


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=1.02,  # [m]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=0.99,  # [m]
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0, 255.0])
        self.duration = 0.5  # [s]


class OperationMujocoUR5eDualCable:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eDualCableEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
