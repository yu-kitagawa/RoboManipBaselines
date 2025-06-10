import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, offset_pose):
    base_pose = pin.SE3(
        pin.rpy.rpyToMatrix(np.pi / 2, 0.0, 0.25 * np.pi),
        op.env.unwrapped.get_body_pose("board")[0:3],
    )
    return base_pose * offset_pose


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            offset_pose=pin.SE3(
                pin.rpy.rpyToMatrix(0.0, 0.125 * np.pi, 0.0),
                np.array([0.0, -0.2, -0.4]),
            ),
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op, offset_pose=pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.3]))
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_open()


class OperationMujocoUR5eCloth:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eClothEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
