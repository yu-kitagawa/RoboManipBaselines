import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, offset_pos):
    target_pos = (
        0.5
        * (
            op.env.unwrapped.get_geom_pose("fook1")[0:3]
            + op.env.unwrapped.get_geom_pose("fook2")[0:3]
        )
        + offset_pos
    )
    return pin.SE3(pin.rpy.rpyToMatrix(0.0, 1.5 * np.pi, np.pi), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            offset_pos=np.array([-0.2, 0.05, -0.05]),  # [m]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            offset_pos=np.array([-0.15, 0.05, -0.05]),  # [m]
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_close()


class OperationMujocoXarm7Ring:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoXarm7RingEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
