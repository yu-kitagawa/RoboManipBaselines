import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, delta_pos_z):
    target_pos = op.env.unwrapped.get_geom_pose("scoop_handle")[0:3]
    target_pos[2] += delta_pos_z
    return pin.SE3(pin.rpy.rpyToMatrix(np.pi, 0.0, np.pi / 2), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            delta_pos_z=0.2,  # [m]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            delta_pos_z=0.15,  # [m]
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_close()


class OperationMujocoUR5eParticle:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eParticleEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
