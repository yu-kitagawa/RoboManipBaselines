import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, delta_pos_z):
    target_pos = op.env.unwrapped.get_link_pose("chain_end", "box")[0:3]
    target_pos[2] += delta_pos_z
    return pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            delta_pos_z=0.22,  # [m]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            delta_pos_z=0.14,  # [m]
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([150.0])
        self.duration = 0.5  # [s]


class OperationIsaacUR5eChain:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eChainEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
            GraspPhase(self),
        ]
