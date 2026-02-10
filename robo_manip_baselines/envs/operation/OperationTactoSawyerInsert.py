import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


class GraspPhase1(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.04])
        self.duration = 0.2


def get_target_se3(op):
    se3 = op.motion_manager.body_manager_list[0].current_se3
    se3.translation[2] -= 0.08
    return pin.SE3(se3.rotation, se3.translation)


class ReachPhase(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
        )
        self.duration = 0.4  # [s]


class GraspPhase2(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.018])
        self.duration = 0.2


class OperationTactoSawyerInsert:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/TactoSawyerInsertEnv-v0",
            render_mode=render_mode,
        )

    def get_pre_motion_phases(self):
        return [
            GraspPhase1(self),
            ReachPhase(self),
            GraspPhase2(self),
        ]
