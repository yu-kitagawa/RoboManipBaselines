import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([170.0])
        self.duration = 0.5  # [s]


class OperationMujocoUR5eInsert:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eInsertEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
