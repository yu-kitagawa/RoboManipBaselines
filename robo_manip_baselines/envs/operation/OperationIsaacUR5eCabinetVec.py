import gymnasium as gym

from .OperationIsaacUR5eCabinet import OperationIsaacUR5eCabinet


class OperationIsaacUR5eCabinetVec(OperationIsaacUR5eCabinet):
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eCabinetEnv-v0",
            num_envs=12,
            render_mode=render_mode,
        )
