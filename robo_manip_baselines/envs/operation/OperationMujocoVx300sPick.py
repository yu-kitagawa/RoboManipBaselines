import gymnasium as gym


class OperationMujocoVx300sPick:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoVx300sPickEnv-v0", render_mode=render_mode
        )
