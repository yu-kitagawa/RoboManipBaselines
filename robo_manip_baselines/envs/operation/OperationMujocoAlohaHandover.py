import gymnasium as gym


class OperationMujocoAlohaHandover:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoAlohaHandoverEnv-v0", render_mode=render_mode
        )
