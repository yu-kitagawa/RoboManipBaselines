from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5ePickEnv(MujocoUR5eEnvBase):
    sim_timestep = 0.002
    frame_skip = 16
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (sim_timestep * frame_skip)),
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_pick.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.55 * np.pi,
                    -0.45 * np.pi,
                    np.pi / 2,
                    np.pi,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        # self.original_pole_pos = self.model.body("pole").pos.copy()
        # self.pole_pos_offsets = np.array(
        #     [
        #         [0.0, 0.0, 0.0],
        #         [0.0, 0.04, 0.0],
        #         [0.0, 0.08, 0.0],
        #         [0.0, 0.12, 0.0],
        #         [0.0, 0.16, 0.0],
        #         [0.0, 0.20, 0.0],
        #     ]
        # )  # [m]

    # def _get_success(self):
    #     pass

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            # world_idx = cumulative_idx % len(self.pole_pos_offsets)
            world_idx = 0

        # pole_pos = self.original_pole_pos + self.pole_pos_offsets[world_idx]
        # if self.world_random_scale is not None:
        #     pole_pos += np.random.uniform(
        #         low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
        #     )
        # self.model.body("pole").pos = pole_pos

        return world_idx
