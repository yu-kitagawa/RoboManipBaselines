from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eCabinetEnv(MujocoUR5eEnvBase):
    default_camera_config = {
        "azimuth": 45.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [-0.2, -0.2, 0.8],
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_cabinet.xml",
            ),
            np.array(
                [
                    np.pi,
                    -0.4 * np.pi,
                    -0.65 * np.pi,
                    -0.2 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_cabinet_pos = self.model.body("cabinet").pos.copy()
        self.cabinet_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.cabinet_pos_offsets)

        cabinet_pos = self.original_cabinet_pos + self.cabinet_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            cabinet_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("cabinet").pos = cabinet_pos

        return world_idx
