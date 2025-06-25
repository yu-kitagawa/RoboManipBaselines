from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eInsertEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_insert.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.55 * np.pi,
                    -0.45 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_hole_pos = self.model.body("hole").pos.copy()
        self.hole_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )  # [m]

    def _get_reward(self):
        peg_pos = self.data.body("peg").xpos.copy()
        hole_pos = self.data.body("hole").xpos.copy()

        xy_thre = 0.01  # [m]
        z_thre = hole_pos[2] + 0.05  # [m]
        if (np.max(np.abs(peg_pos[:2] - hole_pos[:2])) < xy_thre) and (
            peg_pos[2] < z_thre
        ):
            return 1.0
        else:
            return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.hole_pos_offsets)

        hole_pos = self.original_hole_pos + self.hole_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            hole_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("hole").pos = hole_pos

        return world_idx
