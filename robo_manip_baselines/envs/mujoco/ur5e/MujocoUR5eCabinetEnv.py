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

        self.target_task = None  # One of [None, "hinge", "slide"]

    def _get_reward(self):
        hinge_thre = 120.0  # [deg]
        hinge_success = self.data.joint("hinge").qpos[0] > np.deg2rad(hinge_thre)
        slide_thre = 0.12  # [m]
        slide_success = self.data.joint("slide").qpos[0] > slide_thre
        if self.target_task is None:
            return 1.0 if hinge_success or slide_success else 0.0
        elif self.target_task == "hinge":
            return 1.0 if hinge_success else 0.0
        elif self.target_task == "slide":
            return 1.0 if slide_success else 0.0
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid target task: {self.target_task}"
            )

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
