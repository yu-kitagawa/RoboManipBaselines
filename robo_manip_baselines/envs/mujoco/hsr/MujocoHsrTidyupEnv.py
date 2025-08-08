from os import path

import mujoco
import numpy as np

from .MujocoHsrEnvBase import MujocoHsrEnvBase


class MujocoHsrTidyupEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_tidyup.xml",
            ),
            np.array([0.0] * 3 + [0.25, -2.0, 0.0, -1.0, 0.0, 0.8]),
            **kwargs,
        )

        self.original_bottle1_pos = self.model.body("bottle1").pos.copy()
        self.original_bottle2_pos = self.model.body("bottle2").pos.copy()
        self.bottle_pos_offsets = np.array(
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
            world_idx = cumulative_idx % len(self.bottle_pos_offsets)

        delta_pos = self.bottle_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

        for bottle_name in ["bottle1", "bottle2"]:
            bottle_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{bottle_name}_freejoint"
            )
            bottle_qpos_addr = self.model.jnt_qposadr[bottle_joint_id]
            if bottle_name == "bottle1":
                original_bottle_pos = self.original_bottle1_pos
            else:  # if bottle_name == "bottle2"
                original_bottle_pos = self.original_bottle2_pos
            self.init_qpos[bottle_qpos_addr : bottle_qpos_addr + 3] = (
                original_bottle_pos + delta_pos
            )

        return world_idx
