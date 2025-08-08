from os import path

import mujoco
import numpy as np

from .MujocoAlohaEnvBase import MujocoAlohaEnvBase


class MujocoAlohaHandoverEnv(MujocoAlohaEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoAlohaEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/aloha/env_aloha_handover.xml",
            ),
            np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.037, 0.037] * 2),
            **kwargs,
        )

        self.original_obj_pos = self.model.body("obj").pos.copy()
        self.original_mat1_pos = self.model.body("mat1").pos.copy()
        self.original_mat2_pos = self.model.body("mat2").pos.copy()
        self.obj_pos_offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.02, 0.0],
                [0.0, -0.04, 0.0],
                [0.0, -0.06, 0.0],
                [0.0, -0.08, 0.0],
                [0.0, -0.10, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.obj_pos_offsets)

        delta_pos = self.obj_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

        obj_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint"
        )
        obj_qpos_addr = self.model.jnt_qposadr[obj_joint_id]
        self.init_qpos[obj_qpos_addr : obj_qpos_addr + 3] = (
            self.original_obj_pos + delta_pos
        )

        self.model.body("mat1").pos = self.original_mat1_pos + delta_pos
        self.model.body("mat2").pos = self.original_mat2_pos - delta_pos

        return world_idx
