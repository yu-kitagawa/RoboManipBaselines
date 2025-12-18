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
                [0.0, -1*0.10/9, 0.0],
                [0.0, -2*0.10/9, 0.0],
                [0.0, -3*0.10/9, 0.0],
                [0.0, -4*0.10/9, 0.0],
                [0.0, -5*0.10/9, 0.0],
                [0.0, -6*0.10/9, 0.0],
                [0.0, -7*0.10/9, 0.0],
                [0.0, -8*0.10/9, 0.0],
                [0.0, -0.10, 0.0],
                [0.0, 1*0.10/9, 0.0],
                [0.0, 2*0.10/9, 0.0],
                [0.0, 3*0.10/9, 0.0],
                [-0.10/9, -4*0.10/9, 0.0],
                [0.10/9, -4*0.10/9, 0.0],
                [-0.10/9, -5*0.10/9, 0.0],
                [0.10/9, -5*0.10/9, 0.0],
                [0.0, -10*0.10/9, 0.0],
                [0.0, -11*0.10/9, 0.0],
                [0.0, -12*0.10/9, 0.0],
            ]
        )  # [m]

    def _get_reward(self):
        obj_base_pos = self.data.geom("obj_base").xpos.copy()
        obj_handle_pos = self.data.geom("obj_handle").xpos.copy()
        mat2_pos = self.data.body("mat2").xpos.copy()
        mat2_half_extents = np.array([0.1, 0.1, 0.08])  # [m]
        right_gripper_pos = self.data.site("right/gripper").xpos.copy()
        left_gripper_pos = self.data.site("left/gripper").xpos.copy()
        grasp_thre = 0.1  # [m]

        reward = 0.0
        if np.all(np.abs(obj_base_pos - mat2_pos) <= mat2_half_extents):
            reward = 1.0
        elif np.linalg.norm(obj_handle_pos - right_gripper_pos) < grasp_thre:
            reward = 0.5
        elif np.linalg.norm(obj_handle_pos - left_gripper_pos) < grasp_thre:
            reward = 0.2

        return reward

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
