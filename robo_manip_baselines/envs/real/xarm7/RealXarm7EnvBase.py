from os import path
import time
import numpy as np
from gymnasium.spaces import Box, Dict

from ..RealEnvBase import RealEnvBase

from xarm.wrapper import XArmAPI


class RealXarm7EnvBase(RealEnvBase):
    action_space = Box(
        low=np.deg2rad(
            [
                -2 * np.pi,
                np.deg2rad(-118),
                -2 * np.pi,
                np.deg2rad(-11),
                -2 * np.pi,
                np.deg2rad(-97),
                -2 * np.pi,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                np.deg2rad(120),
                2 * np.pi,
                np.deg2rad(225),
                2 * np.pi,
                np.pi,
                2 * np.pi,
                840.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.gripper_action_idx = 7
        self.arm_action_idxes = slice(0, 7)
        self.arm_urdf_path = path.join(
            path.dirname(__file__), "../../assets/common/robots/xarm7/xarm7.urdf"
        )
        self.arm_root_pose = None
        self.ik_eef_joint_id = 7
        self.ik_arm_joint_ids = slice(0, 7)
        self.init_qpos = init_qpos
        self.qvel_limit = np.deg2rad(180)  # [rad/s]

        # Connect to xArm7
        print("[RealXarm7EnvBase] Start connecting the xArm7.")
        self.robot_ip = robot_ip
        self.xarm_api = XArmAPI(self.robot_ip)
        self.xarm_api.connect()
        self.xarm_api.motion_enable(enable=True)
        self.xarm_api.clean_error()
        self.xarm_api.set_mode(6)
        self.xarm_api.set_state(0)
        self.xarm_api.set_collision_sensitivity(1)
        self.xarm_api.clean_gripper_error()
        self.xarm_api.set_gripper_mode(0)
        self.xarm_api.set_gripper_enable(True)
        self.xarm_api.ft_sensor_enable(1)
        time.sleep(0.5)
        self.xarm_api.ft_sensor_set_zero()
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        self.arm_qpos_actual = joint_states[0]
        print("[RealXarm7EnvBase] Finish connecting the xArm7.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)

    def close(self):
        self.xarm_api.disconnect()

    def _reset_robot(self):
        print("[RealXarm7EnvBase] Start moving the robot to the reset position.")
        self._set_action(self.init_qpos, duration=None, qvel_limit_scale=0.1, wait=True)
        print("[RealXarm7EnvBase] Finish moving the robot to the reset position.")

    def _set_action(self, action, duration=None, qvel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or qpos for safety
        arm_qpos_command = action[self.arm_action_idxes]
        scaled_qvel_limit = np.clip(qvel_limit_scale, 0.01, 10.0) * self.qvel_limit
        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_qpos_command - self.arm_qpos_actual) / scaled_qvel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            arm_qpos_command_overwritten = self.arm_qpos_actual + np.clip(
                arm_qpos_command - self.arm_qpos_actual,
                -1 * scaled_qvel_limit * duration,
                scaled_qvel_limit * duration,
            )
            # if np.linalg.norm(arm_qpos_command_overwritten - arm_qpos_command) > 1e-10:
            #     print("[RealXarm7EnvBase] Overwrite joint command for safety.")
            arm_qpos_command = arm_qpos_command_overwritten

        # Send command to xArm7
        xarm_code = self.xarm_api.set_servo_angle(
            angle=arm_qpos_command,
            speed=scaled_qvel_limit,
            mvtime=duration,
            is_radian=True,
            wait=False,
        )
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")

        # Send command to xArm gripper
        gripper_pos = action[self.gripper_action_idx]
        xarm_code = self.xarm_api.set_gripper_position(gripper_pos, wait=False)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from xArm7
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        arm_qpos = joint_states[0]
        arm_qvel = joint_states[1]
        self.arm_qpos_actual = arm_qpos.copy()

        # Get state from Robotiq gripper
        xarm_code, gripper_pos = self.xarm_api.get_gripper_position()
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        gripper_pos = np.array([gripper_pos], dtype=np.float64)
        gripper_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.array(self.xarm_api.get_ft_sensor_data()[1], dtype=np.float64)
        force = wrench[0:3]
        torque = wrench[3:6]

        return {
            "joint_pos": np.concatenate((arm_qpos, gripper_pos), dtype=np.float64),
            "joint_vel": np.concatenate((arm_qvel, gripper_vel), dtype=np.float64),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }
