import numpy as np


class DataKey:
    """Data key."""

    # Time [s]
    TIME = "time"

    # Reward
    REWARD = "reward"

    # Measured joint position (including both arm and gripper)
    MEASURED_JOINT_POS = "measured_joint_pos"
    # Command joint position (including both arm and gripper)
    COMMAND_JOINT_POS = "command_joint_pos"

    # Measured joint position relative to previous step (including both arm and gripper)
    MEASURED_JOINT_POS_REL = "measured_joint_pos_rel"
    # Command joint position relative to previous step (including both arm and gripper)
    COMMAND_JOINT_POS_REL = "command_joint_pos_rel"

    # Measured joint velocity
    MEASURED_JOINT_VEL = "measured_joint_vel"
    # Command joint velocity
    COMMAND_JOINT_VEL = "command_joint_vel"

    # Measured joint torque
    MEASURED_JOINT_TORQUE = "measured_joint_torque"
    # Command joint torque
    COMMAND_JOINT_TORQUE = "command_joint_torque"

    # Measured gripper joint position
    MEASURED_GRIPPER_JOINT_POS = "measured_gripper_joint_pos"
    # Command gripper joint position
    COMMAND_GRIPPER_JOINT_POS = "command_gripper_joint_pos"

    # Measured end-effector pose (tx, ty, tz, qw, qx, qy, qz)
    # Note: This is the end-effector pose corresponding to the measured joint position.
    MEASURED_EEF_POSE = "measured_eef_pose"
    # Command end-effector pose (tx, ty, tz, qw, qx, qy, qz)
    # Note: This is the target end-effector pose for IK, not the end-effector pose corresponding to the command joint position.
    COMMAND_EEF_POSE = "command_eef_pose"

    # Measured end-effector pose relative to previous step in the previous pose frame (tx, ty, tz, roll, pitch, yaw)
    MEASURED_EEF_POSE_REL = "measured_eef_pose_rel"
    # Command end-effector pose relative to previous step in the previous pose frame (tx, ty, tz, roll, pitch, yaw)
    COMMAND_EEF_POSE_REL = "command_eef_pose_rel"

    # Measured end-effector velocity (vx, vy, vz, wx, wy, wz)
    MEASURED_EEF_VEL = "measured_eef_vel"
    # Command end-effector velocity (vx, vy, vz, wx, wy, wz)
    COMMAND_EEF_VEL = "command_eef_vel"

    # Measured end-effector wrench (fx, fy, fz, nx, ny, nz)
    MEASURED_EEF_WRENCH = "measured_eef_wrench"
    # Command end-effector wrench (fx, fy, fz, nx, ny, nz)
    COMMAND_EEF_WRENCH = "command_eef_wrench"

    # Measured velocity of omni-directional mobile base
    MEASURED_MOBILE_OMNI_VEL = "measured_mobile_omni_vel"
    # Command velocity of omni-directional mobile base
    COMMAND_MOBILE_OMNI_VEL = "command_mobile_omni_vel"

    # All keys of measured data
    MEASURED_DATA_KEYS = [
        MEASURED_JOINT_POS,
        MEASURED_JOINT_POS_REL,
        MEASURED_JOINT_VEL,
        # MEASURED_JOINT_TORQUE,
        MEASURED_GRIPPER_JOINT_POS,
        MEASURED_EEF_POSE,
        MEASURED_EEF_POSE_REL,
        # MEASURED_EEF_VEL,
        MEASURED_EEF_WRENCH,
        MEASURED_MOBILE_OMNI_VEL,
    ]

    # All keys of command data
    COMMAND_DATA_KEYS = [
        COMMAND_JOINT_POS,
        COMMAND_JOINT_POS_REL,
        # COMMAND_JOINT_VEL,
        # COMMAND_JOINT_TORQUE,
        COMMAND_GRIPPER_JOINT_POS,
        COMMAND_EEF_POSE,
        COMMAND_EEF_POSE_REL,
        # COMMAND_EEF_VEL,
        # COMMAND_EEF_WRENCH,
        COMMAND_MOBILE_OMNI_VEL,
    ]

    @classmethod
    def get_dim(cls, key, env):
        """Get the dimension of the data specified by key."""
        from ..body.ArmManager import ArmConfig

        if key == DataKey.TIME:
            return 1
        elif key in (
            DataKey.MEASURED_JOINT_POS,
            DataKey.COMMAND_JOINT_POS,
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.COMMAND_JOINT_VEL,
            DataKey.MEASURED_JOINT_TORQUE,
            DataKey.COMMAND_JOINT_TORQUE,
        ):
            return sum(
                len(body_config.arm_joint_idxes) + len(body_config.gripper_joint_idxes)
                for body_config in env.unwrapped.body_config_list
                if isinstance(body_config, ArmConfig)
            )
        elif key in (
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
        ):
            return sum(
                len(body_config.gripper_joint_idxes)
                for body_config in env.unwrapped.body_config_list
                if isinstance(body_config, ArmConfig)
            )
        elif key in (
            DataKey.MEASURED_EEF_POSE,
            DataKey.COMMAND_EEF_POSE,
            DataKey.MEASURED_EEF_POSE_REL,
            DataKey.COMMAND_EEF_POSE_REL,
            DataKey.MEASURED_EEF_VEL,
            DataKey.COMMAND_EEF_VEL,
            DataKey.MEASURED_EEF_WRENCH,
            DataKey.COMMAND_EEF_WRENCH,
        ):
            num_eef = len(
                [
                    body_config
                    for body_config in env.unwrapped.body_config_list
                    if isinstance(body_config, ArmConfig)
                    and (body_config.eef_idx is not None)
                ]
            )

            if key in (DataKey.MEASURED_EEF_POSE, DataKey.COMMAND_EEF_POSE):
                return 7 * num_eef
            else:
                return 6 * num_eef
        elif key in (DataKey.MEASURED_MOBILE_OMNI_VEL, DataKey.COMMAND_MOBILE_OMNI_VEL):
            return 3
        else:
            raise ValueError(f"[{cls.__name__}] Invalid data key: {key}")

    @classmethod
    def get_measured_key(cls, key):
        """Get measured key."""
        if key.startswith("measured_"):
            return key
        elif key.startswith("command_"):
            return "measured_" + key[len("command_") :]
        else:
            raise ValueError(f"[{cls.__name__}] Invalid data key: {key}")

    @classmethod
    def get_command_key(cls, key):
        """Get command key."""
        if key.startswith("command_"):
            return key
        elif key.startswith("measured_"):
            return "command_" + key[len("measured_") :]
        else:
            raise ValueError(f"[{cls.__name__}] Invalid data key: {key}")

    @classmethod
    def get_rel_key(cls, key):
        """Get the relative key corresponding to the absolute key."""
        if key == DataKey.MEASURED_JOINT_POS:
            return DataKey.MEASURED_JOINT_POS_REL
        elif key == DataKey.COMMAND_JOINT_POS:
            return DataKey.COMMAND_JOINT_POS_REL
        elif key == DataKey.MEASURED_EEF_POSE:
            return DataKey.MEASURED_EEF_POSE_REL
        elif key == DataKey.COMMAND_EEF_POSE:
            return DataKey.COMMAND_EEF_POSE_REL
        else:
            raise ValueError(f"[{cls.__name__}] Relative data key not found: {key}")

    @classmethod
    def get_abs_key(cls, key):
        """Get the absolute key corresponding to the relative key."""
        if key == DataKey.MEASURED_JOINT_POS_REL:
            return DataKey.MEASURED_JOINT_POS
        elif key == DataKey.COMMAND_JOINT_POS_REL:
            return DataKey.COMMAND_JOINT_POS
        elif key == DataKey.MEASURED_EEF_POSE_REL:
            return DataKey.MEASURED_EEF_POSE
        elif key == DataKey.COMMAND_EEF_POSE_REL:
            return DataKey.COMMAND_EEF_POSE
        else:
            raise ValueError(f"[{cls.__name__}] Absolute data key not found: {key}")

    @classmethod
    def get_rgb_image_key(cls, camera_name):
        """Get the rgb image key from the camera name."""
        return camera_name.lower() + "_rgb_image"

    @classmethod
    def get_depth_image_key(cls, camera_name):
        """Get the depth image key from the camera name."""
        return camera_name.lower() + "_depth_image"

    @classmethod
    def get_camera_name(cls, key):
        if cls.is_rgb_image_key(key):
            return key[: -len("_rgb_image")]
        elif cls.is_depth_image_key(key):
            return key[: -len("_depth_image")]
        else:
            raise ValueError(f"[{cls.__name__}] Not rgb or depth key: {key}")

    @classmethod
    def is_rgb_image_key(cls, key):
        """Check if the key is for RGB image."""
        return key.endswith("_rgb_image")

    @classmethod
    def is_depth_image_key(cls, key):
        """Check if the key is for depth image."""
        return key.endswith("_depth_image")

    @classmethod
    def get_plot_scale(cls, key, env):
        """Get scale to plot data."""
        from ..body.ArmManager import ArmConfig

        if key in (
            DataKey.MEASURED_JOINT_POS,
            DataKey.COMMAND_JOINT_POS,
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
        ):
            scale_arr = np.zeros(cls.get_dim(key, env))

            for body_config in env.unwrapped.body_config_list:
                if not isinstance(body_config, ArmConfig):
                    continue

                scale_arr[body_config.arm_joint_idxes] = 1.0
                scale_arr[body_config.gripper_joint_idxes] = 0.01

            return scale_arr
        elif key in (
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
        ):
            return np.full(cls.get_dim(key, env), 0.01)
        elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
            return np.full(cls.get_dim(key, env), 100.0)
        else:
            return np.ones(cls.get_dim(key, env))
