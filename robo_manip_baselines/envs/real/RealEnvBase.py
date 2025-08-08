import concurrent.futures
import os
import re
import time
from abc import ABC, abstractmethod
import sys

import cv2
import gymnasium as gym
import numpy as np
import open3d as o3d
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../real/pyorbbecsdk"))
from pyorbbecsdk import *
from pyorbbecsdk.utils import frame_to_bgr_image


class RealEnvBase(EnvDataMixin, gym.Env, ABC):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        **kwargs,
    ):
        # Setup environment parameters
        self.init_time = time.time()
        self.dt = 0.02  # [s]
        self.world_random_scale = None

    def setup_realsense(self, camera_ids):
        self.cameras = {}
        detected_camera_ids = get_device_ids()
        for camera_name, camera_id in camera_ids.items():
            if camera_id not in detected_camera_ids:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}"
                )

            camera = RealSenseCamera(device_id=camera_id, flip=False)
            frames = camera._pipeline.wait_for_frames()
            color_intrinsics = (
                frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.color_fovy = np.rad2deg(
                2 * np.arctan(color_intrinsics.height / (2 * color_intrinsics.fy))
            )
            depth_intrinsics = (
                frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.depth_fovy = np.rad2deg(
                2 * np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy))
            )

            self.cameras[camera_name] = camera

    def setup_gelsight(self, gelsight_ids):
        self.rgb_tactiles = {}

        if gelsight_ids is None:
            return

        for rgb_tactile_name, gelsight_id in gelsight_ids.items():
            for device_name in os.listdir("/sys/class/video4linux"):
                real_device_name = os.path.realpath(
                    "/sys/class/video4linux/" + device_name + "/name"
                )
                with (
                    open(real_device_name, "rt") as device_name_file
                ):  # "rt": read-text mode ("t" is default, so "r" alone is the same)
                    detected_gelsight_id = device_name_file.read().rstrip()
                if gelsight_id in detected_gelsight_id:
                    tactile_num = int(re.search("\d+$", device_name).group(0))
                    print(
                        f"[{self.__class__.__name__}] Found GelSight sensor. ID: {detected_gelsight_id}, device: {device_name}, num: {tactile_num}"
                    )

                    rgb_tactile = cv2.VideoCapture(tactile_num)
                    if rgb_tactile is None or not rgb_tactile.isOpened():
                        print(
                            f"[{self.__class__.__name__}] Unable to open video source of GelSight sensor."
                        )
                        continue

                    self.rgb_tactiles[rgb_tactile_name] = rgb_tactile
                    break

            if rgb_tactile_name not in self.rgb_tactiles:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified GelSight (name: {rgb_tactile_name}, ID: {gelsight_id}) not detected."
                )
    
    def setup_femtobolt(self):

        context = Context()
        self.pipeline = Pipeline()
        config = Config()
        self.temporal_filter = TemporalFilter(alpha=0.5)
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            print("No proper depth profile, can not generate point cloud")
            return
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        self.has_color_sensor = False
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                self.has_color_sensor = True
        except OBError as e:
            print(e)
        self.pipeline.enable_frame_sync()
        self.pipeline.start(config)
        camera_param = self.pipeline.get_camera_param()
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.point_cloud_filter = PointCloudFilter()
        self.point_cloud_filter.set_camera_param(camera_param)

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def reset(self, *, seed=None, options=None):
        self.init_time = time.time()

        super().reset(seed=seed)

        self._reset_robot()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._set_action(action, duration=self.dt, joint_vel_limit_scale=2.0, wait=True)

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        pass

    @abstractmethod
    def _reset_robot(self):
        pass

    @abstractmethod
    def _set_action(self):
        pass

    def overwrite_command_for_safety(self, action, duration, joint_vel_limit_scale):
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        scaled_joint_vel_limit = (
            np.clip(joint_vel_limit_scale, 0.01, 10.0) * self.joint_vel_limit
        )

        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_joint_pos_command - self.arm_joint_pos_actual)
                    / scaled_joint_vel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            arm_joint_pos_error_max = np.max(
                np.abs(arm_joint_pos_command - self.arm_joint_pos_actual)
            )
            arm_joint_pos_error_thre = np.deg2rad(90)
            duration_thre = 0.1  # [s]
            if (
                arm_joint_pos_error_max > arm_joint_pos_error_thre
                and duration < duration_thre
            ):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Large joint movements are commanded in short duration ({duration} s).\n  command: {arm_joint_pos_command}\n  actual: {self.arm_joint_pos_actual}"
                )

            arm_joint_pos_command_overwritten = self.arm_joint_pos_actual + np.clip(
                arm_joint_pos_command - self.arm_joint_pos_actual,
                -1 * scaled_joint_vel_limit * duration,
                scaled_joint_vel_limit * duration,
            )
            # if np.linalg.norm(arm_joint_pos_command_overwritten - arm_joint_pos_command) > 1e-10:
            #     print(f"[{self.__class__.__name__}] Overwrite joint command for safety.")
            action[self.body_config_list[0].arm_joint_idxes] = (
                arm_joint_pos_command_overwritten
            )

        return action, duration

    @abstractmethod
    def _get_obs(self):
        pass

    def _get_info(self):
        info = {}

        if len(self.camera_names) + len(self.rgb_tactile_names) == 0:
            return info

        # Get images
        info["rgb_images"] = {}
        info["depth_images"] = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for camera_name, camera in self.cameras.items():
                futures[executor.submit(self.get_camera_image, camera_name, camera)] = (
                    camera_name
                )

            rgb_3dcamera_name = "femto"
            futures[executor.submit(self.get_image_pointcloud, rgb_3dcamera_name)] = (
                rgb_3dcamera_name
            )

            for rgb_tactile_name, rgb_tactile in self.rgb_tactiles.items():
                futures[
                    executor.submit(
                        self.get_rgb_tactile_image, rgb_tactile_name, rgb_tactile
                    )
                ] = rgb_tactile_name

            for future in concurrent.futures.as_completed(futures):
                name, rgb_image, depth_image, points = future.result()
                info["rgb_images"][name] = rgb_image
                info["depth_images"][name] = depth_image
                if points is not None:
                    info["point_cloud"] = points

        return info

    def get_camera_image(self, camera_name, camera):
        rgb_image, depth_image = camera.read((640, 480))
        depth_image = (1e-3 * depth_image[:, :, 0]).astype(np.float32)  # [m]
        return camera_name, rgb_image, depth_image

    def get_rgb_tactile_image(self, rgb_tactile_name, rgb_tactile):
        ret, rgb_image = rgb_tactile.read()
        if not ret:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to read tactile image."
            )
        image_size = (640, 480)
        rgb_image = cv2.resize(rgb_image, image_size)
        return rgb_tactile_name, rgb_image, None
    
    def get_image_pointcloud(self, rgb_3dcamera_name):
        frames = self.pipeline.wait_for_frames(100)
        depth_frame = frames.get_depth_frame()
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > 20) & (depth_data < 10000), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)
        depth_image = self.temporal_filter.process(depth_data)
        color_frame = frames.get_color_frame()
        color_image = frame_to_bgr_image(color_frame)
        frame = self.align_filter.process(frames)
        scale = depth_frame.get_depth_scale()
        self.point_cloud_filter.set_position_data_scaled(scale)

        self.point_cloud_filter.set_create_point_format(
            OBFormat.RGB_POINT if self.has_color_sensor and color_frame is not None else OBFormat.POINT)
        point_cloud_frame = self.point_cloud_filter.process(frame)
        points = np.array(self.point_cloud_filter.calculate(point_cloud_frame))
        return rgb_3dcamera_name, color_image, depth_image, points

    def get_joint_pos_from_obs(self, obs):
        """Get joint position from observation."""
        return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs):
        """Get joint velocity from observation."""
        return obs["joint_vel"]

    def get_gripper_joint_pos_from_obs(self, obs):
        """Get gripper joint position from observation."""
        joint_pos = self.get_joint_pos_from_obs(obs)
        gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self)
        )

        for body_config in self.body_config_list:
            if not isinstance(body_config, ArmConfig):
                continue

            gripper_joint_pos[body_config.gripper_joint_idxes_in_gripper_joint_pos] = (
                joint_pos[body_config.gripper_joint_idxes]
            )

        return gripper_joint_pos

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_time(self):
        """Get real-world time. [s]"""
        return time.time() - self.init_time

    @property
    def camera_names(self):
        """Get camera names."""
        return list(self.cameras.keys())

    @property
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return list(self.rgb_tactiles.keys())

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.cameras[camera_name].depth_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError(
            f"[{self.__class__.__name__}] modify_world is not implemented."
        )

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # In a real-world environment, it is not possible to programmatically draw markers
        pass
