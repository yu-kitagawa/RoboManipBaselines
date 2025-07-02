import os
import time

import cv2
import numpy as np

from robo_manip_baselines.common import DataKey, DataManagerVec

from .TeleopBase import TeleopBase


class TeleopBaseVec(TeleopBase):
    DataManagerClass = DataManagerVec

    def setup_args(self, parser=None, argv=None):
        super().setup_args(parser, argv)

        if self.args.replay_log is not None:
            raise NotImplementedError(
                f'[{self.__class__.__name__}] The "replay_log" option is not supported.'
            )

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        self.iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()
            self.motion_manager.draw_markers()

            action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )
            self.action_list = self.env.unwrapped.get_fluctuated_action_list(
                action, update_fluctuation=self.phase_manager.is_phase("TeleopPhase")
            )

            if self.phase_manager.is_phase("TeleopPhase"):
                self.record_data()

            self.env.unwrapped.action_list = self.action_list
            self.env.step(action)
            self.obs_list = self.env.unwrapped.obs_list
            self.info_list = self.env.unwrapped.info_list
            self.obs = self.obs_list[self.env.unwrapped.rep_env_idx]
            self.info = self.info_list[self.env.unwrapped.rep_env_idx]

            self.draw_image()

            if self.args.enable_3d_plot:
                self.draw_point_cloud()

            self.phase_manager.post_update()

            self.key = cv2.waitKey(1)
            self.phase_manager.check_transition()

            if self.key == 27:  # escape key
                self.quit_flag = True
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.phase_manager.is_phase("TeleopPhase") and self.teleop_time_idx > 0:
                self.iteration_duration_list.append(iteration_duration)

            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        self.print_statistics()

        for input_device in self.input_device_list:
            input_device.close()

        # self.env.close()

    def reset(self):
        # Reset motion manager
        self.motion_manager.reset()

        # Reset data
        self.data_manager.reset()
        if self.args.world_idx_list is None:
            world_idx = None
        else:
            world_idx = self.args.world_idx_list[
                self.data_manager.episode_idx % len(self.args.world_idx_list)
            ]

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        self.data_manager.setup_env_world(world_idx)
        self.env.reset()
        print(
            f"[{self.__class__.__name__}] Reset environment. demo_name: {self.demo_name}, world_idx: {self.data_manager.world_idx}, episode_idx: {self.data_manager.episode_idx}"
        )

        # Reset phase manager
        self.phase_manager.reset()

    def record_data(self):
        # Add time
        self.data_manager.append_single_data(
            DataKey.TIME,
            [self.phase_manager.phase.get_elapsed_duration()]
            * self.env.unwrapped.num_envs,
        )

        # Add measured data
        for key in self.env.unwrapped.measured_keys_to_save:
            self.data_manager.append_single_data(
                key,
                [
                    self.motion_manager.get_measured_data(key, obs)
                    for obs in self.obs_list
                ],
            )

        # Add command data
        # TODO: COMMAND_EEF_POSE does not reflect the effect of action fluctuation
        for key in self.env.unwrapped.command_keys_to_save:
            self.data_manager.append_single_data(
                key,
                [
                    self.motion_manager.get_command_data(key)
                    for action in self.action_list
                ],
            )

        # Add relative data
        for key in (
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_EEF_POSE_REL,
            DataKey.COMMAND_EEF_POSE_REL,
        ):
            self.data_manager.append_single_data(
                key, self.data_manager.calc_rel_data(key)
            )

        # Add image
        for camera_name in self.env.unwrapped.camera_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(camera_name),
                [info["rgb_images"][camera_name] for info in self.info_list],
            )
            self.data_manager.append_single_data(
                DataKey.get_depth_image_key(camera_name),
                [info["depth_images"][camera_name] for info in self.info_list],
            )

    def save_data(self):
        filename_list = []
        aug_idx = 0
        for env_idx, success in enumerate(self.env.unwrapped.success_list):
            if not success:
                filename_list.append(None)
                continue
            if env_idx == self.env.unwrapped.rep_env_idx:
                extra_label = "nominal"
            else:
                extra_label = f"augmented{aug_idx:0>3}"
                aug_idx += 1
            filename = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "dataset",
                    f"{self.demo_name}_{self.datetime_now:%Y%m%d_%H%M%S}",
                    f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}_{extra_label}.{self.args.file_format}",
                )
            )
            filename_list.append(filename)
        self.data_manager.save_data(filename_list)
        num_success = sum(filename is not None for filename in filename_list)
        if num_success > 0:
            print(
                "[{}] Save the {} data such as {} etc.".format(
                    self.__class__.__name__,
                    sum(filename is not None for filename in filename_list),
                    next(
                        filename for filename in filename_list if filename is not None
                    ),
                )
            )
        else:
            print(
                f"[{self.__class__.__name__}] Teleoperation succeeded: Save no data because there is no successful data."
            )
