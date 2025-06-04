import os
import sys
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from dataclasses import dataclass
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../work/Isaac-GR00T"))
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from robo_manip_baselines.common import RolloutBase, denormalize_data
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutGr00t(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {8}")

        # get the data config
        data_config = DATA_CONFIG_MAP["ur5e"]

        # get the modality configs and transforms
        modality_config = data_config.modality_config()
        transforms = data_config.transform()

        self.gr00t = Gr00tPolicy(
            model_path=self.args.checkpoint,
            modality_config=modality_config,
            modality_transform=transforms,
            embodiment_tag="new_embodiment",
            device="cuda"
        )

        #self.device = torch.device("cpu")

        # Construct policy
        #self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])

        # Register fook to visualize attention images
        # def forward_fook(_layer, _input, _output):
        #     # Output of MultiheadAttention is a tuple (attn_output, attn_output_weights)
        #     # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        #     _layer.correlation_mat = _output[1][0].detach().cpu().numpy()

        # for layer in self.policy.model.transformer.encoder.layers:
        #     layer.self_attn.correlation_mat = None
        #     layer.self_attn.register_forward_hook(forward_fook)

        # Load checkpoint
        # self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names) + 1,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        self.fig, self.ax = fig_ax

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        plt.figure(self.policy_name)

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        plt.imshow(
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)
        )

        if self.args.win_xy_policy is not None:
            plt.get_current_fig_manager().window.wm_geometry("+20+50")

        if len(self.action_keys) > 0:
            self.action_plot_scale = np.concatenate(
                [DataKey.get_plot_scale(key, self.env) for key in self.action_keys]
            )
        else:
            self.action_plot_scale = np.zeros(0)

    def setup_variables(self):
        super().setup_variables()

    def infer_policy(self):
        # Infer

        state = self.get_state()
        state = state[np.newaxis]

        # images = self.get_images()
        # images[0] = images[0][np.newaxis]

        observation = {
            "video.front_rgb": self.info["rgb_images"]["front"][np.newaxis],
            "video.side_rgb": self.info["rgb_images"]["side"][np.newaxis],
            "video.hand_rgb": self.info["rgb_images"]["hand"][np.newaxis],
            "state.qpos": state,
            "annotation.human.action.task_description": ["open the door"]
        }

        all_actions = self.gr00t.get_action(observation)

        self.policy_action = all_actions["action.action"][0]
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def get_state(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        return state
    
    def get_images(self):
        # Assume all images are the same size
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]
            images.append(image)

        return images

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[0, len(self.camera_names)])

        plt.figure(self.policy_name)

        # Finalize plot
        self.canvas.draw()
        plt.imshow(
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)
        )

    def run(self):
        self.quit_flag = False
        self.inference_duration_list = []

        self.motion_manager.reset()

        self.obs, self.info = self.env.reset(seed=self.args.seed)

        self.time = 0
        self.key = 0

        while True:
            self.phase_manager.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )
            self.obs, _, self.terminated, _, self.info = self.env.step(env_action)

            self.phase_manager.post_update()

            self.time += 1
            self.phase_manager.check_transition()

            if self.time == 300:  # escape key
                self.quit_flag = True
            if self.quit_flag:
                break
