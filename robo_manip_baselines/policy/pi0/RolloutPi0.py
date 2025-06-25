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
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../work/lerobot"))
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

from robo_manip_baselines.common import RolloutBase, denormalize_data
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutPi0(RolloutBase):
    require_task_desc = True

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {8}")

        self.pi0 = PI0Policy.from_pretrained(self.args.checkpoint)

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

        if self.args.win_xy_plot is not None:
            plt.get_current_fig_manager().window.wm_geometry("+20+50")

        if len(self.action_keys) > 0:
            self.action_plot_scale = np.concatenate(
                [DataKey.get_plot_scale(key, self.env) for key in self.action_keys]
            )
        else:
            self.action_plot_scale = np.zeros(0)

    def setup_model_meta_info(self):
        cmd_args = " ".join(sys.argv).lower()
        self.state_keys = ["measured_joint_pos"]
        self.action_keys = ["command_joint_pos"]
        if "aloha" in cmd_args:
            self.camera_names = ["overhead_cam", "wrist_cam_left", "wrist_cam_right"]
            self.state_dim = 14
            self.action_dim = 14
        elif "ur5e" in cmd_args:
            self.camera_names = ["front", "side", "hand"]
            self.state_dim = 7
            self.action_dim = 7

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def infer_policy(self):
        # Infer

        img = {}
        state = self.get_state()
        state = state[np.newaxis]
        state = torch.from_numpy(state.copy()).to("cuda:0")
        state = state.type(torch.float32)
        for camera_name in self.camera_names:
            tmp = self.info["rgb_images"][camera_name][np.newaxis].transpose(0, 3, 1, 2)
            tmp = torch.from_numpy(tmp.copy()).to("cuda:0")
            tmp = tmp.type(torch.float32)
            tmp /= 255
            img[camera_name] = tmp

        # images = self.get_images()
        # images[0] = images[0][np.newaxis]

        observation = {
            "observation.images.front_rgb": img["front"],
            "observation.images.side_rgb": img["side"],
            "observation.images.hand_rgb": img["hand"],
            "observation.state": state,
            "task": [self.args.task_desc],
        }

        action = self.pi0.select_action(observation)
        action = torch.squeeze(action)

        self.policy_action = action.cpu().detach().numpy().astype(np.float64)
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
        self.reset_flag = True
        self.quit_flag = False
        self.inference_duration_list = []

        self.motion_manager.reset()

        self.obs, self.info = self.env.reset(seed=self.args.seed)

        self.time = 0
        self.key = 0

        while True:
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )
            self.obs, self.reward, self.terminated, _, self.info = self.env.step(
                env_action
            )

            self.phase_manager.post_update()

            self.time += 1
            self.phase_manager.check_transition()

            if self.quit_flag:
                break

        if self.args.result_filename is not None:
            print(
                f"[{self.__class__.__name__}] Save the rollout results: {self.args.result_filename}"
            )
            with open(self.args.result_filename, "w") as result_file:
                yaml.dump(self.result, result_file)
