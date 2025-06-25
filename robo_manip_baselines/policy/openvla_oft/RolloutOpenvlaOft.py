import os
import sys
from pathlib import Path
from typing import Union
import json

import cv2
import matplotlib.pylab as plt
import numpy as np
from dataclasses import dataclass
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../openvla/openvla-oft"))
import pickle
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import PROPRIO_DIM

from robo_manip_baselines.common import RolloutBase, denormalize_data


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


class RolloutOpenvlaOft(RolloutBase):
    require_task_desc = True

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {8}")

        # Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
        self.cfg = DeployConfig(
            pretrained_checkpoint = self.args.checkpoint,
            use_l1_regression = True,
            use_diffusion = False,
            use_film = False,
            num_images_in_input = 2,
            use_proprio = True,
            load_in_8bit = False,
            load_in_4bit = False,
            center_crop = True,
            num_open_loop_steps = 8,
            unnorm_key = "ur5e_sample",
        )

        # Load OpenVLA-OFT policy and inputs processor
        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)

        # Load MLP action head to generate continuous actions (via L1 regression)
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)

        # Load proprio projector to map proprio to language embedding space
        self.proprio_projector = get_proprio_projector(self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)

        self.device = torch.device("cpu")

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
        super().setup_plot(fig_ax)

    def setup_model_meta_info(self):
        cmd_args = " ".join(sys.argv).lower()
        self.state_keys = ["measured_joint_pos"]
        self.action_keys = ["command_joint_pos"]
        if "aloha" in cmd_args:
            self.camera_names = ["overhead_cam", "wrist_cam_left", "wrist_cam_right"]
            self.state_dim = 14
            self.action_dim = 14
        elif "ur5e" in cmd_args:
            self.camera_names = ["front", "hand"]
            self.state_dim = 7
            self.action_dim = 7
        else:
            self.camera_names = ["front", "hand"]
            self.state_dim = 7
            self.action_dim = 7

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def infer_policy(self):
        # Infer

        state = self.get_state()
        images = self.get_images()

        observation = {
            "full_image": cv2.resize(images[0], dsize=(256, 256)),
            "left_wrist_image": cv2.resize(images[1], dsize=(256, 256)),
            #"right_wrist_image": cv2.resize(images[2], dsize=(256, 256)),
            "state": state
        }

        # Generate robot action chunk (sequence of future actions)
        all_actions = get_vla_action(self.cfg, self.vla, self.processor, observation, self.args.task_desc, self.action_head, self.proprio_projector)

        self.policy_action = all_actions.pop(0)
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
        # with open(os.path.join(self.args.checkpoint, "dataset_statistics.json"), "r") as f:
        #     stats = json.load(f)
        # data_name = str(self.args.checkpoint).split("+")
        # state = (state - stats[data_name[1]]["action"]["mean"]) / stats[data_name[1]]["action"]["std"]

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

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
