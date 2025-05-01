import os
import sys
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pylab as plt
import numpy as np
from dataclasses import dataclass
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../openvla/openvla-oft"))
import pickle
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action

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
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.model_meta_info['data']['chunk_size']}")

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

    def setup_variables(self):
        super().setup_variables()

        self.all_actions_history = []

    def infer_policy(self):
        # Infer
        # Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
        cfg = DeployConfig(
            pretrained_checkpoint = "../../../openvla/openvla-oft/log/openvla-7b+aloha_sample+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_default--90000_chkpt",
            use_l1_regression = True,
            use_diffusion = False,
            use_film = True,
            num_images_in_input = 3,
            use_proprio = True,
            load_in_8bit = False,
            load_in_4bit = False,
            center_crop = True,
            num_open_loop_steps = 25,
            unnorm_key = "aloha_sample",
        )

        # Load OpenVLA-OFT policy and inputs processor
        vla = get_vla(cfg)
        processor = get_processor(cfg)

        # Load MLP action head to generate continuous actions (via L1 regression)
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

        # Load proprio projector to map proprio to language embedding space
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=14)

        self.device = torch.device("cpu")

        state = self.get_state()
        state = state.squeeze()
        images = self.get_images()

        observation = {
            "full_image": images[0],
            "state": state
        }

        # Generate robot action chunk (sequence of future actions)
        all_actions = get_vla_action(cfg, vla, processor, observation, "do something", action_head, proprio_projector)

        self.all_actions_history.append(
            all_actions.cpu().detach().numpy().astype(np.float64)
        )
        if len(self.all_actions_history) > self.model_meta_info["data"]["chunk_size"]:
            self.all_actions_history.pop(0)

        # Apply temporal ensembling to action
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(self.all_actions_history)))
        exp_weights = exp_weights / exp_weights.sum()
        action = np.zeros(self.action_dim)
        for action_idx, _all_actions in enumerate(reversed(self.all_actions_history)):
            action += exp_weights[::-1][action_idx] * _all_actions[action_idx]
        self.policy_action = denormalize_data(action, self.model_meta_info["action"])
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )
    
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

        # Draw attention images
        attention_shape = (15, 20 * len(self.camera_names))
        for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
            if layer.self_attn.correlation_mat is None:
                continue
            self.ax[1, layer_idx].imshow(
                layer.self_attn.correlation_mat[2:, 1].reshape(attention_shape)
            )
            self.ax[1, layer_idx].set_title(
                f"attention image ({layer_idx})", fontsize=20
            )

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
