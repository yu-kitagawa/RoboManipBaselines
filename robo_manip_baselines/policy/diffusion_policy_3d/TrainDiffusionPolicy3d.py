import argparse
import copy
import os
import sys

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../third_party/3D-Diffusion-Policy/3D-Diffusion-Policy",
    )
)
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.policy.dp3 import DP3
from robo_manip_baselines.common import DataKey, RmbData, TrainBase

from .DiffusionPolicy3dDataset import DiffusionPolicy3dDataset


class TrainDiffusionPolicy3d(TrainBase):
    DatasetClass = DiffusionPolicy3dDataset

    def set_additional_args(self, parser):
        for action in parser._actions:
            if action.dest == "camera_names":
                action.nargs = 1

        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=128)
        parser.set_defaults(num_epochs=3000)
        parser.set_defaults(lr=1e-4)

        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
        )

        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable or disable exponential moving average (EMA)",
        )

        parser.add_argument(
            "--horizon", type=int, default=16, help="prediction horizon"
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of steps in the observation sequence to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=8,
            help="number of steps in the action sequence to output from the policy",
        )

        parser.add_argument(
            "--use_pc_color",
            action="store_true",
            help="Whether to use color information of point cloud",
        )

        parser.add_argument(
            "--encoder_output_dim",
            type=int,
            nargs=1,
            default=64,
            help="output dimensions of encoder in policy",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        # Retrieve point cloud information
        pc_key = DataKey.get_pointcloud_key(self.args.camera_names[0])
        num_points = None
        image_size = None
        min_bound = None
        max_bound = None
        roll_pitch_yaw = None
        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                num_points_new = rmb_data[pc_key].shape[1]
                if num_points is None:
                    num_points = num_points_new
                elif num_points != num_points_new:
                    raise ValueError(
                        f"[{self.__class__.__name__}] num_points is inconsistent in dataset: {num_points} != {num_points_new}"
                    )

                image_size_new = rmb_data.attrs[pc_key + "_image_size"]
                if image_size is None:
                    image_size = image_size_new
                elif not np.array_equal(image_size, image_size_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] image_size is inconsistent in dataset: {image_size} != {image_size_new}"
                    )

                min_bound_new = rmb_data.attrs[pc_key + "_min_bound"]
                if min_bound is None:
                    min_bound = min_bound_new
                elif not np.allclose(min_bound, min_bound_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] min_bound is inconsistent in dataset: {min_bound} != {min_bound_new}"
                    )

                max_bound_new = rmb_data.attrs[pc_key + "_max_bound"]
                if max_bound is None:
                    max_bound = max_bound_new
                elif not np.allclose(max_bound, max_bound_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] max_bound is inconsistent in dataset: {max_bound} != {max_bound_new}"
                    )
                
                roll_pitch_yaw_new = rmb_data.attrs[pc_key + "_roll_pitch_yaw"]
                if roll_pitch_yaw is None:
                    roll_pitch_yaw = roll_pitch_yaw_new
                elif not np.allclose(roll_pitch_yaw, roll_pitch_yaw_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] roll_pitch_yaw is inconsistent in dataset: {roll_pitch_yaw} != {roll_pitch_yaw_new}"
                    )

        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps
        self.model_meta_info["data"]["use_pc_color"] = self.args.use_pc_color
        self.model_meta_info["data"]["num_points"] = num_points
        self.model_meta_info["data"]["n_point_dim"] = 6 if self.args.use_pc_color else 3
        self.model_meta_info["data"]["image_size"] = image_size
        self.model_meta_info["data"]["min_bound"] = min_bound
        self.model_meta_info["data"]["max_bound"] = max_bound
        self.model_meta_info["data"]["roll_pitch_yaw"] = roll_pitch_yaw

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema

    def set_data_stats(self):
        super().set_data_stats()

        # Load dataset
        pc_key = DataKey.get_pointcloud_key(self.args.camera_names[0])
        all_pointcloud = []
        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                # Load pointcloud
                pointcloud = rmb_data[pc_key][:: self.args.skip]
                all_pointcloud.append(pointcloud.reshape(-1, pointcloud.shape[-1]))
        all_pointcloud = np.concatenate(all_pointcloud, dtype=np.float64)

        self.model_meta_info["pointcloud"] = self.calc_stats_from_seq(all_pointcloud)

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {
                "out_min": -1.0,
                "out_max": 1.0,
            }
        else:
            return super().get_extra_norm_config()

    def setup_policy(self):
        # Set policy args
        shape_meta = OmegaConf.create(
            {
                "obs": {},
                "action": {"shape": [len(self.model_meta_info["action"]["example"])]},
            }
        )
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {
                "shape": [len(self.model_meta_info["state"]["example"])],
                "type": "low_dim",
            }
        pointcloud_shape = (
            self.model_meta_info["data"]["num_points"],
            self.model_meta_info["data"]["n_point_dim"],
        )
        shape_meta["obs"]["point_cloud"] = {
            "shape": pointcloud_shape,
            "type": "point_cloud",
        }
        pointcloud_encoder_conf = OmegaConf.create(
            {
                "in_channels": self.model_meta_info["data"]["n_point_dim"],
                "out_channels": self.args.encoder_output_dim,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
            }
        )
        self.model_meta_info["policy"]["args"] = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "num_inference_steps": 10,
            "obs_as_global_cond": True,
            "diffusion_step_embed_dim": 128,
            "down_dims": [512, 1024, 2048],
            "kernel_size": 5,
            "n_groups": 8,
            "pointcloud_encoder_cfg": pointcloud_encoder_conf,
            "use_pc_color": self.args.use_pc_color,
            "encoder_output_dim": self.args.encoder_output_dim,
        }
        self.model_meta_info["policy"]["noise_scheduler_args"] = {
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "clip_sample": True,
            "num_train_timesteps": 100,
            "set_alpha_to_one": True,
            "prediction_type": "sample",
            "steps_offset": 0,
        }

        # Construct policy
        noise_scheduler = DDIMScheduler(
            **self.model_meta_info["policy"]["noise_scheduler_args"]
        )
        self.policy = DP3(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

        # Construct exponential moving average (EMA)
        if self.args.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = EMAModel(
                model=self.ema_policy,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999,
            )

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Transfer to device
        self.policy.cuda()
        if self.args.use_ema:
            self.ema_policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        # Print policy information
        self.print_policy_info()
        print(f"  - use ema: {self.args.use_ema}")
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        data_info = self.model_meta_info["data"]
        print(
            f"  - with color: {self.args.use_pc_color}, num points: {data_info['num_points']}, image size: {data_info['image_size']}, min bound: {data_info['min_bound']}, max bound: {data_info['max_bound']}, roll_pitch_yaw: {data_info['roll_pitch_yaw']}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss, _ = self.policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                if self.args.use_ema:
                    self.ema.step(self.policy)
                batch_result_list.append(
                    self.detach_batch_result(
                        {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            if self.args.use_ema:
                policy = self.ema_policy
            else:
                policy = self.policy
            policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    loss, _ = policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary, policy=policy)
            policy.train()

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=policy)

        # Save last checkpoint
        self.save_current_ckpt("last", policy=policy)

        # Save best checkpoint
        self.save_best_ckpt()
