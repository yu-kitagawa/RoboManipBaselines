import cv2
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    crop_and_resize,
    denormalize_data,
)

from .SarnnPolicy import SarnnPolicy


class RolloutSarnn(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(
            f"  - image crop size list: {self.model_meta_info['data']['image_crop_size_list']}"
        )
        print(f"  - image size list: {self.model_meta_info['data']['image_size_list']}")
        print(
            f"  - num attentions: {self.model_meta_info['policy']['args']['num_attentions']}"
        )

        # Construct policy
        self.policy = SarnnPolicy(
            self.state_dim,
            len(self.camera_names),
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt("cpu")

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

        self.action_plot_scale = np.concatenate(
            [DataKey.get_plot_scale(key, self.env) for key in self.state_keys]
        )

    def reset_variables(self):
        super().reset_variables()

        self.lstm_state = None

        self.policy_action_list = np.empty((0, self.state_dim))
        self.state_list = np.empty((0, self.state_dim))
        self.image_list = None
        self.predicted_image_list = None
        self.attention_list = None
        self.predicted_attention_list = None

    def infer_policy(self):
        state = self.get_state()
        image_list = self.get_images()
        (
            predicted_state,
            predicted_image_list,
            attention_list,
            predicted_attention_list,
            self.lstm_state,
        ) = self.policy(state, image_list, self.lstm_state)
        predicted_state = predicted_state[0].detach().numpy().astype(np.float64)
        self.policy_action = denormalize_data(
            predicted_state, self.model_meta_info["state"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

        # Store for plot
        state = state[0].detach().numpy().astype(np.float64)
        state = denormalize_data(state, self.model_meta_info["state"])
        self.state_list = np.concatenate([self.state_list, state[np.newaxis]])
        self.image_list = [
            image[0].detach().numpy().transpose(1, 2, 0) for image in image_list
        ]
        self.predicted_image_list = [
            predicted_image[0].detach().numpy().transpose(1, 2, 0).clip(0.0, 1.0)
            for predicted_image in predicted_image_list
        ]
        self.attention_list = [
            attention[0].detach().numpy() for attention in attention_list
        ]
        self.predicted_attention_list = [
            predicted_attention[0].detach().numpy()
            for predicted_attention in predicted_attention_list
        ]

    def get_images(self):
        image_crop_size_list = self.model_meta_info["data"]["image_crop_size_list"]
        image_size_list = self.model_meta_info["data"]["image_size_list"]

        # Assume images of different sizes are mixed
        images = []
        for camera_name, image_crop_size, image_size in zip(
            self.camera_names, image_crop_size_list, image_size_list
        ):
            image = self.info["rgb_images"][camera_name]

            image = crop_and_resize(image[np.newaxis], image_crop_size, image_size)[0]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)[torch.newaxis].to(self.device)

            images.append(image)

        return images

    def set_command_data(self):
        action_keys = [DataKey.get_command_key(key) for key in self.state_keys]
        super().set_command_data(action_keys)

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images()

        # Plot action
        self.plot_action()

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    def plot_images(self):
        image_size_list = self.model_meta_info["data"]["image_size_list"]
        for camera_idx, (
            camera_name,
            image,
            predicted_image,
            attention,
            predicted_attention,
            image_size,
        ) in enumerate(
            zip(
                self.camera_names,
                self.image_list,
                self.predicted_image_list,
                self.attention_list,
                self.predicted_attention_list,
                image_size_list,
            )
        ):
            self.ax[0, camera_idx + 1].imshow(image)
            self.ax[0, camera_idx + 1].set_title(f"obs {camera_name}", fontsize=20)
            self.ax[1, camera_idx + 1].imshow(predicted_image)
            self.ax[1, camera_idx + 1].set_title(f"pred {camera_name}", fontsize=20)

            for attention_idx in range(
                self.model_meta_info["policy"]["args"]["num_attentions"]
            ):
                self.ax[0, camera_idx + 1].plot(
                    *(attention[attention_idx] * image_size), "co", markersize=12
                )
                self.ax[0, camera_idx + 1].plot(
                    *(predicted_attention[attention_idx] * image_size),
                    "rx",
                    markersize=12,
                )

    def plot_action(self):
        history_size = 100
        for ax_idx, data_list in zip(
            [0, 1], [self.state_list, self.policy_action_list]
        ):
            ax = self.ax[ax_idx, 0]
            ax.plot(data_list[-1 * history_size :] * self.action_plot_scale)
            if ax_idx == 0:
                title = "obs state"
            else:
                title = "pred state"
            ax.set_title(title, fontsize=20)
            ax.set_xlabel("step", fontsize=16)
            ax.set_xlim(0, history_size - 1)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelsize=16)
            ax.axis("on")
