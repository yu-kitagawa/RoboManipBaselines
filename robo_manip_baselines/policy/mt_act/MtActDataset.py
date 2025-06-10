import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    get_skipped_single_data,
)


class MtActDataset(DatasetBase):
    """Dataset to train MT-ACT policy."""

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, episode_idx):
        skip = self.model_meta_info["data"]["skip"]
        chunk_size = self.model_meta_info["data"]["chunk_size"]
        task_desc_list = self.model_meta_info["data"]["task_desc_list"]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            start_time_idx = np.random.choice(episode_len)

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            rmb_data[key], start_time_idx * skip, key, skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(
                        rmb_data[key][start_time_idx * skip :],
                        key,
                        skip,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            image_keys = [
                DataKey.get_rgb_image_key(camera_name)
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ]
            images = np.stack(
                [
                    # This allows for a common hash of cache
                    rmb_data[key][::skip][start_time_idx]
                    if self.enable_rmb_cache
                    # This allows for minimal loading when reading from HDF5
                    else rmb_data[key][start_time_idx * skip]
                    for key in image_keys
                ],
                axis=0,
            )

            # Load task_desc
            task_desc = rmb_data.attrs["task_desc"]
            task_idx = task_desc_list.index(task_desc)

        # Chunk action
        action_len = action.shape[0]
        action_chunked = np.zeros((chunk_size, action.shape[1]), dtype=np.float64)
        action_chunked[:action_len] = action[:chunk_size]
        is_pad = np.zeros(chunk_size, dtype=bool)
        is_pad[action_len:] = True

        # Pre-convert data
        state, action_chunked, images = self.pre_convert_data(
            state, action_chunked, images
        )

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action_chunked, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)
        task_idx_tensor = torch.tensor(task_idx, dtype=torch.long)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return (
            state_tensor,
            images_tensor,
            action_tensor,
            is_pad_tensor,
            task_idx_tensor,
        )
