import glob
import os
import sys

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/roboagent")
)
import os

from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.common import RmbData, TrainBase

from .MtActDataset import MtActDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainMtAct(TrainBase):
    DatasetClass = MtActDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(image_aug_std=0.1)

        parser.set_defaults(batch_size=64)
        parser.set_defaults(num_epochs=2000)
        parser.set_defaults(lr=1e-5)

        parser.add_argument("--kl_weight", type=int, default=10, help="KL weight")
        parser.add_argument(
            "--chunk_size", type=int, default=20, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension"
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=3200, help="feedforward dimension"
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["chunk_size"] = self.args.chunk_size

        # Set list of all task descriptions
        all_filenames = [
            f
            for f in glob.glob(f"{self.args.dataset_dir}/**/*.*", recursive=True)
            if f.endswith(".rmb")
            or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
        ]
        self.task_desc_list = set()
        for filename in all_filenames:
            with RmbData(filename) as rmb_data:
                self.task_desc_list.add(rmb_data.attrs["task_desc"])
        self.task_desc_list = tuple(sorted(self.task_desc_list))
        self.model_meta_info["data"]["task_desc_list"] = self.task_desc_list

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "lr": self.args.lr,
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": self.args.camera_names,
        }

        # Construct policy
        DETRVAE.set_state_dim(len(self.model_meta_info["state"]["example"]))
        DETRVAE.set_action_dim(len(self.model_meta_info["action"]["example"]))
        self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.args.chunk_size}")

        # Construct text encoder
        self.text_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                task_text = [
                    self.task_desc_list[task_idx] for task_idx in data[-1].tolist()
                ]
                task_emb = self.text_encoder.encode(
                    task_text, convert_to_tensor=True, device="cuda"
                )
                batch_result = self.policy(*[d.cuda() for d in data[:-1]], task_emb)
                loss = batch_result["loss"]
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result(batch_result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    task_text = [
                        self.task_desc_list[task_idx] for task_idx in data[-1].tolist()
                    ]
                    task_emb = self.text_encoder.encode(
                        task_text, convert_to_tensor=True, device="cuda"
                    )
                    batch_result = self.policy(*[d.cuda() for d in data[:-1]], task_emb)
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()
