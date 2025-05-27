# OpenVLA-OFT

## Install

1. **Env to prepare dataset**
Install [rlds_dataset_builder](https://github.com/yu-kitagawa/rlds_dataset_builder) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/yu-kitagawa/rlds_dataset_builder.git
$ cd rlds_dataset_builder
$ pip install -r requirements.txt.
```

2. **Env to train**
Install [OpenVLA-OFT](https://github.com/yu-kitagawa/openvla) by the following commands.
```console
$ # Go to any directory
$ pip install torch torchvision torchaudio
$ git clone -b OFT https://github.com/yu-kitagawa/openvla.git
$ cd openvla
$ pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
$ pip install packaging ninja
$ ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
$ pip install "flash-attn==2.5.5" --no-build-isolation
```

3. **Env to rollout**
Install [OpenVLA-OFT](https://github.com/yu-kitagawa/openvla) as in 2.

Install [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel) according to [here](../../README.md#Install).

## Dataset preparation

Make RMB files in each of train (for training) and val directories (for validation).

Convert your RMB data to RLDS.

```console
$ Use Env 1
$ Go to rlds_dataset_builder directory
$ cd ur5e_sample
# Replace the data path on line 173-174 of the ur5e_sample_dataset_builder.py with your own data path
$ tfds build
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ Use Env 2
$ Go to openvla directory
$ torchrun --standalone --nnodes 1 --nproc-per-node <num gpus> vla-scripts/finetune.py \
--vla_path openvla/openvla-7b --data_root_dir \
<your RLDS path> --dataset_name ur5e_sample \
--run_root_dir ./log --use_l1_regression True --use_diffusion False \
--use_film True --num_images_in_input 2 --use_proprio True --batch_size 2 \
--learning_rate 5e-4 --num_steps_before_decay 5000 --max_steps 10005 \
--use_val_set True --val_freq 1000 --save_freq 1000 \
--save_latest_checkpoint_only False --image_aug True --lora_rank 32 \
--run_id_note <any id>
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ Use Env 3
$ Go to robo_manip_baselines directry
$ python ./bin/Rollout.py OpenvlaOft <task name> --checkpoint <checkpoint dir> --world_idx 0
```
The followings are supported as task name: `MujocoUR5eCable`, `MujocoUR5eRing`, `MujocoUR5eParticle`, `MujocoUR5eCloth`, `MujocoUR5eDoor`.
