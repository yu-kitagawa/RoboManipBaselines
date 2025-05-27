# Isaac GR00T

## Install

1. **Env to prepare dataset**
Install [lerobot](https://github.com/huggingface/lerobot) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/huggingface/lerobot.git
$ cd lerobot
$ pip install -e .
```

2. **Env to train**
Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T
$ pip install -e .
$ pip install --no-build-isolation flash-attn==2.7.1.post4
```

3. **Env to rollout**
Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) as in 2.

Install [MultimodalRobotModel](https://github.com/isri-aist/MultimodalRobotModel) according to [here](../../README.md#Install).

## Dataset preparation

Convert your RMB data to lerobot dataset.

```console
$ Use Env 1
$ Go to robo_manip_baselines directry
$ python ./misc/convert_ur5e_data_to_lerobot.py --raw_dir <your RMB data dir> --repo_id <your name>/<data name> --no_push_to_hub --mode video
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ Use Env 2
$ Go to Isaac GR00T directory
$ python scripts/gr00t_finetune.py --dataset-path <your data dir> --output_dir ./log/<any name> --data_config ur5e --num-gpus <num gpus> --lora_rank 64 --lora_alpha 128 --batch-size 32 --max_steps 10000 --learning_rate 1e-5 --report_to tensorboard
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ Use Env 3
$ Go to robo_manip_baselines directry
$ python ./bin/Rollout.py Gr00t <task name> --checkpoint <checkpoint dir> --world_idx 0
```
The followings are supported as task name: `MujocoUR5eCable`, `MujocoUR5eRing`, `MujocoUR5eParticle`, `MujocoUR5eCloth`, `MujocoUR5eDoor`.
