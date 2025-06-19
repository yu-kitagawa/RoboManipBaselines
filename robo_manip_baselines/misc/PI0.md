# lerobot pi0

## Install

1. **Env to prepare dataset and train**
Install [lerobot](https://github.com/yu-kitagawa/lerobot) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/yu-kitagawa/lerobot.git
$ cd lerobot
$ pip install -e .
$ pip install transformers pytest tensorboard
```

2. **Env to rollout**
Install [lerobot](https://github.com/yu-kitagawa/lerobot) as in 2.

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../README.md#Install).

## Dataset preparation

Convert your RMB data to lerobot dataset.

```console
$ Use Env 1
$ Go to robo_manip_baselines directry
$ python ./misc/convert_ur5e_data_to_lerobot.py --raw_dir <your RMB data dir> --repo_id <your name>/<data name> --no_push_to_hub --mode video
```

## Model Training

Train the model. The trained weights are saved in the `outputs` folder.

```console
$ Use Env 1
$ Go to lerobot directory
$ python lerobot/scripts/train.py --policy.path=lerobot/pi0 --dataset.repo_id=<your data dir>
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ Use Env 2
$ Go to robo_manip_baselines directry
$ python ./bin/Rollout.py Pi0 <task name> --checkpoint <checkpoint dir> --world_idx 0
```
The followings are supported as task name: `MujocoUR5eCable`, `MujocoUR5eRing`, `MujocoUR5eParticle`, `MujocoUR5eCloth`, `MujocoUR5eDoor`.
