#!/bin/bash

source ../../.difp/bin/activate

python ./bin/Train.py DiffusionPolicy --dataset_dir ./dataset/MujocoUR5eCable_20250401_093502 --checkpoint_dir ./checkpoint/DiffusionPolicy/MujocoUR5eCable_20250401_093502

python ./bin/Train.py DiffusionPolicy --dataset_dir ./dataset/MujocoUR5eRing_20250402_132139 --checkpoint_dir ./checkpoint/DiffusionPolicy/MujocoUR5eRing_20250402_132139

python ./bin/Train.py DiffusionPolicy --dataset_dir ./dataset/MujocoUR5eParticle_20250402_133656 --checkpoint_dir ./checkpoint/DiffusionPolicy/MujocoUR5eParticle_20250402_133656

python ./bin/Train.py DiffusionPolicy --dataset_dir ./dataset/MujocoUR5eCloth_20250402_135633 --checkpoint_dir ./checkpoint/DiffusionPolicy/MujocoUR5eCloth_20250402_135633
