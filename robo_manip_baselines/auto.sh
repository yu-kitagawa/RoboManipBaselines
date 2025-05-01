#!/bin/bash

POL=$1
CKPT=$2

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
  python ./bin/Rollout.py ${POL} MujocoUR5eRing --checkpoint ./checkpoint/${POL}/MujocoUR5eRing_20250402_132139/policy_${CKPT}.ckpt --world_idx ${WORLD_IDX} --win_xy_policy 0 642
done
