#!/bin/bash

POL=$1
CKPT=$2

WORLD_IDX_LIST=(0 1 2 3 4 5)
for WORLD_IDX in "${WORLD_IDX_LIST[@]}"; do
  echo "----------world_idx ${WORLD_IDX}----------"
  python ./bin/Rollout.py ${POL} RealUR5eDemo --config ./envs/configs/RealUR5eDemoEnv.yaml --checkpoint ./checkpoint/${POL}/RealUR5eDemo_20250403_152346_sarnn_front/policy_${CKPT}.ckpt --world_idx ${WORLD_IDX} --win_xy_policy 0 642 --wait_before_start
done
