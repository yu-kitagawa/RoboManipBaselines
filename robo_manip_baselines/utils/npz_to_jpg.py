import sys
import os
import glob
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import random
import cv2
from robo_manip_baselines.common import DataKey, DataManager

data_manager = DataManager(env=None)
data_manager.load_data("table_view_env0_000.npz")

hand_images = data_manager.get_data(DataKey.get_rgb_image_key("hand"))[::6]
print(hand_images.shape)
hand_images = hand_images[:, :410, :, :]
for i in range(len(hand_images)):
    cv2.imwrite(f"{i+1:06}.jpg",cv2.cvtColor(hand_images[i], cv2.COLOR_RGB2BGR))
