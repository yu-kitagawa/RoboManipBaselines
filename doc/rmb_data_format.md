# Data format in RoboManipBaselines

## Overview
RoboManipBaselines stores demonstration data in an original file format called **RMB format**.

RMB format has the following two variants. RmbData-Compact is used by default.
- RmbData-Compact: Store color and depth images in mp4 files, all other data in a single HDF5 file.
  - Pros: small file size, Cons: image read/write time overhead.
- RmbData-SingleHDF5: Store all data in a single HDF5 file.
  - Pros: fast read/write, Cons: large file size.

A single `.hdf5` file for RmbData-SingleHDF5 and a single `.rmb` directory for RmbData-Compact store one demonstration episode.
An example of the file structure of RmbData-Compact is shown below.
```console
sample.rmb
├── main.rmb.hdf5
├── hand_depth_image.rmb.mp4
├── front_depth_image.rmb.mp4
├── hand_rgb_image.rmb.mp4
├── front_rgb_image.rmb.mp4
├── side_rgb_image.rmb.mp4
└── side_depth_image.rmb.mp4
```

Data other than images are stored in `main.rmb.hdf5`. Color images are stored in `<camera_name>_rgb_image.rmb.mp4`, and depth images in `<camera_name>_depth_image.rmb.mp4`.
To convert a depth image to mp4, [the videoio library](https://github.com/vguzov/videoio) is used internally. Depth is quantized in units of 1 mm.

See [here](../robo_manip_baselines/misc/README.md#Data-utilities) for utility scripts for data conversion and visualization.

## Loading data
RMB format files can be loaded as follows.
```python
from robo_manip_baselines.common import RmbData, DataKey

with RmbData(rmb_file_path) as rmb_data:
    data_shape = rmb_data[DataKey.MEASURED_JOINT_POS].shape
    single_data = rmb_data[DataKey.MEASURED_JOINT_POS][0]
    sliced_data = rmb_data[DataKey.MEASURED_JOINT_POS][1:4]
    whole_data = rmb_data[DataKey.MEASURED_JOINT_POS][:]

    camera_name = "front"
    single_rgb = rmb_data[DataKey.get_rgb_image_key(camera_name)][1]
    sliced_depth = rmb_data[DataKey.get_depth_image_key(camera_name)][::10]
```

The file format pointed to by `rmb_file_path` can be either RmbData-Compact (`.rmb`) or RmbData-SingleHDF5 (`.hdf5`).

See the items in [DataKey.py](../robo_manip_baselines/common/data/DataKey.py) for the data keys contained in the RMB format data.
