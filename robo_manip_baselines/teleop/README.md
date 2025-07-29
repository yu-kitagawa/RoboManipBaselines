# Data collection by teleoperation

A full list of teleoperation environments can be found in [the environment catalog](../../doc/environment_catalog.md).

## Record demonstration data by teleoperation
You need [SpaceMouse Wireless](https://3dconnexion.com/us/product/spacemouse-wireless) for teleoperation.
Connect SpaceMouse to your PC before launching the script.

Start up the teleoperation environment by the following command:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Teleop.py MujocoUR5eCable
```
https://github.com/user-attachments/assets/59736023-a7f7-4aca-a860-176db84579f7

If you want to use a keyboard as a teleoperation input device instead of SpaceMouse:
```console
$ python ./bin/Teleop.py MujocoUR5eCable --input_device keyboard
```

If you want to use [GELLO](https://wuphilipp.github.io/gello_site) as a teleoperation input device instead of SpaceMouse:
```console
$ python ./bin/Teleop.py MujocoUR5eCable --input_device gello
```

If you want to save the data in [RmbData-SingleHDF5 (`.hdf5`) format instead of RmbData-Compact (`.rmb`)](../../doc/rmb-data-format.md):
```console
$ python ./bin/Teleop.py MujocoUR5eCable --file_format hdf5
```

If you want to collect data only in a simulation environment with limited world indices (for example, only 0 and 5), add the following option:
```console
$ python ./bin/Teleop.py MujocoUR5eCable --world_idx_list 0 5
```

To plot the point cloud, add the following option:
```console
$ python ./bin/Teleop.py MujocoUR5eCable --plot_pointcloud
```

> [!NOTE]
> If you cannot zoom the point cloud view by right-clicking, try changing the matplotlib version: `pip install matplotlib=="3.6.1"`.

To replay the teleoperation motion of the log, add the following option:
```console
$ python ./bin/Teleop.py MujocoUR5eCable --replay_log ./dataset/MujocoUR5eCable_<date_suffix>/MujocoUR5eCable_env0_000.rmb
```
