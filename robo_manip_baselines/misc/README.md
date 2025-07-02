# Misc

## Data utilities
The file format pointed to by `<rmb_file>` can be either RmbData-Compact (`.rmb`) or RmbData-SingleHDF5 (`.hdf5`).

### Visualize demonstration data
Visualize the demonstration data by plotting it.

```console
$ python ./VisualizeData.py <rmb_file>
```

### Convert demonstration data
Convert RMB format file between RmbData-Compact (`.rmb`) and RmbData-SingleHDF5 (`.hdf5`). The format is automatically determined from the file extension.

```console
$ python ./ConvertRmbData.py <rmb_file_in> <rmb_file_out>
```

### Compare demonstration data
Compare the contents of the two RMB format files to see if they match.
Note that when converting images to mp4 files, lossy compression is applied to color images and quantization is applied to depth images, so RmbData-Compact contains some errors.

```console
$ python ./CompareRmbData.py <rmb_file1> <rmb_file2>
```

### Refine demonstration data
Update the task description attribute in RMB format files. It accepts a path to a file or directory and automatically searches for relevant files. If the task description attribute exists and `--overwrite` is not specified, the value is not changed.

```console
$ python ./RefineRmbData.py <path_to_data> --task_desc "<new_description>" [--overwrite]
```

## Visualization utilities
### Visualize camera images
Display the web camera image for recording the experiments.
```console
$ python ./DisplayCameraImage.py --camera_name Webcam --resize_width 800 --win_xy 1000 400
```

Display the cropped camera image. This is useful for image cropping policies such as SARNN.
```console
$ python ./DisplayCameraImage.py --camera_name RealSense --crop_size 280 280
```

A camera can also be specified by `--camera_id` instead of `--camera_name`.

## Video utilities
### Tile rollout videos
The input is a video consisting of a sequence of multiple rollouts, and the output is a tiled video of each rollout.
```console
$ python ./TileRolloutVideos.py <input_video_path> --output_file_name <output_video_path> --task_success_list 1 0 1 0 1 1 --column_num 3
```
The options `--output_file_name`, `--task_success_list`, and `--column_num` can be omitted.

By default, the video separation times are automatically determined by detecting restarts of windows in the video, but can be specified explicitly by adding the `--task_period_list` option as follows:
```console
--task_period_list 00:00.00-00:11.00 00:14.00-00:27.50 00:30.20-00:42.50 00:45.70-00:58.50 01:01.70-01:13.50 01:16.70-01:28.00
```

## Execution utilities
A tool that manages jobs which automatically perform training, rollout, and evaluation.

### Run immediate job
Run one evaluation job immediately and exit (no schedule).

```console
$ python ./AutoEval.py <policy> <env>
```

### Schedule daily run
Schedule an evaluation job to execute every day at the specified time.

```console
$ python ./AutoEval.py <policy> <env> --daily_schedule_time HH:MM
```

### Show queued jobs
Display all enqueued evaluation job IDs.

```console
$ python ./AutoEval.py --job_stat
```

### Delete queued job
Remove a job from the queue by its ID (filename without extension).

```console
$ python ./AutoEval.py --job_del <job_id>
```

### Additional arguments via file
When you need to pass extra arguments to `Train.py` or `Rollout.py`, write them into a text file and supply it with `--args_file_train` or `--args_file_rollout`.
For example, create a file named `train_args.txt` with the following content:
```text
--num_epochs
50000
```

Then invoke:
```console
$ python ./AutoEval.py <policy> <env> --args_file_train train_args.txt
```

### Command-line syntax reference
Complete invocation syntax and all options for `AutoEval.py`. To view it at runtime, run `python AutoEval.py -h`.

```console
$ python ./AutoEval.py [-h] [--job_stat] [--job_del JOB_DEL] [-c COMMIT_ID] [-u REPOSITORY_OWNER_NAME] [--target_dir TARGET_DIR] \
                      [-d INPUT_DATASET_LOCATION] [-k INPUT_CHECKPOINT_FILE] [--args_file_train ARGS_FILE_TRAIN] \
                      [--args_file_rollout ARGS_FILE_ROLLOUT] [--no_train] [--no_rollout] [--check_apt_packages] \
                      [--upgrade_pip_setuptools] [--world_idx_list [WORLD_IDX_LIST ...]] [--result_filename RESULT_FILENAME] \
                      [--seed SEED] [-t HH:MM] \
                      {Mlp,Sarnn,Act,DiffusionPolicy,MtAct} [{Mlp,Sarnn,Act,DiffusionPolicy,MtAct} ...] env
```

> \[!Note]  
> • Automatically clones the repo, trains the specified policy on the given environment, performs rollout, and writes `task_success_list.txt` under `misc/result/<POLICY>/<ENV>/`.  
> • Use `--input_dataset_location` with a URL (download) or local path.  
> • Always uses `policy_last.ckpt` for checkpoint.  
> • Omit `--seed` to use the called script's built-in default; specify `--seed -1` to generate a time-based random seed.
> • Omit `--commit_id` to use the latest origin/master.  
> • Pass `--world_idx_list` for multiple rollout worlds.  
