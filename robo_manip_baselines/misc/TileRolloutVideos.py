import argparse
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from enum import Enum, auto

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

RESIZE_VIDEO_LOGLEVEL_DEFAULT = "info"
RESIZE_VIDEO_LOGLEVEL_QUIET = "warning"


def time_str_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, "%M:%S.%f")
    return timedelta(
        minutes=time_obj.minute,
        seconds=time_obj.second,
        microseconds=time_obj.microsecond,
    ).total_seconds()


def seconds_to_time_str(seconds):
    formatted_time = datetime(
        year=1, month=1, day=1, hour=0, minute=0, second=0
    ) + timedelta(seconds=seconds)
    time_str = formatted_time.strftime("%M:%S.%f")
    tparts = time_str.split(".")
    return f"{tparts[0]}.{tparts[1][:2]}"


def get_video_properties(input_file):
    def get_props(video):
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_rate, frame_w, frame_h, frame_count

    if isinstance(input_file, cv2.VideoCapture):
        return get_props(input_file)
    if isinstance(input_file, str):
        video = cv2.VideoCapture(input_file)
        props = get_props(video)
        video.release()
        return props
    raise TypeError(f"The type is unexpected: {type(input_file)}")


class TaskEventHandler:
    class Stat(Enum):
        INITIAL = auto()
        STARTED = auto()
        STOPPED = auto()

    PRINT_INTERVAL_VIDEO_SEC = 15

    def __init__(
        self, task_period_list, frame_rate, shift_seconds, satur_thresh, quiet
    ):
        self.state = self.Stat.INITIAL

        self.env_idx = 0
        self.task_period_list = task_period_list
        self.frame_rate = frame_rate
        self.shift_seconds = shift_seconds
        self.satur_thresh = satur_thresh
        self.quiet = quiet

    def write_message(
        self, time_str, region_satur_mean, func_name=None, env_str=None, state_name=""
    ):
        if env_str is None:
            env_str = f"env={self.env_idx:<2}"
        tqdm.write(
            f"[{self.__class__.__name__}] "
            + "\t".join(
                [
                    f"{region_satur_mean=:>7.3f}",
                    f"{self.satur_thresh=:>7.3f}",
                    time_str,
                    (
                        func_name
                        if func_name is not None
                        else " " * len(self.start_env.__name__)
                    ),
                    env_str,
                    state_name,
                ]
            )
        )

    def start_env(self, i_frame_curr, region_satur_mean):
        self.state = self.Stat.STARTED
        time_str = seconds_to_time_str(i_frame_curr / self.frame_rate)
        self.write_message(
            time_str, region_satur_mean, func_name=self.start_env.__name__
        )
        self.task_period_list.append(f"{time_str}-")

    def stop_env(self, i_frame_curr, region_satur_mean):
        self.state = self.Stat.STOPPED
        seconds_stop = i_frame_curr / self.frame_rate
        time_str_stop = seconds_to_time_str(seconds_stop)
        time_str_start = self.task_period_list[-1].rstrip("-")
        seconds_start = time_str_to_seconds(time_str_start)
        if seconds_stop < seconds_start:
            time_str_stop = time_str_start
        self.write_message(
            time_str_stop, region_satur_mean, func_name=self.stop_env.__name__
        )
        self.task_period_list[-1] += time_str_stop
        self.env_idx += 1

    def cancel_env(self, i_frame_curr, region_satur_mean):
        self.state = self.Stat.STOPPED
        if self.task_period_list:
            self.task_period_list.pop()
        time_str = seconds_to_time_str(i_frame_curr / self.frame_rate)
        self.write_message(
            time_str, region_satur_mean, func_name=self.cancel_env.__name__
        )

    def initial_stop(self):
        self.state = self.Stat.STOPPED

    def handle(self, is_satur_screen, i_frame_curr, region_satur_mean):
        if (not self.quiet) and (
            i_frame_curr % (self.frame_rate * self.PRINT_INTERVAL_VIDEO_SEC) == 0
        ):
            time_str = seconds_to_time_str(i_frame_curr / self.frame_rate)
            self.write_message(
                time_str,
                region_satur_mean,
                env_str=(
                    f"env={self.env_idx:<2}"
                    if self.state == self.Stat.STARTED
                    else " " * len("env= 0")
                ),
                state_name=self.state.name,
            )
        if self.state == self.Stat.STARTED:
            if not is_satur_screen:
                i_frame_stop = i_frame_curr - self.shift_seconds * self.frame_rate
                time_str_start = self.task_period_list[-1].rstrip("-")
                i_frame_start = int(
                    self.frame_rate * time_str_to_seconds(time_str_start)
                )
                if i_frame_start <= i_frame_stop:
                    self.stop_env(i_frame_stop, region_satur_mean)
                else:
                    self.cancel_env(i_frame_start, region_satur_mean)
        elif self.state == self.Stat.STOPPED:
            if is_satur_screen:
                self.start_env(
                    i_frame_curr
                    + self.shift_seconds  # Skip brief frames: begin shift post-window
                    * self.frame_rate,
                    region_satur_mean,
                )
        elif self.state == self.Stat.INITIAL:
            if not is_satur_screen:
                self.initial_stop()
            else:
                self.start_env(i_frame_curr, region_satur_mean)
        else:
            raise AssertionError(f"{self.state=}")


class TileRolloutVideos:
    GREEN = (0, 128, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)

    NUM_SAMPLE_FRAMES_FOR_MARGIN_REMOVAL = 30
    CAPTURE_TRIM_COLOR_THRESHOLD = 64

    def __init__(
        self,
        task_success_list,
        column_num,
        codec,
        border_size,
        shift_seconds,
        satur_thresh,
        satur_detection_region_ratio,
        quiet,
    ):
        self.quiet = quiet
        self.task_success_list = task_success_list
        self.column_num = column_num
        self.codec = codec
        self.border_size = border_size
        self.shift_seconds = shift_seconds
        self.satur_thresh = satur_thresh
        self.satur_detection_region_ratio = satur_detection_region_ratio

        self.input_file_name = None
        self.task_period_list = None
        self.frame_rate = None

    @staticmethod
    def resize_video_ifneeded(input_file_name, max_video_width, quiet):
        """Resize a video exceeding max_video_width, saving to a temp file."""
        _, curr_video_width, _, _ = get_video_properties(input_file_name)
        if curr_video_width <= max_video_width:
            return input_file_name
        resize_file_name = os.path.join(
            tempfile.mkdtemp(), f"resized_width_{max_video_width}.mp4"
        )
        ffmpeg.input(input_file_name).output(
            resize_file_name,
            vf=f"scale={max_video_width}:-2",
            loglevel=(
                RESIZE_VIDEO_LOGLEVEL_QUIET if quiet else RESIZE_VIDEO_LOGLEVEL_DEFAULT
            ),
        ).run()
        return resize_file_name

    @classmethod
    def sample_frames(cls, input_file_name):
        cap = cv2.VideoCapture(input_file_name)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_file_name}")

        num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(
            0,
            num_total_frames - 1,
            cls.NUM_SAMPLE_FRAMES_FOR_MARGIN_REMOVAL,
            dtype=int,
        )  # sample frames at equal intervals

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        ret, first_frame = cap.read()  # read the first frame to determine frame size
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {frame_indices[0]}")
        height, width, channels = first_frame.shape
        sampled_frames = np.empty(
            (cls.NUM_SAMPLE_FRAMES_FOR_MARGIN_REMOVAL, height, width, channels),
            dtype=np.uint8,
        )  # pre-allocate array
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame at index {idx}")
            sampled_frames[i] = frame

        cap.release()
        return sampled_frames

    @classmethod
    def remove_white_margin(cls, input_file_name):
        # trim, scale
        sampled_frames = cls.sample_frames(input_file_name)
        dist_mask = np.where(
            np.mean(np.abs((sampled_frames - cls.WHITE).mean(axis=0)), axis=2)
            > cls.CAPTURE_TRIM_COLOR_THRESHOLD
        )
        y_min, y_max, x_min, x_max = (
            func(dist_mask[ax]) for ax in [0, 1] for func in [min, max]
        )

        w_trim = x_max - x_min
        h_trim = y_max - y_min

        if w_trim < 1 or h_trim < 1:
            sys.stderr.write(
                f"Warning in {cls.remove_white_margin.__name__}: "
                f"Either {w_trim=} or {h_trim=} is less than 1, "
                "so no removal of white margin will be performed. "
                f"Returning {input_file_name=} as is.\n"
            )
            return input_file_name

        video_margin_removed = ffmpeg.input(input_file_name).crop(
            x_min, y_min, w_trim, h_trim
        )

        # write
        removed_file_name = os.path.join(tempfile.mkdtemp(), "removed_white_margin.mp4")
        try:
            video_margin_removed.output(removed_file_name).run(overwrite_output=False)
        except ffmpeg.Error:
            sys.stderr.write(f"{(input_file_name, removed_file_name)=}")
            raise

        return removed_file_name

    @staticmethod
    def is_saturated(frame, satur_thresh, lefttop_x, lefttop_y, size_x, size_y):
        crop_x = round(frame.shape[1] * lefttop_x)
        crop_y = round(frame.shape[0] * lefttop_y)
        crop_w = round(frame.shape[1] * size_x)
        crop_h = round(frame.shape[0] * size_y)
        region = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        region_satur = region_hsv[..., 1]
        region_satur_mean = region_satur.mean()
        is_satur_screen = region_satur_mean >= satur_thresh
        return is_satur_screen, region_satur_mean

    def update_task_period_list_ifneeded(self, task_period_list):
        if not self.quiet:
            print(
                f"[{self.__class__.__name__}] {self.update_task_period_list_ifneeded.__name__} ..."
            )

        if len(task_period_list) >= 1:
            return task_period_list

        video = cv2.VideoCapture(self.input_file_name)
        frame_rate, _, _, frame_count = get_video_properties(video)
        task_event_handler = TaskEventHandler(
            task_period_list,
            frame_rate,
            self.shift_seconds,
            self.satur_thresh,
            self.quiet,
        )

        for i_frame_curr in tqdm(
            range(frame_count), desc=self.update_task_period_list_ifneeded.__name__
        ):
            ret, frame = video.read()
            if not ret:
                continue
            is_satur_screen, region_satur_mean = self.is_saturated(
                frame, self.satur_thresh, *self.satur_detection_region_ratio
            )

            task_event_handler.handle(is_satur_screen, i_frame_curr, region_satur_mean)
        task_event_handler.handle(
            is_satur_screen=False,
            i_frame_curr=frame_count,
            region_satur_mean=-float("inf"),
        )

        video.release()
        return task_event_handler.task_period_list

    def parse_frame_periods(self):
        return_frame_periods = []

        video = cv2.VideoCapture(self.input_file_name)

        frame_rate, _, _, _ = get_video_properties(video)
        parsed_values = [
            [
                time_str_to_seconds(time_str) * frame_rate
                for time_str in start_end_time_str.split("-")
            ]
            for start_end_time_str in self.task_period_list
        ]

        for i_frame_start, i_frame_end in parsed_values:
            video.set(cv2.CAP_PROP_POS_FRAMES, i_frame_start)
            ret, frame_start = video.read()
            assert ret
            is_satur_screen, _ = self.is_saturated(
                frame_start, self.satur_thresh, *self.satur_detection_region_ratio
            )
            shift_frame = (
                (self.shift_seconds * frame_rate) if not is_satur_screen else 0.0
            )
            return_frame_periods.append([i_frame_start + shift_frame, i_frame_end])

        video.release()
        return return_frame_periods

    def init_frames(
        self,
        max_trim_len,
        frame_h,
        frame_w,
    ):
        if not self.quiet:
            print(
                f"[{self.__class__.__name__}] {self.init_frames.__name__}, create new array ..."
            )

        row_num = math.ceil(len(self.task_period_list) / self.column_num)

        if len(self.task_success_list) == 0:
            initial_frames = np.full(
                (
                    max_trim_len,  # frame
                    (frame_h + 2 * self.border_size) * row_num,  # height
                    (frame_w + 2 * self.border_size) * self.column_num,  # width
                    3,  # channel
                ),
                self.WHITE,
                dtype=np.uint8,
            )
            return initial_frames
        initial_frames = np.full(
            (
                max_trim_len,  # frame
                (frame_h + 2 * self.border_size) * row_num,  # height
                (frame_w + 2 * self.border_size) * self.column_num,  # width
                3,  # channel
            ),
            self.GREEN,
            dtype=np.uint8,
        )

        if not self.quiet:
            print(
                f"[{self.__class__.__name__}] {self.init_frames.__name__}, set elements ..."
            )
        for i_subvideo, is_task_successful in enumerate(
            tqdm(list(int(x) for x in self.task_success_list))
        ):
            assert is_task_successful in (0, 1)

            if is_task_successful:
                continue

            # outside frame
            x_draw = (frame_w + 2 * self.border_size) * (i_subvideo % self.column_num)
            y_draw = (frame_h + 2 * self.border_size) * (i_subvideo // self.column_num)
            initial_frames[
                :,  # frame
                y_draw : y_draw + (frame_h + 2 * self.border_size),  # height
                x_draw : x_draw + (frame_w + 2 * self.border_size),  # width
                :,  # channel
            ] = self.RED

        # overwrite surplus tiles with white
        total_tiles = row_num * self.column_num
        for i_subvideo in range(len(self.task_success_list), total_tiles):
            x_draw = (frame_w + 2 * self.border_size) * (i_subvideo % self.column_num)
            y_draw = (frame_h + 2 * self.border_size) * (i_subvideo // self.column_num)
            initial_frames[
                :,
                y_draw : y_draw + (frame_h + 2 * self.border_size),
                x_draw : x_draw + (frame_w + 2 * self.border_size),
                :,
            ] = self.WHITE

        return initial_frames

    def read_video(
        self,
        input_file_name,
        task_period_list,
        keep_white_margin,
        max_video_width,
    ):
        if not self.quiet:
            print(
                f"[{self.__class__.__name__}] {self.resize_video_ifneeded.__name__} ..."
            )
        input_file_name = self.resize_video_ifneeded(
            input_file_name, max_video_width, self.quiet
        )

        if not keep_white_margin:
            if not self.quiet:
                print(
                    f"[{self.__class__.__name__}] {self.remove_white_margin.__name__} ..."
                )
            input_file_name = self.remove_white_margin(input_file_name)

        self.input_file_name = input_file_name

        self.task_period_list = self.update_task_period_list_ifneeded(task_period_list)
        if not self.quiet:
            print(
                f"[{self.__class__.__name__}] task_period_list {' '.join(self.task_period_list)}"
            )
        assert len(task_period_list) >= 1, f"{len(task_period_list)=}"
        if len(self.task_success_list) >= 1:
            if len(self.task_success_list) != len(task_period_list):
                raise ValueError(
                    (
                        "lengths of task_success_list "
                        f"({len(self.task_success_list)}) and "
                        "task_period_list "
                        f"({len(task_period_list)}) must be same."
                    )
                )

        if not self.quiet:
            print(
                f"[{self.__class__.__name__}] {self.parse_frame_periods.__name__} ..."
            )

        frame_periods = self.parse_frame_periods()
        max_trim_len = int(math.ceil(max(e - s for s, e in frame_periods)))

        if not self.quiet:
            print(f"[{self.__class__.__name__}] {get_video_properties.__name__} ...")
        self.frame_rate, frame_w, frame_h, _ = get_video_properties(input_file_name)

        if not self.quiet:
            print(f"[{self.__class__.__name__}] {self.init_frames.__name__} ...")
        initial_frames = self.init_frames(
            max_trim_len,
            frame_h,
            frame_w,
        )

        return initial_frames, frame_periods

    def tile_video(self, final_frames, frame_periods):
        if not self.quiet:
            print(f"[{self.__class__.__name__}] {self.tile_video.__name__} ...")

        video = cv2.VideoCapture(self.input_file_name)
        _, frame_w, frame_h, frame_count = get_video_properties(video)

        for i_frame_curr in tqdm(range(frame_count), desc=self.tile_video.__name__):
            ret, frame = video.read()
            if not ret:
                continue
            for i_subvideo, (i_frame_start, i_frame_end) in enumerate(frame_periods):
                i_frame_draw = int(
                    np.clip(i_frame_curr - i_frame_start, 0, final_frames.shape[0] - 1)
                )
                x_draw = self.border_size + (frame_w + 2 * self.border_size) * (
                    i_subvideo % self.column_num
                )
                y_draw = self.border_size + (frame_h + 2 * self.border_size) * (
                    i_subvideo // self.column_num
                )
                if i_frame_curr < i_frame_start:
                    continue
                if i_frame_curr <= i_frame_end:
                    final_frames[
                        i_frame_draw,  # frame
                        y_draw : y_draw + frame_h,  # height
                        x_draw : x_draw + frame_w,  # width
                        ...,  # channel
                    ] = frame
                    if (i_frame_end - i_frame_curr) <= 1.0:
                        final_frames[
                            i_frame_draw:,  # frame
                            y_draw : y_draw + frame_h,  # height
                            x_draw : x_draw + frame_w,  # width
                            ...,  # channel
                        ] = frame

        video.release()
        return final_frames

    def write_video(self, final_frames, output_file_name):
        if not self.quiet:
            print(f"[{self.__class__.__name__}] {self.write_video.__name__} ...")

        # Get the output directory and create it if it does not exist
        output_dir_name = os.path.dirname(output_file_name)
        if output_dir_name:  # false if empty, "/", or no directory (e.g., "output.mp4")
            if not os.path.exists(
                output_dir_name
            ):  # check if the directory does not exist
                os.makedirs(output_dir_name)

        # If the output_file_name already exists, rename it
        if os.path.exists(output_file_name):
            backup_file_name = output_file_name + "~"
            os.rename(output_file_name, backup_file_name)
            print(
                f"The existing file '{output_file_name}' has been renamed to '{backup_file_name}'."
            )

        # Write video frames to a temporary file
        temp_dir_name = tempfile.gettempdir()
        temp_file_name = os.path.join(temp_dir_name, "temp_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer_out = cv2.VideoWriter(
            temp_file_name,
            fourcc,
            self.frame_rate,
            (final_frames.shape[2], final_frames.shape[1]),
        )
        for i_frame_curr in tqdm(
            range(final_frames.shape[0]), desc=writer_out.write.__name__
        ):
            writer_out.write(final_frames[i_frame_curr, ...])
        writer_out.release()

        # Use FFmpeg to create high-compression video file
        subprocess.run(
            [
                "ffmpeg",
                # command-line tool for processing video, audio
                "-i",
                # input specifier
                temp_file_name,
                # input file name
                "-c:v",
                # video codec specifier
                "libx264",
                # H.264 encoder for efficient compression
                "-preset",
                # preset option controlling speed-compression tradeoff
                "slow",
                # slower processing for improved compression
                "-crf",
                # constant rate factor for quality control
                "23",
                # quality level value, lower equals better quality
                "-c:a",
                # audio codec specifier
                "copy",
                # direct audio stream copying
                output_file_name,
                # output file name
            ],
            check=False,
            # do not raise exception on command failure
        )
        os.remove(temp_file_name)


def parse_arg():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file_name", type=str)
    parser.add_argument("--task_period_list", "-p", nargs="*", default=[])
    parser.add_argument("--task_success_list", "-s", nargs="*", default=[])
    parser.add_argument(
        "--output_file_name",
        "-o",
        type=str,
        default="./output_" + os.path.splitext(os.path.basename(__file__))[0] + ".mp4",
    )
    parser.add_argument("--column_num", "-n", type=int, default=3)
    parser.add_argument(
        "--codec",
        "-c",
        type=str,
        default="mp4v",
        # OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec
        # id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
    )
    parser.add_argument("--border_size", "-b", type=int, default=15)
    parser.add_argument(
        "--keep_white_margin", action="store_true", help="keep white margins in image"
    )
    parser.add_argument(
        "--max_video_width",
        "-w",
        type=int,
        default=2560,
        help=(
            "maximum width to which the video will be scaled down if it is too large"
        ),
    )
    parser.add_argument(
        "--shift_seconds",
        "-t",
        type=float,
        default=0.5,
        help="remove seconds when starting from a dull screen",
    )
    parser.add_argument(
        "--satur_thresh",
        "-e",
        type=float,
        default=5.5,
        help="threshold used to determine if the screen is saturated",
    )
    parser.add_argument(
        "--satur_detection_region_ratio",
        "-r",
        nargs=4,
        metavar=("LEFTTOP_X", "LEFTTOP_Y", "SIZE_X", "SIZE_Y"),
        type=float,
        default=[0, 0, 1, 1],
        help="ratio for the region used to detect if the screen is saturated",
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    for r in args.satur_detection_region_ratio:
        assert 0.0 <= r <= 1.0

    assert args.input_file_name != args.output_file_name

    if not args.quiet:
        print(f"{args=}")
    return args


def main():
    args = parse_arg()

    handler = TileRolloutVideos(
        args.task_success_list,
        args.column_num,
        args.codec,
        args.border_size,
        args.shift_seconds,
        args.satur_thresh,
        args.satur_detection_region_ratio,
        args.quiet,
    )

    initial_frames, frame_periods = handler.read_video(
        args.input_file_name,
        args.task_period_list,
        args.keep_white_margin,
        args.max_video_width,
    )

    final_frames = handler.tile_video(
        initial_frames,
        frame_periods,
    )

    handler.write_video(final_frames, args.output_file_name)

    print(
        f"[{TileRolloutVideos.__class__.__name__}] Save a video: {args.output_file_name}"
    )
    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()
