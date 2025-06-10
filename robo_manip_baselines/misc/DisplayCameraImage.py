import argparse
import os
import re

import cv2
import numpy as np

from robo_manip_baselines.common import crop_and_resize


class DisplayCameraImage:
    def __init__(self, camera_name=None, camera_id=None):
        if (camera_name is not None) and (camera_id is None):
            self.camera_name = camera_name
            self.cap = self.get_capture_from_name(camera_name)
        elif (camera_name is None) and (camera_id is not None):
            self.cap = cv2.VideoCapture(camera_id)
            self.camera_name = f"video{camera_id}"
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Only one of camera_name and camera_id should be specified. camera_name: {camera_name}, camera_id: {camera_id}"
            )

    def get_capture_from_name(self, camera_name):
        print(f"[{self.__class__.__name__}] {camera_name=}")
        cap = None
        for device_name in os.listdir("/sys/class/video4linux"):
            print(
                f"[{self.__class__.__name__}] {self.get_capture_from_name.__name__}, {device_name=}"
            )
            real_device_name = os.path.realpath(
                "/sys/class/video4linux/" + device_name + "/name"
            )
            with open(real_device_name, "r", encoding="utf-8") as device_name_file:
                detected_device_id = device_name_file.read().rstrip()
            if camera_name in detected_device_id:
                device_num = int(re.search(r"\d+$", device_name).group(0))
                print(
                    f"[{self.__class__.__name__}] {self.get_capture_from_name.__name__}, "
                    f"Found device. ID: {detected_device_id}"
                )
                cap = cv2.VideoCapture(device_num)
                if cap is None or not cap.isOpened():
                    print(
                        f"[{self.__class__.__name__}] {self.get_capture_from_name.__name__}, "
                        f"Unable to open video source ({device_num=})."
                    )
                    continue
                break
        if cap is None:
            raise LookupError(
                f"[{self.__class__.__name__}] Error: No device matching '{camera_name}' was found."
            )
        return cap

    def display_camera_output(self, crop_size=None, resize_width=None, win_xy=None):
        window_name = f"{self.camera_name} Live"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if win_xy is not None:
            cv2.moveWindow(window_name, *win_xy)

        print(f"[{self.__class__.__name__}] Press q on image to exit.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.__class__.__name__}] Failed to read the capture.")
                break

            if resize_width is None:
                resize_size = None
            else:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                resize_height = int(resize_width / aspect_ratio)
                resize_size = (resize_width, resize_height)
                cv2.resizeWindow(window_name, *resize_size)

            frame = crop_and_resize(frame[np.newaxis], crop_size, resize_size)[0]

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_name", type=str, default=None)
    parser.add_argument("--camera_id", type=int, default=None)
    parser.add_argument("--crop_size", type=int, nargs=2, default=None)
    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--win_xy", type=int, nargs=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    disp = DisplayCameraImage(camera_name=args.camera_name, camera_id=args.camera_id)
    disp.display_camera_output(
        crop_size=args.crop_size, resize_width=args.resize_width, win_xy=args.win_xy
    )
