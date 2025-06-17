import argparse
import glob
import os

import numpy as np

from robo_manip_baselines.common import DataKey, RmbData


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--only_stats",
        action="store_true",
        help="whether to print only statistics for the entire data set",
    )

    return parser.parse_args()


class PrintRmbData:
    def __init__(self, path, only_stats):
        self.path = path
        self.only_stats = only_stats

    def run(self):
        if os.path.isdir(self.path):
            file_list = sorted(
                [
                    f
                    for f in glob.glob(f"{self.path}/**/*.*", recursive=True)
                    if f.endswith(".rmb")
                    or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
                ]
            )
            if not file_list:
                print(
                    f"[{self.__class__.__name__}] No target files found in directory: {self.path}"
                )
                return
        else:
            file_list = [self.path]

        stats = {entry: [] for entry in ["episode_len", "success_once", "success_last"]}
        for file_path in file_list:
            print(f"[{self.__class__.__name__}] Open {file_path}")

            try:
                with RmbData(file_path) as rmb_data:
                    stats["episode_len"].append(len(rmb_data[DataKey.TIME]))
                    if DataKey.REWARD in rmb_data.keys():
                        stats["success_once"].append(
                            np.any(rmb_data[DataKey.REWARD][:] > 0.0)
                        )
                        stats["success_last"].append(rmb_data[DataKey.REWARD][-1] > 0.0)
                    else:
                        stats["success_once"].append(False)
                        stats["success_last"].append(False)

                    if self.only_stats:
                        continue

                    print("  Attrs:")
                    for k, v in rmb_data.attrs.items():
                        print(f"    - {k}: {v}")

                    print("  Data:")
                    for k in rmb_data.keys():
                        v = rmb_data[k]
                        print(f"    - {k}: {v.shape}")
            except (OSError, IOError, ValueError) as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        stats = {k: np.array(v) for k, v in stats.items()}
        print(f"[{self.__class__.__name__}] Statistics of the entire data set:")
        print(
            f"  - episode len mean: {int(stats['episode_len'].mean())}, std: {int(stats['episode_len'].std())}, "
            f"min: {stats['episode_len'].min()}, max: {stats['episode_len'].max()}"
        )
        print(
            f"  - success once: {np.sum(stats['success_once'])} / {len(file_list)},  "
            f"last: {np.sum(stats['success_last'])} / {len(file_list)}"
        )


if __name__ == "__main__":
    print_data = PrintRmbData(**vars(parse_argument()))
    print_data.run()
