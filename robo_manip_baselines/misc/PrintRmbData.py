import argparse
import glob
import os

from robo_manip_baselines.common import RmbData


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help=("path to data (*.hdf5 or *.rmb) " "or directory containing them"),
    )

    return parser.parse_args()


class PrintRmbData:
    def __init__(self, path):
        self.path = path

    def run(self):
        if os.path.isdir(self.path):
            pattern_hdf5 = os.path.join(self.path, "**", "*.hdf5")
            pattern_rmb = os.path.join(self.path, "**", "*.rmb")
            file_list = glob.glob(pattern_hdf5, recursive=True) + glob.glob(
                pattern_rmb, recursive=True
            )
            file_list = sorted(file_list)
            if not file_list:
                print(
                    f"[{self.__class__.__name__}] No target files found in directory: {self.path}"
                )
                return
        else:
            file_list = [self.path]

        for file_path in file_list:
            print(f"[{self.__class__.__name__}] Open {file_path}")

            try:
                with RmbData(file_path) as rmb_data:
                    print(f"[{self.__class__.__name__}] Attrs:")
                    for k, v in rmb_data.attrs.items():
                        print(f"  - {k}: {v}")

                    print(f"[{self.__class__.__name__}] Data:")
                    for k in rmb_data.keys():
                        v = rmb_data[k]
                        print(f"  - {k}: {v.shape}")
            except (OSError, IOError, ValueError) as e:
                print(f"[Error] Failed to load {file_path}: {e}")


if __name__ == "__main__":
    print_data = PrintRmbData(**vars(parse_argument()))
    print_data.run()
