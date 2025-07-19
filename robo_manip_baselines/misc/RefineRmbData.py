import argparse

from robo_manip_baselines.common import RmbData, find_rmb_files


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
        "--task_desc", type=str, required=True, help="task description to set"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


class RefineRmbData:
    def __init__(self, path, task_desc, overwrite=False):
        self.path = path
        self.task_desc_new = task_desc
        self.overwrite = overwrite

    def run(self):
        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in rmb_path_list:
            print(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                task_desc_orig = rmb_data.attrs.get("task_desc", "")
                if isinstance(task_desc_orig, bytes):
                    task_desc_orig = task_desc_orig.decode("utf-8")

                if task_desc_orig and not self.overwrite:
                    raise ValueError(
                        f"[{self.__class__.__name__}] task_desc already exists and is non-empty: {task_desc_orig} (use --overwrite to replace)"
                    )

                print(
                    f"Set task_desc from '{task_desc_orig}' to '{self.task_desc_new}'"
                )
                rmb_data.attrs["task_desc"] = self.task_desc_new


if __name__ == "__main__":
    refine = RefineRmbData(**vars(parse_argument()))
    refine.run()
