import glob
import os
import random


def find_rmb_files(base_path, num_files=None):
    if base_path.rstrip("/").endswith((".rmb", ".hdf5")):
        rmb_path_list = [base_path]
    elif os.path.isdir(base_path):
        rmb_path_list = sorted(
            [
                f
                for f in glob.glob(f"{base_path}/**/*.*", recursive=True)
                if f.endswith(".rmb")
                or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
            ]
        )
    else:
        raise ValueError(f"[find_rmb_files] RMB file not found: {base_path}")

    if num_files is not None:
        if num_files > len(rmb_path_list):
            raise ValueError(
                f"[find_rmb_files] Requested num_files={num_files} exceeds total available files={len(rmb_path_list)}."
            )
        rmb_path_list = sorted(random.sample(rmb_path_list, num_files))

    return rmb_path_list
