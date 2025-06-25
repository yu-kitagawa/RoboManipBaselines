import argparse
import datetime
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import venv
from pathlib import Path
from urllib.parse import urlparse

import schedule
import yaml

JSON_STATIC_METADATA_KEYS = [
    "invocation_id",
    "timestamp",
    "policy",
]
JOB_BASE_PARAM_KEYS = [
    "env",
    "commit_id",
    "repository_owner_name",
    "target_dir",
    "input_checkpoint_file",
    "no_train",
    "no_rollout",
]
AUTOEVAL_INIT_PARAM_KEYS = ["policy"] + JOB_BASE_PARAM_KEYS
AUTOEVAL_EXECUTE_PARAM_KEYS = [
    "input_dataset_location",
    "input_checkpoint_file",
    "args_file_train",
    "args_file_rollout",
    "check_apt_packages",
    "upgrade_pip_setuptools",
    "world_idx_list",
    "result_filename",
    "seed",
]
JOB_ALL_PARAM_KEYS = list(
    dict.fromkeys(JOB_BASE_PARAM_KEYS + AUTOEVAL_EXECUTE_PARAM_KEYS)
)
JSON_INVOCATION_REGISTER_KEYS = JSON_STATIC_METADATA_KEYS + JOB_ALL_PARAM_KEYS


class AutoEval:
    """Class that automatically performs installation, training, and rollout."""

    REPOSITORY_NAME = "RoboManipBaselines"
    THIRD_PARTY_PATHS = {
        "Sarnn": ["eipl"],
        "Act": ["act", "detr"],
        "DiffusionPolicy": ["diffusion_policy"],
        "MtAct": ["roboagent"],
    }
    APT_REQUIRED_PACKAGE_NAMES = [
        "libosmesa6-dev",
        "libgl1-mesa-glx",
        "libglfw3",
        "patchelf",
    ]

    def __init__(
        self,
        policy,
        env,
        commit_id,
        repository_owner_name=None,
        target_dir=None,
        input_checkpoint_file=None,
        no_train=False,
        no_rollout=False,
    ):
        """Initialize the instance with default or provided configurations."""
        self.policy = policy
        self.env = env
        self.commit_id = commit_id
        self.repository_owner_name = repository_owner_name
        self.no_train = no_train
        self.no_rollout = no_rollout

        if target_dir is None:
            print(f"[{self.__class__.__name__}] target_dir was {target_dir}.")
            target_dir = tempfile.gettempdir()
            print(
                f"[{self.__class__.__name__}] Using default temporary directory: {target_dir}"
            )

        # Resolve destination path for the repository
        self.repository_dir = self.resolve_repository_path(target_dir)
        self.input_checkpoint_file = input_checkpoint_file

        # Set the Python executable path for the virtual environment
        venv_dir = os.path.join(self.repository_dir, "..", "venv")
        self.venv_python = os.path.join(venv_dir, "bin", "python")
        venv.create(venv_dir, with_pip=True)
        self.dataset_dir = None
        self.lock_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "." + Path(__file__).resolve().stem + ".lock",
        )

        self.result_datetime_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "result/",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    @classmethod
    def resolve_repository_path(cls, target_dir):
        """Resolve a suitable repository path under the given directory."""

        # explicitly assert precondition that target_dir is not None
        assert target_dir is not None, f"[{cls.__name__}] target_dir must not be None"

        # normalize input path by removing trailing separator if provided
        normalized_path = target_dir.rstrip(os.sep)

        # if path already has expected structure, use it as-is
        if os.path.basename(normalized_path) == cls.REPOSITORY_NAME and re.fullmatch(
            rf"^{re.escape(cls.__name__)}_\d{{8}}_\d{{6}}_.+$",
            os.path.basename(os.path.dirname(normalized_path)),
        ):
            return normalized_path

        # otherwise, create new timestamped temporary directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"Rmb{cls.__name__}_{timestamp}_"
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=normalized_path)
        print(f"[{cls.__name__}] Temporary directory created: {temp_dir}")
        final_repository_path = os.path.join(temp_dir, cls.REPOSITORY_NAME)

        # reject paths that appear to be files (i.e., contain an extension)
        basename = os.path.basename(final_repository_path)
        _, ext = os.path.splitext(basename)
        if ext:
            raise IOError(f"[{cls.__name__}] File paths are not allowed: {target_dir}")

        # verify that parent directory exists
        parent_dir = os.path.dirname(final_repository_path)
        assert os.path.isdir(
            parent_dir
        ), f"[{cls.__name__}] Parent directory does not exist: {parent_dir}"

        return final_repository_path

    @classmethod
    def exec_command(cls, command, cwd=None):
        """Execute a shell command, optionally in the specified working directory,
        and return lines from standard output that match the given regex pattern."""
        print(
            f"[{cls.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Executing command: {' '.join(command)}",
            flush=True,
        )

        with subprocess.Popen(
            command,
            cwd=cwd,
            shell=False,  # secure default: using list avoids shell injection risks
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for stdout_line in process.stdout:
                print(stdout_line, end="", flush=True)

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, " ".join(command))

    def git_clone(self):
        """Clone the target Git repository into a temporary directory."""
        git_clone_url = (
            f"https://github.com/{self.repository_owner_name}/"
            + f"{self.REPOSITORY_NAME}.git"
        )
        self.exec_command(
            [
                "git",
                "clone",
                "--recursive",
                git_clone_url,
                self.repository_dir,
            ],
        )
        if self.commit_id is not None:
            self.exec_command(
                ["git", "switch", "--detach", self.commit_id],
                cwd=self.repository_dir,
            )

    @classmethod
    def check_apt_packages_installed(cls, package_names):
        """Check if required APT packages are installed."""
        for pkg in package_names:
            dpkg_query_result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Status}", pkg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if (
                dpkg_query_result.returncode != 0
                or b"install ok installed" not in dpkg_query_result.stdout
            ):
                raise AssertionError("APT package not installed: " + pkg)
        print(f"[{cls.__name__}] All required APT packages are installed.")

    def install_common(self):
        """Install common dependencies required for the environment."""
        self.exec_command(
            [self.venv_python, "-m", "pip", "install", "-e", "."],
            cwd=self.repository_dir,
        )

    def install_each_policy(self):
        """Install dependencies specific to each policy."""
        self.exec_command(
            [
                self.venv_python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[" + camel_to_snake(self.policy).replace("_", "-") + "]",
            ],
            cwd=self.repository_dir,
        )

        if self.policy in self.THIRD_PARTY_PATHS:
            self.exec_command(
                [self.venv_python, "-m", "pip", "install", "-e", "."],
                cwd=os.path.join(
                    self.repository_dir,
                    "third_party/",
                    *self.THIRD_PARTY_PATHS[self.policy],
                ),
            )

    @classmethod
    def adjust_dataset_url(cls, dataset_url):
        """Adjust and normalize the dataset URL for compatibility."""
        # Remove '\&'
        for _ in range(len(dataset_url)):
            if r"\&" not in dataset_url:
                break

            # With subprocess.Popen(shell=False), use '&' without escape.
            dataset_url = dataset_url.replace(r"\&", r"&")

        # dl=0 -> dl=1
        if dataset_url.endswith("dl=0"):
            print(
                f"[{cls.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] The URL ends with 'dl=0'. Changing it to 'dl=1'."
            )
            dataset_url = dataset_url[: -len("dl=0")] + "dl=1"

        assert dataset_url.endswith(
            "dl=1"
        ), f"[{cls.__name__}] Error: The URL '{dataset_url}' does not end with 'dl=1'."
        return dataset_url

    def download_dataset(self, dataset_url):
        """Download the dataset from the specified URL."""
        zip_filename = "dataset.zip"
        self.exec_command(
            [
                "wget",
                "-O",
                os.path.join(self.dataset_dir, zip_filename),
                self.adjust_dataset_url(dataset_url),
            ],
        )
        try:
            self.exec_command(
                [
                    "unzip",
                    "-d",
                    "dataset/",
                    zip_filename,
                ],
                cwd=self.dataset_dir,
            )
        except subprocess.CalledProcessError as e:
            e_stderr = e.stderr
            if e_stderr:
                e_stderr = e_stderr.strip()
            sys.stderr.write(
                f"[{self.__class__.__name__}] Warning: Command failed with return code "
                f"{e.returncode}. {e_stderr}\n"
            )
        rmb_items_count = len(
            glob.glob(os.path.join(self.dataset_dir, "**", "*.rmb"), recursive=True)
        )
        print(
            f"[{self.__class__.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{rmb_items_count} data files in RMB format have been unzipped."
        )

    def get_dataset(self, input_dataset_location):
        if bool(urlparse(input_dataset_location).scheme):
            # If the input is a URL, download the dataset to a fixed repo-local path.

            # Set dataset dir under repo (must exist from setup).
            self.dataset_dir = os.path.join(
                self.repository_dir, "robo_manip_baselines/dataset"
            )

            # Validate dataset dir exists.
            if not os.path.isdir(self.dataset_dir):
                raise FileNotFoundError(
                    f"expected dataset directory not found: {self.dataset_dir}"
                )

            # Download and unpack dataset to self.dataset_dir.
            self.download_dataset(input_dataset_location)
        elif os.path.isdir(input_dataset_location):
            self.dataset_dir = input_dataset_location
            return
        else:
            raise ValueError(f"Invalid: {input_dataset_location=}.")

    def train(self, args_file_train, seed):
        """Execute the training process using the specified arguments file."""
        assert self.dataset_dir

        if seed == -1:
            seed = int(time.time() * 1000000) % (2**32)
            print(
                f"[{self.__class__.__name__}] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {seed=}"
            )

        command = [
            self.venv_python,
            os.path.join(self.repository_dir, "robo_manip_baselines/bin/Train.py"),
            self.policy,
            "--dataset_dir",
            self.dataset_dir,
            "--checkpoint_dir",
            os.path.join(
                self.repository_dir,
                "robo_manip_baselines/checkpoint",
                self.policy,
                self.env,
            ),
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if args_file_train:
            command.append("@" + args_file_train)

        self.exec_command(command)

    def rollout(
        self,
        args_file_rollout,
        world_idx_list,
        result_filename,
        input_checkpoint_file,
    ):
        """Execute the rollout using the provided configuration and duration."""
        if self.input_checkpoint_file:
            input_checkpoint_file = self.input_checkpoint_file
        else:
            input_checkpoint_file = os.path.join(
                self.repository_dir,
                "robo_manip_baselines/checkpoint",
                self.policy,
                self.env,
                "policy_last.ckpt",
            )
        if not os.path.isfile(input_checkpoint_file):
            raise FileNotFoundError(
                f"checkpoint file not found: {input_checkpoint_file}"
            )
        command = [
            self.venv_python,
            os.path.join(self.repository_dir, "robo_manip_baselines/bin/Rollout.py"),
            self.policy,
            self.env,
            "--checkpoint",
            input_checkpoint_file,
            "--auto_exit",
            "--no_plot",
            "--no_render",
            "--save_last_image",
            "--output_image_dir",
            os.path.join(
                self.repository_dir,
                "robo_manip_baselines/checkpoint_dir/",
                self.policy,
                self.env,
            ),
        ]
        if world_idx_list:
            command.append("--world_idx_list")
            command.extend(map(str, world_idx_list))
        if result_filename:
            actual_result_filename = (
                result_filename.replace("<REPOSITORY_DIR>", self.repository_dir)
                .replace("<POLICY>", self.policy)
                .replace("<ENV>", self.env)
                .replace(
                    "<TIMESTAMP>", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            )
            command.extend(["--result_filename", f"{actual_result_filename}"])
        if args_file_rollout:
            command.append("@" + args_file_rollout)

        self.exec_command(command)

        if not result_filename:
            print(
                f"[{self.__class__.__name__}] Notice: Result loading was skipped as no path was provided via 'result_filename'."
            )
            return []

        assert os.path.isfile(
            actual_result_filename
        ), f"Result file not found: {actual_result_filename}"

        with open(actual_result_filename, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "success" not in data:
            raise KeyError(f"'success' field is missing in {actual_result_filename}")
        return list(map(int, data["success"]))

    def save_result(self, task_success_list):
        """Save task_success_list."""

        output_dir_path = os.path.join(self.result_datetime_dir, self.policy, self.env)
        os.makedirs(output_dir_path, exist_ok=True)
        output_file_path = os.path.join(output_dir_path, "task_success_list.txt")

        if os.path.exists(output_file_path):
            base, ext = os.path.splitext(output_file_path)
            max_attempts = 100
            for counter in range(1, max_attempts + 1):
                new_file_path = f"{base}_old_{counter}{ext}"
                if not os.path.exists(new_file_path):
                    os.rename(output_file_path, new_file_path)
                    print(
                        f"[{self.__class__.__name__}] "
                        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Existing file renamed to: {new_file_path}"
                    )
                    break
            else:
                raise RuntimeError(
                    f"Exceeded {max_attempts} attempts to rename existing file. "
                    f"Too many conflicting versions exist in: {output_dir_path}"
                )

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, task_success_list)))
        print(
            f"[{self.__class__.__name__}] "
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"File has been saved: {output_file_path}"
        )

    def execute_job(
        self,
        input_dataset_location,
        input_checkpoint_file,
        args_file_train,
        args_file_rollout,
        check_apt_packages,
        upgrade_pip_setuptools,
        world_idx_list,
        result_filename,
        seed=None,
    ):
        """
        Execute a single job:
        1) Acquire lock
        2) Git clone & checkout
        3) Create venv & check/install dependencies
        4) Train (unless no_train)
        5) Rollout and save results (unless no_rollout)
        """
        with open(self.lock_file_path, "w", encoding="utf-8") as lock_file:
            print(f"[{self.__class__.__name__}] Lock file: {self.lock_file_path}")
            print(f"[{self.__class__.__name__}] Attempting to acquire lock...")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[{self.__class__.__name__}] Lock acquired. Starting processing...")

            # Clone the Git repository and switch to the specified commit
            self.git_clone()

            # Create virtual environment and check APT packages
            venv.create(os.path.join(self.venv_python, "../../../venv/"), with_pip=True)
            if check_apt_packages:
                self.check_apt_packages_installed(self.APT_REQUIRED_PACKAGE_NAMES)
            if upgrade_pip_setuptools:
                self.exec_command(
                    [
                        self.venv_python,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "pip",
                        "setuptools",
                        "wheel",
                    ]
                )

            # Install common dependencies and policy-specific dependencies
            self.install_common()
            self.install_each_policy()

            # Training phase
            if not self.no_train:
                self.get_dataset(input_dataset_location)
                self.train(args_file_train, seed)
            else:
                print(
                    f"[{self.__class__.__name__}] "
                    "Dataset download and training disabled due to settings."
                )

            # Rollout & save results
            if not self.no_rollout:
                task_success_list = self.rollout(
                    args_file_rollout,
                    world_idx_list,
                    result_filename,
                    input_checkpoint_file,
                )
                self.save_result(task_success_list)
            else:
                print(
                    f"[{self.__class__.__name__}] "
                    "Rollout execution disabled due to current settings."
                )

        local_vars = locals()
        print(
            f"[{self.__class__.__name__}] Processing completed: "
            + textwrap.shorten(
                ", ".join(
                    (
                        f"{key}={os.path.basename(os.path.normpath(local_vars[key]))!r}"
                        if isinstance(local_vars[key], str)
                        and os.path.basename(os.path.normpath(local_vars[key]))
                        != local_vars[key]
                        else f"{key}={local_vars[key]!r}"
                    )
                    for key in AUTOEVAL_EXECUTE_PARAM_KEYS
                    if key in local_vars
                ),
                width=1000,
                placeholder="…",
            )
        )
        try:
            os.remove(self.lock_file_path)
            print(
                f"[{self.__class__.__name__}] Lock file removed: {self.lock_file_path}"
            )
        except OSError as e:
            print(
                f"[{self.__class__.__name__}] Warning: failed to remove lock file ({e})"
            )


def camel_to_snake(name):
    """Converts camelCase or PascalCase to snake_case (also converts the first letter to lowercase)"""
    name = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", name
    )  # Insert '_' between a lowercase/number and an uppercase letter
    name = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name
    )  # Insert '_' between consecutive uppercase letters followed by a lowercase letter
    name = name[0].lower() + name[1:]  # Convert the first letter to lowercase
    return name.lower()


def add_job_queue_arguments(parser):
    """Add job queue-related arguments to the parser."""
    parser.add_argument(
        "--job_stat",
        "--jstat",
        dest="job_stat",
        action="store_true",
        help="show all currently enqueued job IDs",
    )
    parser.add_argument(
        "--job_del",
        "--jdel",
        dest="job_del",
        type=str,
        help="delete previously enqueued job by job ID (filename without extension)",
    )


def parse_argument():
    """Parse and return the command-line arguments."""

    # Simplified parser for job_stat and job_del with aliases
    if (
        "--job_stat" in sys.argv
        or "--job_del" in sys.argv
        or "--jstat" in sys.argv
        or "--jdel" in sys.argv
    ):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="show or delete job queue without requiring full arguments",
        )
        add_job_queue_arguments(parser)
        return parser.parse_args()

    # Full parser for regular execution (including required positional arguments)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a parser for the evaluation based on a specific commit.",
    )

    add_job_queue_arguments(parser)
    parser.add_argument(
        "policies",
        type=str,
        nargs="+",
        choices=["Mlp", "Sarnn", "Act", "DiffusionPolicy", "MtAct"],
        help="policies",
    )
    parser.add_argument("env", type=str, help="environment")
    parser.add_argument(
        "-c",
        "--commit_id",
        type=str,
        required=False,
        default=None,
        help="git commit id to evaluate; uses latest commit if omitted",
    )
    parser.add_argument(
        "-u",
        "--repository_owner_name",
        type=str,
        default="isri-aist",
        help="github repository owner name (user or organization)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=False,
        default=None,
        help="base directory used throughout program for repository clone, "
        "virtual environment, dataset, and result outputs; "
        "if not specified (i.e. None), system temporary directory will be used as base",
    )
    parser.add_argument(
        "-d",
        "--dataset_location",
        "--input_dataset_location",
        type=str,
        dest="input_dataset_location",
        help="specify URL of online storage or local directory that provides input dataset",
    )
    parser.add_argument(
        "-k",
        "--input_checkpoint_file",
        type=str,
        required=False,
        help="specify checkpoint file to use",
    )
    parser.add_argument(
        "--args_file_train",
        type=str,
        required=False,
        help="specify argument file for training parameters; used to configure training process with detailed options",
    )
    parser.add_argument(
        "--args_file_rollout",
        type=str,
        required=False,
        help="specify argument file for rollout parameters; used to configure rollout process with detailed options",
    )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="disable training step if set",
    )
    parser.add_argument(
        "--no_rollout",
        action="store_true",
        help="disable rollout step if set",
    )
    parser.add_argument(
        "--check_apt_packages",
        action="store_true",
        help="check and install required APT packages before proceeding",
    )
    parser.add_argument(
        "--upgrade_pip_setuptools",
        action="store_true",
        help="upgrade pip and setuptools before execution",
    )
    parser.add_argument(
        "--world_idx_list",
        type=int,
        nargs="*",
        help="list of world indexes",
    )
    parser.add_argument(
        "--result_filename",
        type=str,
        default="<REPOSITORY_DIR>/robo_manip_baselines/misc/result_<POLICY>_<ENV>_<TIMESTAMP>.yaml",
        help="file path (*.yaml) to save rollout results, specify 'none' or '-' to disable saving",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="random seed; use -1 to generate different value on each run",
    )
    parser.add_argument(
        "-t",
        "--daily_schedule_time",
        type=str,
        required=False,
        metavar="HH:MM",
        help="daily schedule time, for example 18:30",
    )
    parsed_args = parser.parse_args()

    # Validate HH:MM time format
    if parsed_args.daily_schedule_time:
        if not re.fullmatch(
            r"(?:[01]\d|2[0-3]):[0-5]\d", parsed_args.daily_schedule_time
        ):
            raise ValueError(
                f"Invalid time format for --daily_schedule_time: "
                f"'{parsed_args.daily_schedule_time}'. Expected HH:MM (00-23,00-59)."
            )

    # Handle result_filename being "none" or "-"
    if parsed_args.result_filename.lower() in ("none", "-"):
        print(
            f"[{AutoEval.__name__}] Notice: '--result_filename' is set to '{parsed_args.result_filename}', disabling result saving."
        )
        parsed_args.result_filename = None

    return parsed_args


def show_queue_status(queue_dir):
    """Display currently enqueued job IDs (JSON filenames without extension)."""
    queued_files = glob.glob(os.path.join(queue_dir, "*.json"))
    if not queued_files:
        print(f"[{AutoEval.__name__}] There are currently no jobs enqueued.")
    else:
        print(f"=== [{AutoEval.__name__}] Enqueued Job IDs ===")
        for jf in sorted(queued_files):
            job_id = os.path.splitext(os.path.basename(jf))[0]
            print(f"- {job_id}")


def delete_queued_job(queue_dir, job_id):
    """Delete the specified job JSON from the queue directory."""
    target_file = os.path.join(queue_dir, f"{job_id}.json")
    if os.path.isfile(target_file):
        os.remove(target_file)
        print(f"[{AutoEval.__name__}] Deleted job '{job_id}' from the queue.")
    else:
        print(
            f"[{AutoEval.__name__}] The specified job ID '{job_id}' does not exist in the queue."
        )


def main():
    args = parse_argument()

    # Determine base directory for job queue
    queue_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f".sys_queue_{Path(__file__).resolve().stem}",
    )
    os.makedirs(queue_dir, exist_ok=True)

    if args.job_stat:
        show_queue_status(queue_dir)
        return

    if args.job_del:
        delete_queued_job(queue_dir, args.job_del)
        return

    def register_invocation():
        """Register a JSON file per policy in queue_dir and return a list of invocation IDs."""
        # Exclusive control using a lock file
        lock_path = os.path.join(queue_dir, "register.lock")
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            # Generate a timestamp string common to all policies in this invocation
            created = []
            for policy in args.policies:
                # Construct unique invocation ID by combining timestamp and policy
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                invocation_id = f"{ts}_{policy}"
                # Aggregate invocation details including common parameters and the policy
                info = {
                    "invocation_id": invocation_id,
                    "timestamp": ts,
                    "policy": policy,
                    **{k: getattr(args, k) for k in JOB_ALL_PARAM_KEYS},
                }

                # Check if keys match
                missing = set(JSON_INVOCATION_REGISTER_KEYS) - set(info.keys())
                extra = set(info.keys()) - set(JSON_INVOCATION_REGISTER_KEYS)
                if missing or extra:
                    raise RuntimeError(
                        f"JSON keys mismatch: missing={missing}, extra={extra}"
                    )

                # Persist invocation details as a JSON file named by invocation ID
                fn = os.path.join(queue_dir, f"{invocation_id}.json")
                with open(fn, "w", encoding="utf-8") as jf:
                    json.dump(info, jf, indent=4)
                created.append(invocation_id)
            # Release the exclusive lock
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        # Remove lock file after releasing the lock to clean up
        os.remove(lock_path)
        # Log registration status for each invocation
        for inv in created:
            print(f"[{AutoEval.__name__}] Job registered: {inv}")
        return created

    def handle_all_jobs():
        """
        Execute all jobs in the queue sequentially and
        delete each corresponding JSON file upon completion.
        """
        has_error = False

        # Register a single invocation containing multiple policies
        _ = register_invocation()

        # Display list of pending invocation files
        inv_files = sorted(glob.glob(os.path.join(queue_dir, "*.json")))
        print(f"\n[{AutoEval.__name__}] === Pending Invocations ===")
        for f in inv_files:
            print(f"- {os.path.basename(f)}")
        print()

        # Process each invocation sequentially
        for inv_file in inv_files:
            with open(inv_file, "r", encoding="utf-8") as jf:
                inv_info = json.load(jf)
            job_id = inv_info["invocation_id"]

            print(f"\n[{AutoEval.__name__}] Execute job: {job_id}")
            auto_eval = AutoEval(*(inv_info[k] for k in AUTOEVAL_INIT_PARAM_KEYS))
            try:
                auto_eval.execute_job(
                    *(inv_info[k] for k in AUTOEVAL_EXECUTE_PARAM_KEYS)
                )
            except Exception:
                import traceback

                traceback.print_exc()
                has_error = True

            # After completing this invocation, delete its JSON file
            os.remove(inv_file)
            print(f"[{AutoEval.__name__}] Removed invocation file: {inv_file}")
        print(
            f"[{AutoEval.__name__}] Completed job: {job_id}: {'Error' if has_error else 'OK'}"
        )

    # Immediate execution mode (no schedule)
    if not args.daily_schedule_time:
        handle_all_jobs()
        return

    # Scheduling mode
    scheduled_msg = f"[{AutoEval.__name__}] Scheduled daily run at {args.daily_schedule_time}. Waiting..."

    def scheduled_job():
        handle_all_jobs()
        print(scheduled_msg, flush=True)

    schedule.every().day.at(args.daily_schedule_time).do(scheduled_job)
    print(scheduled_msg, flush=True)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{AutoEval.__name__}] Scheduler stopped by user.")


if __name__ == "__main__":
    main()
