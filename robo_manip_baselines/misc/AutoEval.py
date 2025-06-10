import argparse
import fcntl
import glob
import os
import re
import subprocess
import sys
import tempfile
import time
import venv
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import schedule

LOCK_FILE_PATH = str(
    Path("/tmp")
    / f"{'_'.join(Path(__file__).resolve().parts[-4:-1] + (Path(__file__).resolve().stem,))}.lock"
)
ROLLOUT_RESULT_PRINT_PATTERN = re.compile(r"Rollout result: (success|failure)")


class AutoEval:
    """Class that automatically performs installation, training, and rollout."""

    REPOSITORY_NAME = "RoboManipBaselines"
    THIRD_PARTY_PATHS = {
        "Sarnn": ["eipl"],
        "Act": ["act", "detr"],
        "DiffusionPolicy": ["diffusion_policy"],
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
        checkpoint_base_dir=None,
        is_rollout_disabled=False,
    ):
        """Initialize the instance with default or provided configurations."""
        self.policy = policy
        self.env = env
        self.commit_id = commit_id
        self.repository_owner_name = repository_owner_name
        self.is_rollout_disabled = is_rollout_disabled

        self.repository_tmp_dir = os.path.join(tempfile.mkdtemp(), self.REPOSITORY_NAME)
        self.checkpoint_temp_dir = (
            tempfile.mkdtemp(dir=checkpoint_base_dir)
            if checkpoint_base_dir is not None
            else self.repository_tmp_dir
        )
        self.venv_python = os.path.join(tempfile.mkdtemp(), "venv/bin/python")
        self.dataset_dir = None

        self.result_datetime_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "result/",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    @classmethod
    def exec_command(cls, command, cwd=None, stdout_line_match_pattern=None):
        """Execute a shell command, optionally in the specified working directory,
        and return lines from standard output that match the given regex pattern."""
        print(
            f"[{cls.__name__}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing command: {' '.join(command)}",
            flush=True,
        )

        matched_results = []

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
                if stdout_line_match_pattern:
                    match = stdout_line_match_pattern.match(stdout_line)
                    if match:
                        matched_results.append(match)

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        return matched_results

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
                self.repository_tmp_dir,
            ],
        )
        if self.commit_id is not None:
            self.exec_command(
                ["git", "switch", "--detach", self.commit_id],
                cwd=self.repository_tmp_dir,
            )

    def check_apt_packages_installed(self, package_names):
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

    def install_common(self):
        """Install common dependencies required for the environment."""
        self.exec_command(
            [self.venv_python, "-m", "pip", "install", "-e", "."],
            cwd=self.repository_tmp_dir,
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
            cwd=self.repository_tmp_dir,
        )

        if self.policy in self.THIRD_PARTY_PATHS:
            self.exec_command(
                [self.venv_python, "-m", "pip", "install", "-e", "."],
                cwd=os.path.join(
                    self.repository_tmp_dir,
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
                f"[{cls.__name__}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] The URL ends with 'dl=0'. Changing it to 'dl=1'."
            )
            dataset_url = dataset_url[: -len("dl=0")] + "dl=1"

        assert dataset_url.endswith(
            "dl=1"
        ), f"[{cls.__name__}] Error: The URL '{dataset_url}' does not end with 'dl=1'."
        return dataset_url

    def download_dataset(self, dataset_url):
        """Download the dataset from the specified URL."""
        zip_filename = "dataset.zip"
        dataset_temp_dir = tempfile.mkdtemp()
        self.exec_command(
            [
                "wget",
                "-O",
                os.path.join(dataset_temp_dir, zip_filename),
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
                cwd=dataset_temp_dir,
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
            glob.glob(os.path.join(dataset_temp_dir, "dataset", "*.rmb"))
        )
        print(
            f"[{self.__class__.__name__}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {rmb_items_count} data files in RMB format have been unzipped."
        )
        self.dataset_dir = os.path.join(
            dataset_temp_dir,
            "dataset/",
        )

    def get_dataset(self, dataset_location):
        if bool(urlparse(dataset_location).scheme):
            self.download_dataset(dataset_location)
            return
        elif os.path.isdir(dataset_location):
            self.dataset_dir = dataset_location
            return
        else:
            raise ValueError(f"Invalid: {dataset_location=}.")

    def train(self, args_file_train, seed):
        """Execute the training process using the specified arguments file."""
        assert self.dataset_dir

        if seed == -1:
            seed = int(time.time() * 1000000) % (2**32)
            print(
                f"[{self.__class__.__name__}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {seed=}"
            )

        command = [
            self.venv_python,
            os.path.join(self.repository_tmp_dir, "robo_manip_baselines/bin/Train.py"),
            self.policy,
            "--dataset_dir",
            self.dataset_dir,
            "--checkpoint_dir",
            os.path.join(
                self.checkpoint_temp_dir,
                "robo_manip_baselines/checkpoint_dir/",
                self.policy,
                self.env,
            ),
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if args_file_train:
            command.append("@" + args_file_train)

        self.exec_command(command)

    def rollout(self, args_file_rollout, rollout_duration, rollout_world_idx_list):
        """Execute the rollout using the provided configuration and duration."""

        task_success_list = []
        for world_idx in rollout_world_idx_list or [None]:
            command = [
                self.venv_python,
                os.path.join(
                    self.repository_tmp_dir, "robo_manip_baselines/bin/Rollout.py"
                ),
                self.policy,
                self.env,
                "--checkpoint",
                os.path.join(
                    self.checkpoint_temp_dir,
                    "robo_manip_baselines/checkpoint_dir/",
                    self.policy,
                    self.env,
                    "policy_last.ckpt",
                ),
                "--duration",
                f"{rollout_duration}",
                "--no_plot",
                "--no_render",
                "--save_last_image",
                "--output_image_dir",
                os.path.join(
                    self.checkpoint_temp_dir,
                    "robo_manip_baselines/checkpoint_dir/",
                    self.policy,
                    self.env,
                ),
            ]
            if world_idx is not None:
                command.extend(["--world_idx", f"{world_idx}"])
            if args_file_rollout:
                command.append("@" + args_file_rollout)

            reward_statuses = self.exec_command(
                command, stdout_line_match_pattern=ROLLOUT_RESULT_PRINT_PATTERN
            )
            assert len(reward_statuses) == 1, f"{len(reward_statuses)=}"
            assert reward_statuses[0].group(1) in (
                "success",
                "failure",
            ), f"{reward_statuses[0].group(1)=}"

            task_success_list.append(int(reward_statuses[0].group(1) == "success"))

        return task_success_list

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
                        f"[{self.__class__.__name__}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Existing file renamed to: {new_file_path}"
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
            f"[{self.__class__.__name__}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] File has been saved: {output_file_path}"
        )

    def start(
        self,
        dataset_location,
        args_file_train,
        args_file_rollout,
        rollout_duration,
        rollout_world_idx_list,
        seed=None,
    ):
        """Start all required processes."""
        with open(LOCK_FILE_PATH, "w", encoding="utf-8") as lock_file:
            print(f"[{self.__class__.__name__}] Lock file: {LOCK_FILE_PATH}")
            print(f"[{self.__class__.__name__}] Attempting to acquire lock...")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[{self.__class__.__name__}] Lock acquired. Starting processing...")
            try:
                self.git_clone()
                venv.create(
                    os.path.join(self.venv_python, "../../../venv/"), with_pip=True
                )
                self.check_apt_packages_installed(self.APT_REQUIRED_PACKAGE_NAMES)

                self.install_common()
                self.install_each_policy()

                self.get_dataset(dataset_location)
                self.train(args_file_train, seed)
                if not self.is_rollout_disabled:
                    task_success_list = self.rollout(
                        args_file_rollout, rollout_duration, rollout_world_idx_list
                    )
                    self.save_result(task_success_list)

                print(f"[{self.__class__.__name__}] Processing completed.")

            finally:
                print(f"[{self.__class__.__name__}] Releasing lock.")


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


def parse_argument():
    """Parse and return the command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a parser for the evaluation based on a specific commit.",
    )
    parser.add_argument(
        "policies",
        type=str,
        nargs="+",
        choices=["Mlp", "Sarnn", "Act", "DiffusionPolicy"],
        help="policies",
    )
    parser.add_argument(
        "env",
        type=str,
        help="environment",
    )
    parser.add_argument("-c", "--commit_id", type=str, required=False, default=None)
    parser.add_argument(
        "-u",
        "--repository_owner_name",
        type=str,
        default="isri-aist",
        help="github repository owner name (user or organization)",
    )
    parser.add_argument(
        "-d",
        "--dataset_location",
        type=str,
        required=True,
        help="specify URL of online storage or local directory",
    )
    parser.add_argument("--args_file_train", type=str, required=False)
    parser.add_argument("--args_file_rollout", type=str, required=False)
    parser.add_argument(
        "--rollout_duration",
        type=float,
        required=False,
        default=30.0,
        help="duration of rollout in seconds, disable rollout step if value is zero or negative",
    )
    parser.add_argument(
        "--rollout_world_idx_list",
        type=int,
        nargs="*",
        required=True,
        help="list of world indicies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="random seed; use -1 to generate different value on each run",
    )
    parser.add_argument(
        "--checkpoint_base_dir",
        type=str,
        required=False,
        default=None,
        help="directory for temporary checkpoint storage if specified",
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

    if parsed_args.daily_schedule_time:
        if not re.fullmatch(
            r"(?:[01]\d|2[0-3]):[0-5]\d", parsed_args.daily_schedule_time
        ):
            parser.error(
                f"Invalid time format for --daily_schedule_time: "
                f"'{parsed_args.daily_schedule_time}'. Expected HH:MM (00-23,00-59)."
            )

    return parsed_args


if __name__ == "__main__":
    args = parse_argument()

    is_rollout_disabled = args.rollout_duration <= 0.0
    if is_rollout_disabled:
        print(
            f"[{AutoEval.__class__.__name__}] rollout step is disabled because {args.rollout_duration=} (<= 0)."
        )

    def run_once():
        """Execute the start function."""
        for policy in args.policies:
            auto_eval = AutoEval(
                policy,
                args.env,
                args.commit_id,
                args.repository_owner_name,
                args.checkpoint_base_dir,
                is_rollout_disabled,
            )
            auto_eval.start(
                args.dataset_location,
                args.args_file_train,
                args.args_file_rollout,
                args.rollout_duration,
                args.rollout_world_idx_list,
                args.seed,
            )

    # Immediate execution mode (when -t is not specified)
    if not args.daily_schedule_time:
        run_once()
        print(f"[{AutoEval.__class__.__name__}] Completed one-time run. Exiting.")
        sys.exit(0)

    # Scheduling mode
    schedule.every().day.at(args.daily_schedule_time).do(run_once)
    print(
        f"[{AutoEval.__class__.__name__}] Scheduled daily run at {args.daily_schedule_time}. Waiting...",
        flush=True,
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n[{AutoEval.__class__.__name__}] Scheduler stopped by user.")
