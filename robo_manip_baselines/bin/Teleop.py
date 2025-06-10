import argparse
import importlib
import importlib.util
import os
import sys

import yaml


def main():
    env_utils_spec = importlib.util.spec_from_file_location(
        "EnvUtils",
        os.path.join(os.path.dirname(__file__), "..", "common/utils/EnvUtils.py"),
    )
    env_utils_module = importlib.util.module_from_spec(env_utils_spec)
    env_utils_spec.loader.exec_module(env_utils_module)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for the teleop switching between different environments. The actual arguments are handled by another internal argument parser.",
        add_help=False,
    )
    parser.add_argument(
        "env",
        type=str,
        help="environment",
        nargs="?",
        default=None,
        choices=env_utils_module.get_env_names(),
    )
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and continue"
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.env is None:
        parser.print_help()
        return
    elif args.help:
        parser.print_help()
        print("\n================================\n")
        sys.argv += ["--help"]

    if "Isaac" in args.env:
        from isaacgym import (
            gymapi,  # noqa: F401
            gymtorch,  # noqa: F401
            gymutil,  # noqa: F401
        )
    if args.env.endswith("Vec"):
        from robo_manip_baselines.teleop import TeleopBaseVec as TeleopBase
    else:
        from robo_manip_baselines.teleop import TeleopBase

    operation_module = importlib.import_module(
        f"robo_manip_baselines.envs.operation.Operation{args.env}"
    )
    OperationEnvClass = getattr(operation_module, f"Operation{args.env}")

    # The order of parent classes must not be changed in order to maintain the method resolution order (MRO)
    class Teleop(OperationEnvClass, TeleopBase):
        pass

    if args.config is None:
        config = {}
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    teleop = Teleop(**config)
    teleop.run()


if __name__ == "__main__":
    main()
