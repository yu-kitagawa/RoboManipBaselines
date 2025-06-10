import argparse
import importlib
import re
import sys


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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for the train switching between different policies and environments. The actual arguments are handled by another internal argument parser.",
        fromfile_prefix_chars="@",
        add_help=False,
    )
    parser.add_argument(
        "policy",
        type=str,
        nargs="?",
        default=None,
        choices=["Mlp", "Sarnn", "Act", "MtAct", "DiffusionPolicy"],
        help="policy",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and continue"
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.policy is None:
        parser.print_help()
        return
    elif args.help:
        parser.print_help()
        print("\n================================\n")
        sys.argv += ["--help"]

    policy_module = importlib.import_module(
        f"robo_manip_baselines.policy.{camel_to_snake(args.policy)}"
    )
    TrainPolicyClass = getattr(policy_module, f"Train{args.policy}")

    train = TrainPolicyClass()
    train.run()
    train.close()


if __name__ == "__main__":
    main()
