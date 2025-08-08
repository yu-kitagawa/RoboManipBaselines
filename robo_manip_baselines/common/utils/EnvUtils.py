import importlib
import pkgutil


def get_env_names(operation_parent_module_str="robo_manip_baselines.envs.operation"):
    operation_parent_module = importlib.import_module(operation_parent_module_str)
    operation_module_prefix = "Operation"

    env_names = [
        name[len(operation_module_prefix) :]
        for _, name, _ in pkgutil.iter_modules(operation_parent_module.__path__)
        if name.startswith(operation_module_prefix)
    ]

    return env_names
