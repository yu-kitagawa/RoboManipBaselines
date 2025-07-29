import re


def remove_prefix(s, prefix):
    """Support Python 3.8 and below."""
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s


def remove_suffix(s, suffix):
    """Support Python 3.8 and below."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


def camel_to_snake(name):
    """Converts camelCase or PascalCase to snake_case (also converts the first letter to lowercase)."""
    name = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", name
    )  # Insert '_' between a lowercase/number and an uppercase letter
    name = re.sub(
        r"([a-z])([0-9])", r"\1_\2", name
    )  # Insert '_' between a lowercase and a number letter
    name = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name
    )  # Insert '_' between consecutive uppercase letters followed by a lowercase letter
    name = name[0].lower() + name[1:]  # Convert the first letter to lowercase
    return name.lower()
