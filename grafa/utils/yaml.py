"""Utilities for interacting with YAML files."""

import yaml


def get_database_name(yaml_path: str | None = None, yaml_str: str | None = None) -> str:
    """Get the database name from a YAML configuration string.

    Parameters
    ----------
    yaml_path : str | None, optional
        Path to the YAML file containing the database configuration.
    yaml_str : str | None, optional
        YAML string containing the database configuration.

    Returns
    -------
    str
        The name of the database defined in the YAML

    Raises
    ------
    ValueError
        If the database name is not defined in the YAML or if yaml_str is None
    """
    # Load the node definition from either file or string
    if yaml_path is not None:
        with open(yaml_path) as f:
            yaml_str = f.read()
    elif yaml_str is None:
        raise ValueError("Either yaml_path or yaml_str must be provided")

    definition = yaml.safe_load(yaml_str)
    database_info = definition.get("database", {})
    if "name" not in database_info:
        raise ValueError("Database name is required")
    return database_info["name"]
