"""Dynamic models for Grafa databases, generated from a YAML definition."""

from typing import Any

from grafa.models import (
    _BUILT_IN_ALLOWED_RELATIONSHIPS,
    _BUILT_IN_NODE_SUBTYPES,
    GrafaBaseNode,
    GrafaDatabase,
    Relationship,
)

import fsspec
import yaml
from pydantic import ConfigDict, Field, create_model

_VALID_OPTIONS: set[str] = {
    "link_to_chunk",
    "searchable",
    "semantic_search",
    "text_search",
    "embedding_template",
    "unique_name",
}


def map_type(type_str: str) -> Any:
    """Map custom type strings to Python types.

    Parameters
    ----------
    type_str : str
        The type string to map to a Python type

    Returns
    -------
    tuple
        A tuple containing the Python type and ... for required field

    Raises
    ------
    ValueError
        If the type string is not recognized
    """
    type_mapping = {
        "STRING": (str, ...),
        "INTEGER": (int, ...),
        "FLOAT": (float, ...),
        "BOOLEAN": (bool, ...),
        "LIST": (list[str], ...),
    }
    if type_str.upper() not in type_mapping:
        raise ValueError(f"Unrecognized type: {type_str}")
    return type_mapping[type_str.upper()]


def load_definitions(
    yaml_path: str | None = None,
    yaml_str: str | None = None,
    db_name: str | None = None,
) -> GrafaDatabase:
    """Load node type definitions from a YAML file or string and create dynamic Pydantic models for a database.

    This is an async version of the function that handles file I/O asynchronously using fsspec,
    supporting multiple file systems (local, s3, gcs, etc.).

    Parameters
    ----------
    yaml_path : str | None, optional
        Path to the YAML file containing node and relationship definitions.
        Can be a local path or URL (s3://, gs://, etc.)
    yaml_str : str | None, optional
        YAML string containing node and relationship definitions
    db_name : str | None, optional
        Name to use for the database. If not provided, will use name from YAML file

    Returns
    -------
    GrafaDatabase
        A Pydantic model representing the database configuration

    Raises
    ------
    ValueError
        If neither yaml_path nor yaml_str is provided, or there are invalid configurations.
    """
    user_defined_node_types = {}
    user_defined_node_relationships = []
    user_defined_node_relationships.extend(_BUILT_IN_ALLOWED_RELATIONSHIPS)
    original_name_to_class_name = {}  # Map original names to prefixed class names
    for node_type in _BUILT_IN_NODE_SUBTYPES.values():
        original_name_to_class_name[node_type.__name__] = node_type.__name__

    # Load the node definition from either file or string
    if yaml_path is not None:
        with fsspec.open(yaml_path) as f:
            yaml_str = f.read()
    elif yaml_str is None:
        raise ValueError("Either yaml_path or yaml_str must be provided")

    definition = yaml.safe_load(yaml_str)

    database_info = definition.get("database", {})
    if db_name is None:
        if "name" not in database_info:
            raise ValueError("Database name is required")
        db_name = database_info["name"]
    if "description" not in database_info:
        raise ValueError("Database description is required")

    node_types = definition.get("node_types", {})
    relationships = definition.get("relationships", [])

    # Format database name for class prefixing - remove spaces, make title case
    db_prefix = db_name.replace(" ", "").title()

    # Iterate over each node type and create a Pydantic model dynamically
    for idx, (node_name, node_info) in enumerate(node_types.items()):
        if node_name in _BUILT_IN_NODE_SUBTYPES:
            raise ValueError(
                f"{node_name} is a reserved type name and cannot be used as a type"
            )
        if node_name.lower() == "user_defined":
            raise ValueError(
                "USER_DEFINED is a reserved label and cannot be used as a type"
            )
        fields = node_info.get("fields", {})
        options = node_info.get("options", {})
        description = node_info.get("description", "")

        # Create a namespaced class name
        prefixed_name = f"{db_prefix}_{node_name}"
        original_name_to_class_name[node_name] = prefixed_name

        # Prepare additional fields from node definition
        additional_fields = {}
        model_config = {}

        for field_name, field_info in fields.items():
            field_type_str = field_info.get("type", "STRING").upper()
            if field_name in GrafaBaseNode.model_config["reserved_fields"]:
                raise ValueError(f"{field_name} cannot be overwritten")
            py_type, default = map_type(field_type_str)
            field_description = field_info.get("description", "")
            additional_fields[field_name] = (
                py_type,
                Field(default, description=field_description),
            )

        # Add options as model config entries
        model_config[
            "link_to_chunk"
        ] = True  # If true, the node will be linked to its referenced chunks
        model_config[
            "semantic_search"
        ] = True  # If true, the node will be indexed for semantic search
        model_config[
            "text_search"
        ] = True  # If true, the node will be indexed for text search
        model_config[
            "embedding_template"
        ] = None  # If provided, it will be used as the embedding template for the node
        model_config["unique_name"] = True  # If true, the node must have a unique name
        for option_name, option_value in options.items():
            if option_name not in _VALID_OPTIONS:
                raise ValueError(f"Invalid option: {option_name}")
            elif option_name == "searchable":
                model_config["semantic_search"] = option_value
                model_config["text_search"] = option_value
            else:
                model_config[option_name] = option_value
        model_config["grafa_database_name"] = db_name
        model_config["grafa_original_type_name"] = node_name
        model_config["user_defined"] = True
        additional_fields["model_config"] = ConfigDict(**model_config)

        # Create the dynamic model using Pydantic's create_model with prefixed name
        model = create_model(
            prefixed_name,
            __base__=GrafaBaseNode,  # Inherit from BaseNode to include its fields
            __doc__=description,  # Add the description as the model's docstring
            **additional_fields,
        )

        # Add the model to the dictionary using original name as key
        user_defined_node_types[node_name] = model
        if model_config["link_to_chunk"]:
            user_defined_node_relationships.append(
                Relationship(
                    from_type="GrafaChunk",
                    to_type=node_name,
                    type="REFERENCES",
                    description=f"A {node_name} is referenced by a chunk",
                )
            )

    # Update relationships to use prefixed class names
    for relationship in relationships:
        # Validate that both from and to node types exist
        if relationship["from"] not in user_defined_node_types:
            raise ValueError(
                f"Node type '{relationship['from']}' referenced in relationship does not exist"
            )
        if relationship["to"] not in user_defined_node_types:
            raise ValueError(
                f"Node type '{relationship['to']}' referenced in relationship does not exist"
            )

        # Use original names in the relationship for user-friendly API
        user_defined_node_relationships.append(
            Relationship(
                from_type=relationship["from"],
                to_type=relationship["to"],
                type=relationship["type"],
                description=relationship["description"],
                user_defined=True,
            )
        )

    return GrafaDatabase(
        name=db_name,
        description=database_info["description"],
        language=database_info["language"]
        if "language" in database_info
        else "English",
        node_types={**_BUILT_IN_NODE_SUBTYPES, **user_defined_node_types},
        allowed_relationships=user_defined_node_relationships,
        yaml=yaml_str,
        original_name_to_class_name=original_name_to_class_name,
        grafa_database_name=db_name,
    )
