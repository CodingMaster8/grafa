"""Prompt for creating an embedding for a chunk of text."""

import re

# Import GrafaBaseNode type for type hints only
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from grafa.models import GrafaBaseNode

GRAFA_CHUNK_EMBEDDING_TEMPLATE = """
<text_excerpt>
{content}
</text_excerpt>
<tags>
{tags}
</tags>
<origin_document_info>
<name>
{original_document_name}
</name>
<context>
{original_document_context}
</context>
<summary>
{original_document_summary}
</original_document_summary>
</origin_document_info>
"""

def populate_template(template: str, node: 'GrafaBaseNode', extra_fields: dict = None) -> str:
    """Populate the template with the node information.

    Parameters
    ----------
    template : str
        The template to populate
    node : GrafaBaseNode
        The node to populate the template with
    extra_fields : dict
        Extra fields to populate the template with

    Returns
    -------
    str
        The populated template
    """
    # Extract all placeholders from the template
    placeholders = re.findall(r'{([^{}]*)}', template)

    # Get node data and combine with extra fields
    node_data = node.model_dump()
    combined_data = {**node_data}
    if extra_fields:
        combined_data.update(extra_fields)

    # Check if all placeholders are available in the combined data
    missing_placeholders = [p for p in placeholders if p not in combined_data]
    if missing_placeholders:
        raise ValueError(f"Missing required placeholders in data: {', '.join(missing_placeholders)}")

    # Only include the fields that are needed for the template
    template_data = {k: combined_data[k] for k in placeholders if k in combined_data}

    return template.format(**template_data)

def get_default_embedding_template(cls: Type['GrafaBaseNode']) -> str:
    """Get the default embedding template for a node class.

    Parameters
    ----------
    cls : Type[GrafaBaseNode]
        The node class to generate the default embedding template for

    Returns
    -------
    str
        The default embedding template for the node class
    """
    # Get all fields that are not in metadata_attributes
    metadata_attributes = cls.model_config.get("metadata_attributes", set())
    non_metadata_fields = [field for field in cls.model_fields.keys()
                          if field not in metadata_attributes]

    # Create an XML template with all non-metadata fields
    xml_template = "<document>\n"
    xml_template += f"<document_type>{cls.model_config.get('grafa_original_type_name', cls.__name__)}</document_type>\n"
    for field in non_metadata_fields:
        xml_template += f"<{field}>{{{field}}}</{field}>\n"
    xml_template += "</document>"
    return xml_template
