"""Deduplication of entities."""

import random
import string

from grafa.models import GrafaBaseNode

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import langfuse_context, observe

from .parser import IntegerOutputParser
from .prompt import DEDUPLICATION_PROMPT_TEMPLATE, MERGE_ENTITY_PROMPT_TEMPLATE


@observe
async def deduplicate_entity(
    entity: GrafaBaseNode,
    similar_entities: list[GrafaBaseNode],
    llm: Runnable,
    language: str,
) -> GrafaBaseNode:
    """
    Compare an entity with a list of similar entities and will return the entity that is the result of the deduplication.

    A deduplication is either:
    - The entity is a duplicate of an existing entity, in which case the LLM will return the existing entity, with updated properties.
    - The entity is unique, in which case the LLM will return a new entity node.

    Parameters
    ----------
    entity : GrafaBaseNode
        New Concept Entity identified
    similar_entities : list[GrafaBaseNode]
        list of similar existing concept nodes
    llm : Runnable
        The LLM to use for deduplication and merging
    language: str
        The language of the database

    Returns
    -------
    new_entity_node: GrafaBaseNode
        The entity node that is the result of the deduplication
    """
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    parser = IntegerOutputParser()

    prompt = PromptTemplate(
        template=DEDUPLICATION_PROMPT_TEMPLATE,
        input_variables=["entity", "similar_entities"],
    )

    chain = prompt | llm | parser

    response = await chain.ainvoke(
        {
            "entity": entity.get_embedding_text(),
            "similar_entities": str(
                {
                    i: similar_entity.get_embedding_text()
                    for i, similar_entity in enumerate(similar_entities)
                }
            ),
        },
        config={"callbacks": [langfuse_handler]},
    )

    if response == -1:
        similar_entity_names = set(
            [similar_entity.name for similar_entity in similar_entities]
        )
        if entity.model_config.get("unique_name", False):
            while entity.name in similar_entity_names:
                entity.name = (
                    entity.name
                    + " ("
                    + "".join(random.choices(string.ascii_letters, k=3))
                    + ")"
                )
        return entity
    else:
        return await _merge_entity(entity, similar_entities[response], llm, language)


@observe
async def _merge_entity(
    entity: GrafaBaseNode, similar_entity: GrafaBaseNode, llm: Runnable, language: str
) -> GrafaBaseNode:
    """
    Merge an entity with a similar entity.

    Parameters
    ----------
    entity : GrafaBaseNode
        The entity to merge
    similar_entity : GrafaBaseNode
        The similar entity to merge with
    llm : Runnable
        The LLM to use for merging
    language: str
        The language of the database

    Returns
    -------
    merged_entity : GrafaBaseNode
        The merged entity
    """
    langfuse_handler = langfuse_context.get_current_langchain_handler()
    parser = PydanticOutputParser(pydantic_object=entity.get_pydantic_template())

    prompt = PromptTemplate(
        template=MERGE_ENTITY_PROMPT_TEMPLATE,
        input_variables=["entity", "old_entity"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "language": language,
        },
    )

    chain = prompt | llm | parser

    response = await chain.ainvoke(
        {
            "entity": entity.get_embedding_text(),
            "old_entity": similar_entity.get_embedding_text(),
        },
        config={"callbacks": [langfuse_handler]},
    )

    update_fields = response.model_dump()
    update_fields.pop("name")
    update_fields.pop("grafa_original_type_name")

    # Track if any fields were actually updated
    fields_updated = False
    for field, value in update_fields.items():
        if getattr(similar_entity, field) != value:
            setattr(similar_entity, field, value)
            fields_updated = True

    # Only increment version if fields were actually updated
    if fields_updated:
        similar_entity.version += 1
    return similar_entity
