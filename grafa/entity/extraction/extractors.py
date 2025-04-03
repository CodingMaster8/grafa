"""Extractors for entities and relationships in the knowledge graph."""

from typing import List, Type

from grafa.models import Entity, GrafaBaseNode, Relationship

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import langfuse_context, observe

from .models import EntityOutput, RelationshipOutput
from .prompt import (
    ENTITY_EXTRACTOR_PROMPT_TEMPLATE,
    RELATIONSHIP_EXTRACTOR_PROMPT_TEMPLATE,
)


@observe
async def extract_entities(
    chunk_representation: str,
    possible_entity_models: List[Type[Entity]],
    llm: Runnable,
    language: str,
) -> EntityOutput:
    """
    Extract relevant entities and their properties from a chunk of text.

    Parameters
    ----------
    chunk_representation : str
        Data of the chunk to analyze
    possible_entity_models : List[Type[Entity]]
        The possible entity models to extract
    llm : Runnable
        Language model to use for entity extraction
    language: str
        Desired output language

    Returns
    -------
    output_model : EntityOutput
        The output model to use for entity extraction
    """
    output_model = EntityOutput.get_pydantic_template(possible_entity_models)
    parser = PydanticOutputParser(pydantic_object=output_model)
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    prompt = PromptTemplate(
        template=ENTITY_EXTRACTOR_PROMPT_TEMPLATE,
        input_variables=["chunk_representation"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "language": language,
        },
    )

    chain = prompt | llm | parser

    return await chain.ainvoke(
        {"chunk_representation": chunk_representation},
        config={"callbacks": [langfuse_handler]},
    )


@observe
async def extract_relationships(
    chunk_representation: str,
    entities: List[GrafaBaseNode],
    allowed_relationships: List[Relationship],
    llm: Runnable,
) -> RelationshipOutput:
    """
    Extract relationships between entities based on allowed relationship types, in the chunk of text.

    Parameters
    ----------
    chunk_representation : str
        Data of the chunk to analyze
    entities : List[GrafaBaseNode]
        List of entities extracted from a chunk of text
    allowed_relationships : List[Relationship]
        List of relationship types that are allowed in the knowledge graph
    llm : Runnable
        Language model to use for relationship extraction

    Returns
    -------
    RelationshipOutput
        Pydantic model containing the extracted relationships
    """
    allowed_relationship_types = [
        relationship.type for relationship in allowed_relationships
    ]
    output_model = RelationshipOutput.get_pydantic_template(allowed_relationship_types)
    parser = PydanticOutputParser(pydantic_object=output_model)
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    prompt = PromptTemplate(
        template=RELATIONSHIP_EXTRACTOR_PROMPT_TEMPLATE,
        input_variables=[
            "chunk_representation",
            "allowed_relationships_str",
            "entities_str",
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    allowed_relationships_str = "\n".join(
        [str(relationship) for relationship in allowed_relationships]
    )
    entities_str = str(
        {
            i: entity.name
            + " (Type: "
            + entity.model_config.get("grafa_original_type_name")
            + " Synonyms: "
            + str(entity.synonyms)
            + ")"
            for i, entity in enumerate(entities)
        }
    )

    return await chain.ainvoke(
        {
            "chunk_representation": chunk_representation,
            "allowed_relationships_str": allowed_relationships_str,
            "entities_str": entities_str,
        },
        config={"callbacks": [langfuse_handler]},
    )
