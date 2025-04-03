"""Extractors for entities and relationships in the knowledge graph."""

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import langfuse_context, observe

from .models import ConceptIdentificationOutput
from .prompt import CONCEPT_EXTRACTOR_PROMPT_TEMPLATE


@observe
async def extract_concepts(
    query: str,
    llm: Runnable,
) -> ConceptIdentificationOutput:
    """
    Extract concepts from a query.

    Parameters
    ----------
    query : str
        Query string
    llm : Runnable
        Language model to use for concept extraction

    Returns
    -------
    output_model : ConceptIdentificationOutput
        The output model to use for concept identification
    """
    parser = PydanticOutputParser(pydantic_object=ConceptIdentificationOutput)
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    prompt = PromptTemplate(
        template=CONCEPT_EXTRACTOR_PROMPT_TEMPLATE,
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )

    chain = prompt | llm | parser

    return await chain.ainvoke(
        {"query": query},
        config={"callbacks": [langfuse_handler]},
    )
