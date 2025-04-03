"""Module for tagging chunks."""
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import observe

from .models import TaggingOutput
from .prompt import TAGGING_PROMPT


@observe
async def tag_chunk(
    text: str, llm: Runnable, output_language: str = "English"
) -> TaggingOutput:
    """
    Extract tags from a text chunk using a language model.

    This function creates a prompt using a predefined tagging template, invokes the language model
    to extract relevant tags from the provided text, and parses the output into a TaggingOutput object.

    Parameters
    ----------
    text : str
        The text chunk to extract tags from.
    llm : Runnable
        The language model to use for extracting tags.
    output_language : str, default="English"
        The language of the output tags.

    Returns
    -------
    TaggingOutput
       Output of the tagging model
    """
    # Initialize a parser with the TaggingOutput model
    parser = PydanticOutputParser(pydantic_object=TaggingOutput)

    # Create a prompt for tagging based on the TAGGING_PROMPT template
    prompt = PromptTemplate(
        template=TAGGING_PROMPT,
        input_variables=["document", "output_language"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Construct the chain: prompt -> LLM -> parser
    chain = prompt | llm | parser

    return await chain.ainvoke({"document": text, "output_language": output_language})
