"""Implementation for agentic chunking."""

from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import langfuse_context, observe

from ..chunking_models import ChunkOutput
from .models import AgenticChunkingOutput
from .prompt import AGENTIC_CHUNKING
from .utils import calculate_tokens_document, max_split_text_by_tokens


@observe
async def _agentic_chunker(
    current_summary: str,
    llm: Runnable,
    input_text: List[str],
    output_language: str = "Spanish",
) -> AgenticChunkingOutput:
    """
    Process text to determine optimal chunking points and update document summary.

    Uses an LLM to analyze text and determine where to split content into semantically
    coherent chunks, while maintaining and updating a running summary.

    Parameters
    ----------
    current_summary : str
        The current summary of previously processed text
    llm : Runnable
        Language model to use for chunking decisions
    input_text : List[str]
        List of text lines to be analyzed for chunking
    output_language : str, default="Spanish"
        Language for the output summary


    Returns
    -------
    AgenticChunkingOutput
        Contains the updated summary and line number where text should be cut

    Raises
    ------
    ValueError
        If no language model is provided
    """
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=AgenticChunkingOutput)

    values = {
        "current_summary": current_summary,
        "input_text": input_text,
        "output_language": output_language,
    }

    prompt = PromptTemplate(
        template=AGENTIC_CHUNKING,
        input_variables=["current_summary", "input_text", "output_language"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    return await chain.ainvoke(values, config={"callbacks": [langfuse_handler]})


@observe
async def agentic_chunking(
    document: str,
    llm: Runnable,
    max_token_chunk_size: int = 100,
    verbose: bool = False,
    output_language: str = "Spanish",
) -> ChunkOutput:
    """
    Perform agentic chunking on a document, producing semantically meaningful chunks.

    This function divides a document into chunks based on token size limits and uses
    an LLM agent to determine optimal splitting points while maintaining semantic coherence.
    It also generates and updates a summary of the document as it processes.

    Parameters
    ----------
    document : str
        The text content of the document to be chunked
    llm : Runnable
        Language model to use for agentic chunking decisions
    max_token_chunk_size : int, default=100
        Maximum number of tokens allowed in each chunk
    verbose : bool, default=False
        Whether to print progress information during chunking
    output_language : str, default="Spanish"
        Language for the output summary

    Returns
    -------
    ChunkOutput
        Object containing list of text chunks and a summary of the document
    """
    if verbose:
        print("Chunking for document started...")

    # Initial values
    chunks = []
    current_summary = ""
    double_max_token_size = max_token_chunk_size * 2
    token_count = float("inf")  # Ensures agentic chunking is run at least once.

    # Convert document to List of lines
    document = document.splitlines()
    # Remove empty strings and strip whitespace
    document = [line.strip() for line in document if line.strip()]

    # While document has a length in tokens bigger than chunk size. If not that is final chunk!
    while token_count > max_token_chunk_size:
        if verbose:
            print(f"There are {len(document)} lines on the document")

        # Get the chunk 2 times bigger of max token chunk size
        chunk = max_split_text_by_tokens(double_max_token_size, document)

        # index the chunk for LLM readability
        indexed_chunk = [f"{i+1}. -  {line}" for i, line in enumerate(chunk)]

        # Make the agent call
        output = await _agentic_chunker(
            current_summary, llm, indexed_chunk, output_language
        )

        # agent defines where to split document and updates the summary
        cut = output.line_number
        current_summary = output.updated_summary

        if verbose:
            print(f"Cutting Document on line {cut}")
            print(f"Actual Summary : {current_summary}")

        new_chunk = document[:cut]

        # Join strings on the list
        new_chunk = " ".join(new_chunk)

        chunks.append(new_chunk)
        document = document[cut:]

        token_count = calculate_tokens_document(document)

    if len(document) != 0:
        document = " ".join(document)
        chunks.append(document)

    return ChunkOutput(chunks=chunks, summary=current_summary)
