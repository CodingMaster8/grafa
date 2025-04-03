"""
Basic Chunker.

Uses LangChain's RecursiveCharacterTextSplitter to divide text into chunks
of specified size with overlap between consecutive chunks.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .chunking_models import ChunkOutput


def chunk_text(document: str, chunk_size: int = 2000, chunk_overlap: int = 200):
    """Chunk text content into smaller segments.

    Uses LangChain's RecursiveCharacterTextSplitter to divide text into chunks
    of specified size with overlap between consecutive chunks.

    Parameters
    ----------
    document : str
        The content of the file to be chunked
    chunk_size : int, default=2000
        The size of each chunk
    chunk_overlap : int, default=200
        The overlap between consecutive chunks

    Returns
    -------
    ChunkOutput
        Object containing list of text chunks and an empty summary
    """
    #LangChain helper function
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    length_function = len,
    is_separator_regex = False,
    )

    item_text_chunks = text_splitter.split_text(document) # split the text into chunks

    return ChunkOutput(chunks=item_text_chunks, summary="")
