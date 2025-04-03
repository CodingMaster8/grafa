"""Implementation for semantic chunking."""

import re

import numpy as np
from langchain_core.embeddings import Embeddings

from ..chunking_models import ChunkOutput
from ..chunking_utils import calculate_cosine_distances, combine_sentences


def semantic_chunking(
    embedding_model: Embeddings,
    document: str,
    use_percentile: bool = True,
    percentile_threshold: int = 95,
    threshold_value: float = 0.1,
) -> ChunkOutput:
    """
    Perform semantic chunking on a document by splitting it into coherent chunks based on semantic similarity.

    This function divides the document into sentences, combines them into meaningful units,
    calculates embeddings for each unit, and then identifies natural breakpoints where
    the semantic similarity changes significantly.

    Parameters
    ----------
    embedding_model : Embeddings
        The embedding model to use for semantic chunking
    document : str
        The content of the document to be chunked
    use_percentile : bool, default=True
        Whether to use percentile-based thresholding for determining breakpoints
    percentile_threshold : int, default=95
        The percentile value to use as threshold when use_percentile is True
    threshold_value : float, default=0.1
        The absolute threshold value to use when use_percentile is False

    Returns
    -------
    ChunkOutput
        Object containing list of text chunks and an empty summary
    """
    # Divide the text into n sentences
    # Improved regex to handle listings
    pattern = r"(?<!\d)\.(?=\s+|$)|[?!](?=\s+|$)"
    single_sentences_list = re.split(pattern, document)

    # Remove empty strings and strip whitespace
    single_sentences_list = [
        sentence.strip() for sentence in single_sentences_list if sentence.strip()
    ]

    sentences = [
        {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
    ]

    sentences = combine_sentences(sentences)

    embeddings = embedding_model.embed_documents(
        [x["combined_sentence"] for x in sentences]
    )

    for i, sentence in enumerate(sentences):
        sentence["combined_sentence_embedding"] = embeddings[i]

    distances, sentences = calculate_cosine_distances(sentences)

    if use_percentile:
        breakpoint_distance_threshold = np.percentile(distances, percentile_threshold)
    else:
        breakpoint_distance_threshold = threshold_value

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]  # The indices of those breakpoints on your list

    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index - 1

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group])
        chunks.append(combined_text)

        # Update the start index for the next group
        start_index = index

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
        chunks.append(combined_text)

    # Return ChunkOutput with chunks and empty summary
    return ChunkOutput(chunks=chunks, summary="")
