"""Semantic Chunking but using sentences ending in dots, etc."""
import re

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import observe

from ..chunking_models import ChunkOutput
from ..chunking_utils import calculate_cosine_distances, combine_sentences
from .prompt import SEMI_AGENTIC_CHUNKING


@observe
def _agentic_separator(
    llm: Runnable = None,
    separator: str = "",
    before_sentence: str = "",
) -> bool:
    """Determine if a sentence should be part of a new chunk based on semantic meaning.

    Uses an LLM to evaluate whether the separator sentence should start a new chunk
    or continue the current chunk containing the before_sentence.

    Parameters
    ----------
    llm : Runnable
        Language model chain to evaluate the semantic relationship
    separator : str
        The sentence being evaluated as a potential chunk boundary
    before_sentence : str
        The preceding sentence or context

    Returns
    -------
    bool
        True if separator should start a new chunk, False if it belongs with before_sentence
    """
    values = {"SEPARATOR": separator, "BEFORE_SENTENCE": before_sentence}

    prompt = PromptTemplate(
        template=SEMI_AGENTIC_CHUNKING,
        input_variables=["SEPARATOR", "BEFORE_SENTENCE"],
    )

    chain = prompt | llm

    response = chain.invoke(values).content.strip()

    if response.lower() not in ["true", "false"]:
        raise ValueError(f"The LLM response '{response}' is not a valid boolean")

    return response.lower() == "true"


def semantic_agentic_chunking(
    document: str,
    llm: Runnable,
    embedding_model: Embeddings,
    use_percentile: bool = True,
    percentile_threshold: int = 95,
    threshold_value: float = 0.1,
) -> ChunkOutput:
    """
    Perform semantic chunking with agentic validation on a document.

    This function divides the document into sentences, combines short sentences with
    semantically similar neighbors, calculates embeddings, and identifies natural
    breakpoints. It then uses an LLM to validate each potential breakpoint.

    Parameters
    ----------
    document : str
        The content of the document to be chunked
    llm : Runnable
        Language model to use for agentic validation of chunk boundaries
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

    # Function to calculate cosine distances
    def sentence_cosine_distances(middle_embedding, prev_embedding, next_embedding):
        distances = {}
        middle_array = np.array(
            middle_embedding
        )  # Convert middle_embedding to numpy array

        if prev_embedding is not None:
            prev_array = np.array(
                prev_embedding
            )  # Convert prev_embedding to numpy array
            distances["Before"] = np.linalg.norm(middle_array - prev_array)

        if next_embedding is not None:
            next_array = np.array(
                next_embedding
            )  # Convert next_embedding to numpy array
            distances["Next"] = np.linalg.norm(middle_array - next_array)

        # Return the label of the smallest distance
        if distances:
            return min(distances, key=distances.get)
        return None

    # Process sentences
    i = 0
    while i < len(single_sentences_list):
        sentence = single_sentences_list[i]

        if len(sentence) < 35:
            print(i, sentence)
            middle_sentence_embedding = embedding_model.embed_query(sentence)

            # Handle previous and next sentences with default values
            prev_embedding = None
            next_embedding = None

            if i > 0:  # Previous sentence exists
                prev_sentence = single_sentences_list[i - 1]
                prev_embedding = embedding_model.embed_query(prev_sentence)
            else:
                prev_sentence = ""

            if i < len(single_sentences_list) - 1:  # Next sentence exists
                next_sentence = single_sentences_list[i + 1]
                next_embedding = embedding_model.embed_query(next_sentence)
            else:
                next_sentence = ""

            print("BEFORE ----")
            print(prev_sentence)
            print("----")
            print("NEXT ---")
            print(next_sentence)
            print("----")

            # Calculate cosine distances and determine where to append
            results = sentence_cosine_distances(
                middle_sentence_embedding, prev_embedding, next_embedding
            )

            if results == "Before":
                print(
                    f"Sentence : {sentence} Should be part of previous sentence : {prev_sentence}"
                )
                single_sentences_list[i - 1] = prev_sentence + " " + sentence
                single_sentences_list.pop(i)
                i -= 1  # Adjust index after popping
            elif results == "Next":
                print(
                    f"Sentence : {sentence} --- Should be part of next sentence  :  --- {next_sentence}"
                )
                single_sentences_list[i + 1] = sentence + " " + next_sentence
                single_sentences_list.pop(i)
            else:
                print(f"No suitable match found for sentence: {sentence}")

        else:
            i += 1  # Move to the next sentence if no merging is needed

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

    print("DISTANCES")
    print(distances)

    if use_percentile:
        breakpoint_distance_threshold = np.percentile(distances, percentile_threshold)
        print("Using 95th Percentile Threshold")
    else:
        breakpoint_distance_threshold = threshold_value

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]  # The indices of those breakpoints on your list
    print("Indices Above Threshold")
    print(indices_above_thresh)

    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        print("----------SEPARATOR----------")
        separator = sentences[index]["sentence"]
        print(separator)
        print("-----------BEFORE_SENTENCE-------------")
        before_sentence = sentences[index - 1]["sentence"]
        print(before_sentence)

        # Agentic Validation by LLM
        response = _agentic_separator(
            llm, separator=separator, before_sentence=before_sentence
        )
        print("---------LLM RESPONSE--------")
        print(response)

        if response:
            # The end index is the current breakpoint
            end_index = index - 1

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index
        else:
            print(f"Chunking at sentence {index} was avoided by LLM dictamen")

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
        chunks.append(combined_text)

    # Return ChunkOutput with chunks and empty summary
    return ChunkOutput(chunks=chunks, summary="")

    # TODO: Handle case where separator is a sentence in index 0
