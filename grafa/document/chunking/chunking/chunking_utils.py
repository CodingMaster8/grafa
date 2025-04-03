"""Utility functions for chunkers."""

from sklearn.metrics.pairwise import cosine_similarity


def combine_sentences(sentences, buffer_size=1):
    """Combine each sentence with its surrounding context.

    For each sentence in the input list, creates a combined version that includes
    a specified number of sentences before and after it as context.

    Parameters
    ----------
    sentences : list of dict
        List of sentence dictionaries, each containing at least a 'sentence' key
    buffer_size : int, default=1
        Number of sentences to include before and after the current sentence

    Returns
    -------
    list of dict
        The input list with a new 'combined_sentence' key added to each dictionary
    """
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences):
    """Calculate cosine distances between consecutive sentence embeddings.

    Computes the semantic distance between each sentence and the next one
    in the sequence using cosine distance between their embeddings.

    Parameters
    ----------
    sentences : list of dict
        List of sentence dictionaries, each containing a 'combined_sentence_embedding' key

    Returns
    -------
    tuple
        - list of float: Cosine distances between consecutive sentences
        - list of dict: The input list with 'distance_to_next' added to each dictionary
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences
