"""Utility functions for the agentic chunker."""
import tiktoken


def max_split_text_by_tokens(max_tokens_chunks, text_input):
    """Split the input text into chunks based on max_tokens_chunks.

    Parameters
    ----------
    max_tokens_chunks : int
        Maximum number of tokens per chunk.
    text_input : list of str
        Input text lines to split.

    Returns
    -------
    list of str
        List of text lines that fit within the token limit.
    """
    tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')

    chunk = []
    current_token_count = 0

    for line in text_input:
        # Tokenize the line and count the tokens
        tokenized_line = tokenizer.encode(line)
        line_token_count = len(tokenized_line)

        # Check if adding this line exceeds the max token limit
        if current_token_count + line_token_count > max_tokens_chunks:
            # If it does, finalize the current chunk and start a new one
            return chunk

        # Add the current line to the chunk
        chunk.append(line)
        current_token_count += line_token_count

    # Return the chunk even if it doesn't reach the max token limit
    return chunk


def calculate_tokens_document(document):
    """Calculate the total token count for a list of strings.

    Parameters
    ----------
    document : list of str
        List of text lines to calculate tokens for.

    Returns
    -------
    int
        Total token count for all lines.
    """
    # Initialize the tokenizer (using GPT-4's tokenizer)
    tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')

    token_count = 0

    for line in document:
        # Tokenize the line and count the tokens
        tokenized_line = tokenizer.encode(line)
        line_token_count = len(tokenized_line)

        token_count += line_token_count

    return token_count
