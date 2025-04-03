"""Prompts for the semantic agentic chunker."""

SEMI_AGENTIC_CHUNKING= """
You will be provided with two sentences.
Your taks is to recognize if the second sentence is part of the chunk of the first sentence or if it should be a new chunk.
If the second sentence is highly correlated to the first one you MUST OUTPUT False.
If the second sentence is not correlated to the first one and should be part of a new chunk you MUST OUTPUT True.

You must output ONLY the boolean value and it must be a Python-style literal.

<guidelines>
 Here's a comprehensive list of rules for finding the concepts:

    1. ** If the two sentences talk about the same topic they probably are correlated
    2. ** If each sentence talk about something different they probably are not correlated.
    3. ** Take into consideration the semantic meaning of the sentences.
    4. ** The separator sentence is where the cut of chunks is being made.

Output Example: True

</guidelines>


<separator>
{SEPARATOR}
</separator>

<before_sentence>
{BEFORE_SENTENCE}
</before_sentence>
"""
