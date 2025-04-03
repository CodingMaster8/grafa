"""Prompts for the entity search."""

CONCEPT_EXTRACTOR_PROMPT_TEMPLATE = """
Your task is to recognize most relevant concepts inside a user query.
You must output a JSON object with the following schema:

{format_instructions}

<guidelines>
1. Concepts can be more than one word joined by a stopword.
    Example : 'Return of Investment' is a concept of more than one word.
2. Ignore stopwords.
3. Prioritize large words
4. Ignore dates
5. Do not invent concepts, they must be explicit concepts that appear in the query.
</guidelines>

<query>
{query}
</query>
"""
