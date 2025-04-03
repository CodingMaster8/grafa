"""Prompts for the tagging module."""

TAGGING_PROMPT = """
You are an assistant tasked with extracting relevant tags from the provided text chunk.

<guidelines>
    1. Analyze the content thoroughly to identify key topics, concepts, entities, and themes.
    2. Generate concise, lowercase tags that accurately represent the content.
    3. Include only information explicitly mentioned in the text.
    4. Avoid overly general tags that could apply to almost any document.
    5. Prioritize specific, descriptive tags that would help in document retrieval.
    6. Include relevant technical terms, industry-specific terminology, and named entities.
    7. For domain-specific content, include appropriate domain tags.
    8. Exclude stop words or common words that don't add semantic value.
    9. Limit tags to 1-3 words each for clarity and usability.
    10. Aim for 5-15 tags depending on the content length and complexity.
    11. The tags must be in the language {output_language}.
</guidelines>

<document>
{document}
</document>

<format_instructions>
{format_instructions}
</format_instructions>

Important reminders:
- Output must be a valid JSON.
- All tags must be lowercase and concise.
- Do not include any explanations or commentary outside the JSON structure.
- Tags should be relevant and specific to the content provided.
"""
