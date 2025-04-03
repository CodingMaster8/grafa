"""Prompts for the agentic chunker."""
AGENTIC_CHUNKING="""
You are an assistant tasked with splitting a document into chunks based on logical breaks in the text.
Your job is to analyze the provided lines and suggest the best line number to cut. The goal is to create chunks that are semantically coherent and do not split ideas, sentences, or paragraphs unnaturally.
You will be provided with a text fragment of a document and the current summary of processed text of the same document that has
been processed before.

Do not explain your reasons, just output the JSON response.

<guidelines>
 Here's a comprehensive list of rules:

    1. ** Identify the best line number where the text can be split, ensuring semantic coherence.
    2. ** Take into consideration the current_summary for defining where to split.
    3. ** The updated summary must be an update of the current_summary.
    4. ** The updated summary must be in the language {output_language}.
    5. ** To correctly update the summary follow the next stepts:
         -> Analyze what changed in the document by asking the following question "How does this part of the document update your understanding given the past summary"
         -> Only update the summary if you find something relevant that is new or rejects the current summary.
         -> The updated summary must be a combination of new found topics and past topics
         -> Try making the summary less than 100 tokens.

</guidelines>

### Example:
If the text is:
<l-1> Introduction to machine learning...
<l-2> It is a field that studies algorithms...
<l-3> Applications include robotics, AI...
<l-4> Now let's discuss deep learning.

A good split might be:
line_number: 3, -> reason: Line 3 completes the discussion about general machine learning, making it a natural break before introducing deep learning.

### Input:
Current Summary (context of the text so far):
{current_summary}

Document Lines to Analyze:
{input_text}

<format_instructions>
{format_instructions}
</format_instructions>

The output must be a valid JSON. Nothing else.
"""
