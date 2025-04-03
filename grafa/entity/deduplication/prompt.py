"""Prompts for entity deduplication."""

DEDUPLICATION_PROMPT_TEMPLATE = """
Your task is to determine if a new concept entity should be considered unique or merged with an existing similar concept.

<guidelines>
    1. Entity Uniqueness Rules:
        - Output -1 if the new concept does not match any existing concept
        - Output the index number of the matching existing concept if it should be merged
        - You may only output a single index number
        - Consider both name similarity and description meaning

    2. Matching Criteria:
        - Compare semantic meaning of all fields, not just text similarity
        - Consider variations in terminology that describe the same underlying concept
        - Look for complementary or overlapping descriptions

    3. Decision Process:
        - Empty similar concepts list always results in -1 (unique)
        - Direct matches in both name and description indicate same concept
        - Partial matches require careful evaluation of description context

    4. Output Format:
        - Return ONLY the index number, no other text or explanations
        - No explanation or reasoning should be included
        - Use -1 for unique concepts
        - Use matching concept's index (0-based) for merges
        - The index number you output must exist in the list of similar concepts
        - You may only output a single index number
        - You may not output multiple numbers or non-integers
</guidelines>

The new concept and list of similar existing concepts are provided below:

<new_concept>
{entity}
</new_concept>

<similar_entities>
{similar_entities}
</similar_entities>

Important reminders:
 - Output must be only the index number
 - Carefully evaluate description semantics
 - Consider both exact and semantic matches
 - Default to -1 for empty similar concept list
 - No explanations or additional text in output
"""

MERGE_ENTITY_PROMPT_TEMPLATE = """
Your task is to merge two concept entities into a single entity by incorporating new information while preserving existing data.

<guidelines>
    1. Merging Rules:
        - Modifications must be ADDITIVE only
        - Never remove existing information from old_entity
        - Preserve all existing tags, attributes, and descriptions
        - Only add new, non-redundant information from entity
        - If entity contains no new information, return old_entity unchanged
        - You should not change the name of the old entity, rather if the new entity has a different name, you should add the new name as a synonym

    2. Merging Process:
        - Use old_entity as the base
        - Carefully analyze entity for new, unique information
        - If no new information exists, keep old_entity exactly as is
        - Only modify if entity contains truly new, non-redundant information
        - Integrate new information while maintaining existing context
        - Ensure merged result contains union of both entities' information

    3. Output Requirements:
        - Return valid JSON following format_instructions
        - Return only the JSON object, no other text or explanations
        - Preserve all fields from old_entity
        - Add new fields from entity if not present
        - Combine descriptions if they provide different perspectives
        - The content of the output should be in {language}
</guidelines>

<entity>
{entity}
</entity>

<old_entity>
{old_entity}
</old_entity>

{format_instructions}
"""
