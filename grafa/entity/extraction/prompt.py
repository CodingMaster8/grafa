"""Prompts for the entity and relationship extraction."""

ENTITY_EXTRACTOR_PROMPT_TEMPLATE= """
Your task is to recognize all the unique entities in a provided text excerpt.

<guidelines>
    0. Only information that is explicitly stated in the excerpt should be included in the output.
    1. Do not consider entities that appear only in the origin document context but not in the excerpt.
    2. Any entity that is in the excerpt and appears as an entity type in the schema, must be included in the output.
    3. You must only identify entities of the types defined in the schema.
    4. You must not add any entity type that is not defined in the schema.
    5. The entity as well as its attributes, must be taken from the excerpt. You may not extrapolate or make up any information that is not explicitly stated in the excerpt.
    6. If the excerpt language does not match the desired output language:
        - Translate attribute values while preserving entity names that are proper nouns (e.g., "Siemens AG" remains unchanged)
        - For culture-specific terms without direct translations, keep the original term and add an explanation in parentheses
        - Examples:
            - Proper nouns to preserve: People names, company names, product names, location names
            - Culture-specific terms: "Hygge" (Danish concept of coziness), "Fika" (Swedish coffee break tradition)
    7. For the synonyms field:
       - Include only exact matches from the excerpt or common abbreviations (e.g., "WHO" for "World Health Organization")
       - Exclude hypernyms/hyponyms (e.g., "vehicle" is not a synonym for "car")
       - If conflicting synonyms exist, prioritize the most frequently mentioned variant
    8. Any synonyms identified must be clearly supported by the excerpt.
    9. Synonyms should only be added if they occur in the source text (or in common usage) and are not the primary entity name.
    10. For entities matching multiple schema types:
    - You must choose one and only one type for the entity
    - Choose the most specific type defined in the schema
    - If two types are equally specific, prioritize the type that captures more of the entity's key attributes mentioned in the text
    - Example: If "Apple Inc." could be both "Organization" and "Company", choose "Company" as it's more specific
    11. Handle implicit references by:
        - Extracting only if accompanied by explicit identifiers (e.g., "the CEO (John Doe)" → Person entity, assuming such an entity type is defined in the schema)
        - Ignoring standalone role references without names ("the CEO said...")
    12. Handle entity coreference (multiple mentions of the same entity):
    - Consolidate information from all mentions into a single entity entry
    - For conflicting attribute information, prioritize the most specific/detailed mention
    - Include pronouns' information only when they clearly refer to a single entity
    13. Entity context considerations:
    - Extract entities even when mentioned in hypothetical contexts.
    - Do not extract entities that are explicitly negated (e.g., "not affiliated with Amazon")
    - For entities in quotations or reported speech, extract them normally but don't include speculative attributes
</guidelines>

Both the excerpt and information about the origin document are provided below:

{chunk_representation}

This excerpt may contain entities as those mentioned in the schema below:

{format_instructions}

You need to output only a JSON file following the format of the schema.

The desired output language is {language}.

Important reminders:
 - If an entity is mentioned in the excerpt, and fits any of the entity types in the schema, it must be included in the output.
 - Double check that the entity is extracted only once.
 - Double check that the entity type actually matches the entity.
 - The translation should occur only for the attributes while preserving entity names if the names are language-independent or require a specific case.
 - The output must strictly conform to the JSON format defined by the schema, with no additional keys or extraneous information.

Final verification steps:
1. Verify every entity matching schema types is extracted
2. Confirm each entity appears exactly once
3. Ensure all attributes are explicitly mentioned in the text
4. Validate that entity type assignments match schema definitions
5. Check that JSON format strictly follows the provided schema
6. Return only the JSON, nothing else.
"""


RELATIONSHIP_EXTRACTOR_PROMPT_TEMPLATE = """
Your task is to identify all valid relationships between entities in a provided text excerpt, strictly following the defined relationship schema.

<guidelines>
    0. Only relationships explicitly stated or clearly implied in the excerpt should be included.
    1. Use only relationship types defined in the schema - never invent new types.
    2. Relationships must follow the directionality defined in the schema (source → target).
    3. Each relationship must be unique - avoid duplicate entries for the same entity pair + type.
    4. Source and target entities must exist in the provided entity list.
    5. Do not consider relationships that appear only in the origin document context but not in the excerpt.
    6. Handle relationship coreference by:
        - Combining multiple mentions of the same relationship into a single entry
        - Prioritizing the most specific/detailed mention when conflicts exist
    7. For ambiguous relationships:
        - Only include if supported by strong contextual evidence
        - Exclude if multiple interpretations are equally plausible
    8. Handle negated relationships by:
        - Including explicit negations only if required by schema (e.g., "NOT_AFFILIATED_WITH")
        - Excluding implicit negations unless explicitly stated
    9. Relationship properties considerations:
        - Only include properties defined in the schema
        - Property values must be directly stated in the text
    10. For temporal relationships:
        - Use exact time references when available
        - Avoid inferring temporal sequences without explicit indicators
    11. Handle quantitative relationships by:
        - Including numerical values exactly as stated
        - Avoiding approximations unless explicitly qualified (e.g., "approximately")
    12. Relationship context rules:
        - Extract relationships mentioned in hypothetical contexts
        - Exclude relationships in negated contexts (e.g., "they considered but rejected the merger")
    13. Relationship strength indicators:
        - Do not infer relationship strength unless explicitly qualified
        - Use schema-defined relationship types without modification
</guidelines>

The excerpt and information about the origin document are provided below:

{chunk_representation}

The entities you will be finding relationships between are provided below:

<entities>
{entities_str}
</entities>

The relationship types you will be finding are provided below:

<allowed_relationships>
{allowed_relationships_str}
</allowed_relationships>

Remember, only these kinds of relationships are allowed. Not any other relationships.

You must output only a JSON file following this format:
{format_instructions}

Important reminders:
 - Double-check relationship direction matches schema definition
 - Verify both source and target entities exist in the entity list
 - Ensure relationship types exactly match schema definitions
 - Confirm no duplicate relationships exist
 - The output must strictly conform to the JSON schema with no additional keys

Final verification steps:
1. Validate all relationships match schema types and directions
2. Confirm source/target entities exist in provided list
3. Check for duplicate relationships
4. Ensure no schema violations exist
5. Verify JSON structure matches required format exactly
6. Return only the JSON, nothing else.
"""
