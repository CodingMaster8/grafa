"""Template Factory Models for the entity extraction processes."""
import json
from typing import Any, List, Literal, Type, Union

from pydantic import BaseModel, Field, create_model

from grafa.models import Entity


class FoundRelationship(BaseModel):
    """A relationship that was found in the text."""

    type: str = Field(..., description="The type of the relationship")
    from_entity_name: str = Field(..., description="The name of the entity that the relationship starts from")
    from_entity_type: str = Field(..., description="The type of the entity that the relationship starts from")
    from_entity_index: int = Field(..., description="The index of the entity that the relationship starts from")
    to_entity_name: str = Field(..., description="The name of the entity that the relationship ends at")
    to_entity_type: str = Field(..., description="The type of the entity that the relationship ends at")
    to_entity_index: int = Field(..., description="The index of the entity that the relationship ends at")

    @classmethod
    def get_pydantic_template(cls, allowed_relationship_types: List[str]) -> str:
        """Get the Pydantic template for the relationship.

        Parameters
        ----------
        allowed_relationship_types : List[str]
            List of allowed relationship types to use in the Literal type annotation

        Returns
        -------
        str
            The Pydantic template for the relationship
        """
        template_model = create_model(
            cls.__name__,
            __doc__=cls.__doc__,
            type=(Literal[tuple(allowed_relationship_types)], Field(..., description="The type of the relationship")),
            from_entity_index=(int, Field(..., description="The index of the entity that the relationship starts from")),
            to_entity_index=(int, Field(..., description="The index of the entity that the relationship ends at")),
        )
        return template_model

class RelationshipOutput(BaseModel):
    """Output of the relationship extraction."""

    relationships: List[FoundRelationship] = Field(..., description="List of relationships")

    @classmethod
    def get_pydantic_template(cls, allowed_relationship_types: List[str]) -> str:
        """Get the Pydantic template for the relationship output.

        Parameters
        ----------
        allowed_relationship_types : List[str]
            List of allowed relationship types to use in the Literal type annotation

        Returns
        -------
        str
            The Pydantic template for the relationship output
        """
        relationship_template = FoundRelationship.get_pydantic_template(allowed_relationship_types)
        template_model = create_model(
            cls.__name__,
            __doc__=cls.__doc__,
            __base__=cls,
            relationships=(List[relationship_template], Field(..., description="List of relationships")),
        )

        return template_model

class EntityOutput(BaseModel):
    """Output of the entity extraction.

    Attributes
    ----------
    entities : List[Entity]
        List of extracted entities.
    """

    entities: List[Entity] = Field(..., description="List of entities")

    @classmethod
    def get_pydantic_template(cls, entity_models: List[Type[Entity]]) -> Type[BaseModel]:
        """
        Get the Pydantic template for a certain combination of entity models, including custom JSON deserialization.

        Parameters
        ----------
        entity_models : List[Type[Entity]]
            List of entity model classes to be used for deserializing individual entity JSON objects.
            Each model must expose its original type via the 'grafa_original_type_name' attribute
            on its model configuration.

        Returns
        -------
        Type[BaseModel]
            A dynamically created Pydantic model class with a custom from_json method for proper deserialization.
        """
        # Create a union type from the provided entity models.
        allowed_entities = Union[tuple(entity_models)]
        template_model = create_model(
            cls.__name__,
            __doc__=cls.__doc__,
            __base__=cls,
            entities=(List[allowed_entities], Field(..., description="List of entities")),
        )

        # Save the list of allowed entity models within the template model for later use.
        template_model._entity_models = entity_models

        @classmethod
        def from_json(cls_, data: Union[str, dict]) -> Any:
            """
            Deserialize JSON for EntityOutput.

            This method deserializes the JSON input for the entity output, converting each
            object in the 'entities' list into an instance of the appropriate entity model
            as determined by its 'grafa_original_type_name' field.

            Parameters
            ----------
            data : Union[str, dict]
                The JSON string or dictionary representing the entity output.

            Returns
            -------
            EntityOutput
                An instance of the dynamically created EntityOutput model with entities properly deserialized.

            Raises
            ------
            ValueError
                If an entity's type (grafa_original_type_name) is not recognized.
            """
            # If data is a JSON string, parse it into a dictionary.
            if isinstance(data, str):
                data = json.loads(data)

            processed_entities = []
            # Iterate over each entity in the input JSON.
            for entity_data in data.get("entities", []):
                # Extract the entity type name from the JSON.
                type_name = entity_data.get("grafa_original_type_name")
                matched_model = None
                # Search for the matching model using the grafa_original_type_name
                for model in cls_._entity_models:
                    # Retrieve the designated type name from the model's configuration or default to its class name.
                    model_type_name = getattr(model.model_config, "grafa_original_type_name", None) or model.__name__
                    if model_type_name == type_name:
                        matched_model = model
                        break
                if matched_model is None:
                    raise ValueError(f"Unknown entity type: {type_name}")
                # Deserialize the entity data into an object of the matched model.
                # Use model_validate instead of parse_obj for Pydantic v2 compatibility
                processed_entities.append(matched_model.model_validate(entity_data))
            # Replace the raw entities JSON with the deserialized entity objects.
            data["entities"] = processed_entities
            return cls_.model_validate(data)

        # Attach the custom from_json method to the generated template model.
        template_model.from_json = from_json
        return template_model
