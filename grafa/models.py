"""Models for the Grafa database."""
import base64
from abc import ABC
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional, Type

from grafa.embedding_template import (
    GRAFA_CHUNK_EMBEDDING_TEMPLATE,
    get_default_embedding_template,
    populate_template,
)

import aioboto3
from botocore.exceptions import ClientError
from langchain_core.embeddings import Embeddings
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, ConfigDict, Field, create_model
from uuid_extensions import uuid7

if TYPE_CHECKING:
    from grafa.client import GrafaClient

# Initialize as empty dictionary that will be populated via decorator
_BUILT_IN_NODE_SUBTYPES = {}

__all__ = [
    "GrafaBaseNode",
    "GrafaDocument",
    "GrafaChunk",
    "GrafaDocumentHistory",
    "Relationship",
    "GrafaDatabase",
    "_BUILT_IN_NODE_SUBTYPES",
]


def register_node_type(cls):
    """
    Register a class in the _BUILT_IN_NODE_SUBTYPES dictionary.

    Parameters
    ----------
    cls : Type
        The class to register

    Returns
    -------
    Type
        The same class, unmodified
    """
    _BUILT_IN_NODE_SUBTYPES[cls.__name__] = cls
    return cls


class Entity(BaseModel):
    """Defines a node in the knowledge graph."""

    name: str = Field(description="Name of the entity")


class Relationship(BaseModel):
    """Defines a relationship between two node types."""

    from_type: str = Field(description="Source node type")
    to_type: str = Field(description="Target node type")
    type: str = Field(description="Type of relationship")
    description: str = Field(description="Description of the relationship")
    user_defined: bool = Field(
        default=False, description="Whether the relationship is user defined"
    )

    def __repr__(self):
        """Return a string representation suitable for LLM prompts."""
        return f"Relationship(from_type='{self.from_type}', to_type='{self.to_type}', type='{self.type}', description='{self.description}')"

    @property
    def relationship_type(self) -> str:
        """Formatted relationship type for Neo4j."""
        return self.type.upper().replace(" ", "_")


_BUILT_IN_ALLOWED_RELATIONSHIPS = [
    Relationship(
        from_type="GrafaDatabase",
        to_type="GrafaDocument",
        type="HAS_DOCUMENT",
        description="A database contains a document",
    ),
    Relationship(
        from_type="GrafaDocument",
        to_type="GrafaChunk",
        type="HAS_CHUNK",
        description="A document contains a chunk",
    ),
    Relationship(
        from_type="GrafaDocument",
        to_type="GrafaDocumentHistory",
        type="HAS_HISTORY",
        description="A document has a history of past versions",
    ),
    Relationship(
        from_type="GrafaDocumentHistory",
        to_type="GrafaDocumentHistory",
        type="NEXT_VERSION",
        description="A document history succeeded by another document history",
    ),
    Relationship(
        from_type="GrafaChunk",
        to_type="GrafaChunk",
        type="NEXT_CHUNK",
        description="A chunk is followed by another chunk",
    ),
]  # REFERENCES is allowed towards nodes that the user wants to link to a GrafaChunk


class GrafaBaseNode(BaseModel, ABC):
    """Base class for all nodes."""

    uuid: str = Field(
        default_factory=lambda: str(uuid7()),
        description="Unique identifier for the node, auto-generated if not provided",
    )
    name: str = Field(description="Name of the entity in the target language")
    synonyms: list[str] = Field(
        default_factory=list,
        description="List of alternative names/synonyms for the entity. These must be in the target language, and must be relevant to the document of origin.",
    )
    version: int = Field(default=1, description="Version number of the node")
    create_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the node was created",
    )
    update_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the node was last updated",
    )
    grafa_database_name: str = Field(
        description="Name of the database this node belongs to"
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Embedding of the node"
    )
    text_representation: Optional[str] = Field(
        default=None,
        description="Text representation of the node used for text indexing",
    )
    model_config = ConfigDict(
        metadata_attributes={
            "uuid",
            "version",
            "create_date",
            "update_date",
            "model_config",
            "grafa_database_name",
            "grafa_original_type_name",
            "embedding",
            "text_representation",
        },
        reserved_fields={
            "uuid",
            "version",
            "create_date",
            "update_date",
            "model_config",
            "grafa_database_name",
            "grafa_original_type_name",
            "embedding",
            "text_representation",
        },
        user_defined=False,
    )

    # Class level dictionary to store relationship definitions per database and class
    # Key is (database_name, class_name)
    # Value is a dictionary with key (from_type, to_type, rel_type) mapping to Relationship
    _relationship_registry: ClassVar[
        Dict[tuple[str, str], Dict[tuple[str, str, str], Relationship]]
    ] = {}

    # Neo4j connection to use (set per instance)
    _neo4j_driver: AsyncGraphDatabase | None = None
    # Embedding function to use (set per instance)
    _embedding_function: Embeddings | None = None
    # Embedding dimension to use (set per instance)
    _embedding_dimension: int | None = None
    # Semantic similarity function to use (set per instance)
    _semantic_similarity_function: str = "cosine"

    def dump_non_metadata(self) -> dict:
        """Dump the non-metadata fields of the node."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k
            in [
                f
                for f in self.model_fields.keys()
                if f not in self.model_config.get("metadata_attributes", set())
            ]
        }

    @classmethod
    def get_neo4j_label(cls) -> str:
        """Get the Neo4j label for the node class.

        Returns
        -------
        str
            The Neo4j label for the node class
        """
        return cls.model_config.get("grafa_original_type_name", cls.__name__)

    @classmethod
    def get_pydantic_template(cls) -> Type:
        """Get the Pydantic template for the node class.

        Returns
        -------
        Type
            The Pydantic template model class for the node
        """
        # Create a new Pydantic model with only non-metadata fields
        fields = {}
        for field_name, field in cls.model_fields.items():
            if field_name not in cls.model_config["metadata_attributes"]:
                fields[field_name] = (field.annotation, field)

        # Add entity_type field to identify the type of entity
        entity_type = (
            Literal[cls.get_neo4j_label()],
            Field(description="The type of entity this represents"),
        )
        fields["grafa_original_type_name"] = entity_type
        fields["model_config"] = ConfigDict(
            grafa_original_type_name=cls.get_neo4j_label()
        )

        # Create the model with the same name but without metadata fields
        template_model = create_model(
            cls.get_neo4j_label(), __doc__=cls.__doc__, __base__=Entity, **fields
        )

        # Add a method to convert template instance to original class
        def to_original_class(template_instance, grafa_client: "GrafaClient"):
            """Convert template instance to original class instance.

            Parameters
            ----------
            template_instance : object
                Instance of the template class
            grafa_client : GrafaClient
                The Grafa client instance

            Returns
            -------
            GrafaBaseNode
                Instance of the original node class
            """
            # Get data from template instance
            data = template_instance.model_dump()
            # Create an instance of the original class with the data
            return grafa_client._create_unpersisted_node(cls, **data)

        # Attach the method to the template model
        template_model.to_original_class = to_original_class

        return template_model

    def __str__(self):
        """Create a string representation of the node."""
        non_metadata = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.model_config["metadata_attributes"]
        }
        return f"{self.__class__.__name__} ({non_metadata})"

    @classmethod
    def _get_embedding_template(cls) -> str:
        """Get the embedding template for the node.

        Parameters
        ----------
        cls : Type[GrafaBaseNode]
            The node class to get the embedding template for

        Returns
        -------
        str
            The embedding template for the node
        """
        if (
            "embedding_template" in cls.model_config
            and cls.model_config["embedding_template"]
        ):
            return cls.model_config["embedding_template"]
        return get_default_embedding_template(cls)

    def get_embedding_text(self, extra_fields: dict = None) -> str:
        """Get the text for the node."""
        template = self._get_embedding_template()
        return populate_template(template, self, extra_fields)

    def to_cypher_props(self) -> dict:
        """Convert the node to a dictionary of Cypher properties.

        Returns
        -------
        dict
            A dictionary of properties for Cypher queries
        """
        # Convert the node to a dictionary
        data = self.model_dump()

        # Handle special cases
        if "create_date" in data and isinstance(data["create_date"], datetime):
            data["create_date"] = data["create_date"].isoformat()
        if "update_date" in data and isinstance(data["update_date"], datetime):
            data["update_date"] = data["update_date"].isoformat()

        # Remove private fields (starting with _)
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        return data

    async def save_to_neo4j(self, driver=None, allow_merge=False):
        """Save the node to Neo4j.

        Parameters
        ----------
        driver : AsyncGraphDatabase, optional
            The Neo4j driver to use. If not provided, uses the node's driver.
        allow_merge : bool, optional
            If True, allows merging with existing nodes, potentially replacing the UUID.
            Default is False, which preserves the original UUID.

        Raises
        ------
        RuntimeError
            If no Neo4j driver is provided or set
        """
        # Use provided driver or instance driver
        driver = driver or self._neo4j_driver
        if not driver:
            raise RuntimeError("No Neo4j driver provided or set")

        # Update the update_date
        self.update_date = datetime.now(timezone.utc)

        text_search = self.model_config.get("text_search", False)
        semantic_search = self.model_config.get("semantic_search", False)
        if text_search or semantic_search:
            # Get new text representation
            new_text_representation = self.get_embedding_text()

            # Update text representation if it has changed
            if new_text_representation != self.text_representation:
                self.text_representation = new_text_representation

                # Only update embedding if semantic search is enabled and either:
                # 1. No existing embedding, or
                # 2. Text representation has changed
                if semantic_search:
                    if self._embedding_function is None:
                        raise ValueError("Embedding function not set")
                    if self._embedding_dimension is None:
                        raise ValueError("Embedding dimension not set")
                    self.embedding = self._embedding_function(new_text_representation)

        # Get the node label - use class method
        label = self.get_neo4j_label()

        # Add USER_DEFINED label for non-built-in nodes
        is_built_in = self.__class__.__name__ in _BUILT_IN_NODE_SUBTYPES
        labels = f":{label}" if is_built_in else f":USER_DEFINED:{label}"

        # Convert the node to properties dictionary
        props = self.to_cypher_props()

        async with driver.session() as session:
            if allow_merge:
                # Use MERGE to create or update the node, potentially replacing UUID
                query = f"""
                MERGE (n{labels} {{name: $name, grafa_database_name: $database_name}})
                SET n = $props
                RETURN n
                """
                result = await session.run(
                    query,
                    name=self.name,
                    database_name=self.grafa_database_name,
                    props=props,
                )
            else:
                # Check if the node already exists by UUID
                check_query = f"""
                MATCH (n{labels} {{uuid: $uuid}})
                RETURN n
                """
                result = await session.run(check_query, uuid=self.uuid)
                if await result.single():
                    query = f"""
                    MATCH (n{labels} {{uuid: $uuid}})
                    SET n = $props
                    RETURN n
                    """
                else:
                    # Node doesn't exist, create it
                    query = f"""
                    CREATE (n{labels} $props)
                    RETURN n
                    """
                result = await session.run(query, uuid=self.uuid, props=props)
            # Return True if node was created/updated
            return bool(await result.single())

    @classmethod
    async def get_by_uuid(
        cls, uuid: str, driver: AsyncGraphDatabase, database_name: str = ""
    ) -> Optional["GrafaBaseNode"]:
        """Get a node by its UUID.

        Parameters
        ----------
        uuid : str
            The UUID of the node to get
        driver : AsyncGraphDatabase
            The Neo4j driver to use
        database_name : str, optional
            The database name to use for the node label

        Returns
        -------
        Optional[GrafaBaseNode]
            The node if found, None otherwise
        """
        if not driver:
            raise RuntimeError("No Neo4j driver provided")

        # Get the label using the class method - no database prefix
        label = cls.get_neo4j_label()

        # Add database filter if provided
        where_clause = ""
        if database_name:
            where_clause = "WHERE n.grafa_database_name = $database_name"

        query = f"""
        MATCH (n:{label} {{uuid: $uuid}})
        {where_clause}
        RETURN n
        """

        async with driver.session() as session:
            result = await session.run(query, uuid=uuid, database_name=database_name)
            record = await result.single()

            if not record:
                return None

            # Get the node data
            node_data = dict(record["n"])

            # Convert ISO date strings back to datetime objects
            if "create_date" in node_data and isinstance(node_data["create_date"], str):
                node_data["create_date"] = datetime.fromisoformat(
                    node_data["create_date"]
                )
            if "update_date" in node_data and isinstance(node_data["update_date"], str):
                node_data["update_date"] = datetime.fromisoformat(
                    node_data["update_date"]
                )

            # Create a new instance
            instance = cls(**node_data)
            instance._neo4j_driver = driver

            return instance

    async def create_relationship(
        self,
        target: "GrafaBaseNode",
        relationship_type: str,
        properties: dict = None,
        merge_if_exists: bool = True,
    ) -> bool:
        """Create a relationship between this node and a target node.

        Parameters
        ----------
        target : GrafaBaseNode
            The target node to create a relationship with
        relationship_type : str
            The type of relationship to create
        properties : dict, optional
            Properties to set on the relationship
        merge_if_exists : bool, optional
            If True, will merge properties with existing relationship if one exists, by default True

        Returns
        -------
        bool
            True if the relationship was created successfully, False otherwise

        Raises
        ------
        RuntimeError
            If no Neo4j driver is set
        ValueError
            If the relationship type is not allowed
        RuntimeError
            If either source or target node does not exist in the database
        """
        if not self._neo4j_driver:
            raise RuntimeError("No Neo4j driver set")

        # Format the relationship type
        rel_type = relationship_type.upper().replace(" ", "_")

        # Validate that this relationship is allowed
        db_name = self.grafa_database_name

        # Use the original type name for validation
        from_type = self.get_neo4j_label()
        to_type = target.get_neo4j_label()

        # Create the relationship key for lookup
        registry_key = (db_name, from_type)
        relationship_key = (from_type, to_type, rel_type)

        if registry_key not in self._relationship_registry:
            raise ValueError(
                f"Relationship {rel_type} from {from_type} to {to_type} is not allowed - registry key not found"
            )

        if relationship_key not in self._relationship_registry.get(registry_key, {}):
            raise ValueError(
                f"Relationship {rel_type} from {from_type} to {to_type} is not allowed - relationship key not found"
            )

        # Find the relationship definition to get the description
        rel_description = None
        if (
            registry_key in self._relationship_registry
            and relationship_key in self._relationship_registry[registry_key]
        ):
            relationship = self._relationship_registry[registry_key][relationship_key]
            rel_description = relationship.description

        # Get labels for both nodes
        source_label = self.get_neo4j_label()
        target_label = target.get_neo4j_label()

        # Create or update properties with description
        if properties is None:
            properties = {}

        # Add description to properties if found
        if rel_description and "description" not in properties:
            properties["description"] = rel_description

        # Add validation to ensure both nodes are in the same database
        if self.grafa_database_name != target.grafa_database_name:
            raise ValueError(
                f"Cannot create relationship between nodes in different databases: {self.grafa_database_name} and {target.grafa_database_name}"
            )

        # First check if both nodes exist
        check_query = f"""
        MATCH (a:{source_label} {{uuid: $source_uuid}})
        RETURN count(a) as source_exists
        """

        async with self._neo4j_driver.session() as session:
            result = await session.run(check_query, source_uuid=self.uuid)
            record = await result.single()
            if record and record["source_exists"] == 0:
                raise RuntimeError(
                    f"Source node with UUID {self.uuid} does not exist in the database"
                )

        check_query = f"""
        MATCH (b:{target_label} {{uuid: $target_uuid}})
        RETURN count(b) as target_exists
        """

        async with self._neo4j_driver.session() as session:
            result = await session.run(check_query, target_uuid=target.uuid)
            record = await result.single()
            if record and record["target_exists"] == 0:
                raise RuntimeError(
                    f"Target node with UUID {target.uuid} does not exist in the database"
                )

        # Create the relationship with optional property merging
        if merge_if_exists:
            query = f"""
            MATCH (a:{source_label} {{uuid: $source_uuid}}), (b:{target_label} {{uuid: $target_uuid}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r = $props
            ON MATCH SET r += $props
            RETURN r
            """
        else:
            query = f"""
            MATCH (a:{source_label} {{uuid: $source_uuid}}), (b:{target_label} {{uuid: $target_uuid}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r = $props
            RETURN r
            """

        try:
            async with self._neo4j_driver.session() as session:
                result = await session.run(
                    query,
                    source_uuid=self.uuid,
                    target_uuid=target.uuid,
                    props=properties,
                )
            return True
        except Exception:  # noqa: BLE001
            return False

    async def get_related_nodes(
        self,
        relationship_type: str = None,
        target_class: Type["GrafaBaseNode"] = None,
        direction: Literal["OUTGOING", "INCOMING", "BOTH"] = "OUTGOING",
    ) -> List["GrafaBaseNode"]:
        """Get nodes related to this node.

        Parameters
        ----------
        relationship_type : str, optional
            Filter by relationship type
        target_class : Type[GrafaBaseNode], optional
            Filter by target node class
        direction : str, optional
            Direction of the relationship: "OUTGOING", "INCOMING", or "BOTH"

        Returns
        -------
        List[GrafaBaseNode]
            List of related nodes
        """
        if not self._neo4j_driver:
            raise RuntimeError("No Neo4j driver set")

        # Ensure the node is saved
        if not self.uuid:
            await self.save_to_neo4j()

        # Normalize the relationship type if provided
        rel_type = None
        if relationship_type:
            rel_type = relationship_type.upper().replace(" ", "_")

        # Get the target label if a class is provided - no database prefix
        target_label = None
        if target_class:
            target_label = target_class.get_neo4j_label()

        # Build the relationship pattern based on direction
        if direction == "OUTGOING":
            rel_pattern = "-[r{}]->".format(f":{rel_type}" if rel_type else "")
        elif direction == "INCOMING":
            rel_pattern = "<-[r{}]-".format(f":{rel_type}" if rel_type else "")
        else:  # BOTH
            rel_pattern = "-[r{}]-".format(f":{rel_type}" if rel_type else "")

        # Build the target pattern
        target_pattern = f"(b:{target_label})" if target_label else "(b)"

        # Add database filter to ensure we only get nodes from the same database
        database_name = self.grafa_database_name
        where_clause = f"WHERE b.grafa_database_name = '{database_name}'"

        # Query for related nodes
        query = f"""
        MATCH (a:{self.get_neo4j_label()} {{uuid: $uuid}}){rel_pattern}{target_pattern}
        {where_clause}
        RETURN b
        """

        result_nodes = []

        async with self._neo4j_driver.session() as session:
            result = await session.run(query, uuid=self.uuid)

            async for record in result:
                # Get the node data
                node_data = dict(record["b"])

                # Get the node label(s)
                node_labels = record["b"].labels

                # Find the appropriate class for this node
                node_class_name = None
                db_name = self.grafa_database_name

                for label in node_labels:
                    if db_name and label.startswith(f"{db_name}_"):
                        # Strip database prefix
                        node_class_name = label[len(f"{db_name}_") :]
                        break
                    else:
                        node_class_name = label
                        break

                if not node_class_name:
                    # Skip if we can't determine the class
                    continue

                # Get the class from globals
                node_class = globals().get(node_class_name)

                if not node_class:
                    # Skip if class not found
                    continue

                # Convert ISO date strings back to datetime objects
                if "create_date" in node_data and isinstance(
                    node_data["create_date"], str
                ):
                    node_data["create_date"] = datetime.fromisoformat(
                        node_data["create_date"]
                    )
                if "update_date" in node_data and isinstance(
                    node_data["update_date"], str
                ):
                    node_data["update_date"] = datetime.fromisoformat(
                        node_data["update_date"]
                    )

                # Create a new instance
                instance = node_class(**node_data)
                instance._neo4j_driver = self._neo4j_driver

                result_nodes.append(instance)

        return result_nodes

    @classmethod
    async def get_all(
        cls,
        driver: AsyncGraphDatabase = None,
        database_name: str = "",
        limit: int = 100,
        offset: int = 0,
        filters: dict = None,
    ) -> List["GrafaBaseNode"]:
        """Get all nodes of this type.

        Parameters
        ----------
        driver : AsyncGraphDatabase, optional
            The Neo4j driver to use. If not provided, uses the class driver
        database_name : str, optional
            The database name to use for the node label. If not provided,
            uses the database_name from the model_config if available
        limit : int, optional
            Maximum number of nodes to return
        offset : int, optional
            Number of nodes to skip
        filters : dict, optional
            Properties to filter nodes by

        Returns
        -------
        List[GrafaBaseNode]
            List of nodes
        """
        # Use provided driver or class driver
        driver = driver or cls._neo4j_driver
        if not driver:
            raise RuntimeError("No Neo4j driver provided or set")

        # Get the node label - no database prefix
        label = cls.get_neo4j_label()

        # Build filter clause if filters provided
        where_clauses = []
        if database_name:
            where_clauses.append(f"n.grafa_database_name = '{database_name}'")

        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    where_clauses.append(f"n.{key} = '{value}'")
                else:
                    where_clauses.append(f"n.{key} = {value}")

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Query for nodes
        query = f"""
        MATCH (n:{label})
        {where_clause}
        RETURN n
        ORDER BY n.name
        SKIP {offset}
        LIMIT {limit}
        """

        result_nodes = []

        async with driver.session() as session:
            result = await session.run(query)

            async for record in result:
                # Get the node data
                node_data = dict(record["n"])

                # Convert ISO date strings back to datetime objects
                if "create_date" in node_data and isinstance(
                    node_data["create_date"], str
                ):
                    node_data["create_date"] = datetime.fromisoformat(
                        node_data["create_date"]
                    )
                if "update_date" in node_data and isinstance(
                    node_data["update_date"], str
                ):
                    node_data["update_date"] = datetime.fromisoformat(
                        node_data["update_date"]
                    )

                # Create a new instance
                instance = cls(**node_data)
                instance._neo4j_driver = driver

                result_nodes.append(instance)

        return result_nodes

    @classmethod
    async def get_by_name(
        cls, name: str, driver: AsyncGraphDatabase, database_name: str = ""
    ) -> list["GrafaBaseNode"]:
        """Get all nodes with a given name.

        Parameters
        ----------
        name : str
            The name of the nodes to get
        driver : AsyncGraphDatabase
            The Neo4j driver to use
        database_name : str, optional
            The database name to use for the node label

        Returns
        -------
        list[GrafaBaseNode]
            List of nodes with the given name
        """
        if not driver:
            raise RuntimeError("No Neo4j driver provided")

        # Get the label using the class method - no database prefix
        label = cls.get_neo4j_label()

        # Add database filter if provided
        where_clause = ""
        if database_name:
            where_clause = "WHERE n.grafa_database_name = $database_name"

        query = f"""
        MATCH (n:{label} {{name: $name}})
        {where_clause}
        RETURN n
        """

        result_nodes = []
        async with driver.session() as session:
            result = await session.run(query, name=name, database_name=database_name)

            async for record in result:
                # Get the node data
                node_data = dict(record["n"])

                # Convert ISO date strings back to datetime objects
                if "create_date" in node_data and isinstance(
                    node_data["create_date"], str
                ):
                    node_data["create_date"] = datetime.fromisoformat(
                        node_data["create_date"]
                    )
                if "update_date" in node_data and isinstance(
                    node_data["update_date"], str
                ):
                    node_data["update_date"] = datetime.fromisoformat(
                        node_data["update_date"]
                    )

                # Create a new instance
                instance = cls(**node_data)
                instance._neo4j_driver = driver

                result_nodes.append(instance)

        return result_nodes

    async def delete_from_neo4j(self, driver=None) -> bool:
        """Delete the node and all its relationships from Neo4j.

        Parameters
        ----------
        driver : AsyncGraphDatabase, optional
            The Neo4j driver to use. If not provided, uses the node's driver.

        Returns
        -------
        bool
            True if the node was successfully deleted, False otherwise

        Raises
        ------
        RuntimeError
            If no Neo4j driver is provided or set
        """
        # Use provided driver or instance driver
        driver = driver or self._neo4j_driver
        if not driver:
            raise RuntimeError("No Neo4j driver provided or set")

        # If no UUID, nothing to delete
        if not self.uuid:
            return False

        # Get the node label
        label = self.get_neo4j_label()

        # Delete the node and all its relationships
        query = f"""
        MATCH (n:{label} {{uuid: $uuid}})
        DETACH DELETE n
        """

        try:
            async with driver.session() as session:
                result = await session.run(query, uuid=self.uuid)
                # Return True if at least one node was affected
                summary = await result.consume()
                return summary.counters.nodes_deleted > 0
        except Exception as e:  # noqa: BLE001
            print(f"Failed to delete node: {str(e)}")
            return False


@register_node_type
class GrafaDocument(GrafaBaseNode):
    """
    Node the represents a file, a collection of chunks.

    It has versioning to support multiple versions of the same document.
    A document does not contain its content itself, but is rather joined to a collection of chunks.
    """

    path_raw: str = Field(description="Path to the raw document in the S3 bucket")
    path_processed: str | None = Field(
        description="Path to the processed document in the S3 bucket", default=None
    )
    author: str | None = Field(description="Author of the document", default=None)
    source: str | None = Field(
        description="Source/origin of the document", default=None
    )
    context: str | None = Field(description="Context of the document", default=None)
    summary: str | None = Field(default=None, description="Summary of the document")
    extension: str = Field(description="Extension of the document")
    hash_raw: str = Field(description="Hash of the raw document content")
    hash_processed: str | None = Field(
        description="Hash of the processed document content", default=None
    )
    s3_version_raw: str | None = Field(
        default=None, description="Current S3 version ID for raw document"
    )
    s3_version_processed: str | None = Field(
        default=None, description="Current S3 version ID for processed document"
    )
    model_config = ConfigDict(
        link_to_chunk=False,
        semantic_search=False,
    )

    async def get_processed_content(self) -> str:
        """Get the processed content of the document.

        This fetches the actual document content from S3 using the stored path.

        Returns
        -------
        str
            The processed content of the document

        Raises
        ------
        ValueError
            If the processed document path is not set
        RuntimeError
            If there's an error retrieving the content from S3
        """
        if not self.path_processed:
            raise ValueError("Processed document path not available")

        try:
            # Parse the S3 path to get bucket and key
            # Assume path format is s3://bucket-name/path/to/file.ext
            s3_parts = self.path_processed.replace("s3://", "").split("/", 1)
            bucket = s3_parts[0]
            key = s3_parts[1] if len(s3_parts) > 1 else ""
            session = aioboto3.Session()
            async with session.client("s3") as s3_client:
                params = {"Bucket": bucket, "Key": key}
                if self.s3_version_processed:
                    params["VersionId"] = self.s3_version_processed
                response = await s3_client.get_object(**params)
                body = await response["Body"].read()
                return body.decode("utf-8")
        except ClientError as e:
            raise RuntimeError(f"Failed to retrieve document from S3: {str(e)}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error retrieving document content: {str(e)}")

    async def get_chunks(self) -> List["GrafaChunk"]:
        """Get all chunks associated with this document.

        Retrieves all chunks that are linked to this document through
        the HAS_CHUNK relationship.

        Returns
        -------
        List[GrafaChunk]
            List of chunks associated with this document

        Raises
        ------
        RuntimeError
            If no Neo4j driver is set
        """
        if not self._neo4j_driver:
            raise RuntimeError("No Neo4j driver set")
        return await self.get_related_nodes(
            relationship_type="HAS_CHUNK", target_class=GrafaChunk, direction="OUTGOING"
        )


@register_node_type
class GrafaDocumentHistory(GrafaBaseNode):
    """
    Node that represents a historical version of a document.

    When a document is updated, its previous state is stored in this node
    to maintain version history. The document_uuid links it to its current
    version in GrafaDocument.
    """

    document_uuid: str = Field(description="UUID of the corresponding GrafaDocument")
    path_raw: str = Field(
        description="Path to the raw document in the S3 bucket when this version was current"
    )
    path_processed: str | None = Field(
        description="Path to the processed document in the S3 bucket when this version was current",
        default=None,
    )
    author: str | None = Field(
        description="Author of the document for this version", default=None
    )
    source: str | None = Field(
        description="Source/origin of the document for this version", default=None
    )
    context: str | None = Field(
        description="Context of the document for this version", default=None
    )
    summary: str | None = Field(
        default=None, description="Summary of the document for this version"
    )
    extension: str = Field(description="Extension of the document for this version")
    hash_raw: str = Field(
        description="Hash of the raw document content for this version"
    )
    hash_processed: str | None = Field(
        description="Hash of the processed document content for this version",
        default=None,
    )
    s3_version_raw: str | None = Field(
        default=None, description="S3 version ID for raw document in this version"
    )
    s3_version_processed: str | None = Field(
        default=None, description="S3 version ID for processed document in this version"
    )
    document_version: int = Field(
        description="Version number of the document when this history record was created"
    )
    change_reason: str | None = Field(
        default=None,
        description="Reason or description of what changed in this version",
    )
    model_config = ConfigDict(
        link_to_chunk=False,
        semantic_search=False,
    )


@register_node_type
class GrafaChunk(GrafaBaseNode):
    """
    Node that represents a chunk of a document.

    It has a content field that is the actual text of the chunk.
    """

    content: str = Field(description="Text content of the chunk")
    chunk_sequence_id: int = Field(
        description="Sequential identifier of the chunk within its document"
    )
    chunk_length: int = Field(description="Length of the chunk in characters")
    original_document_name: str = Field(description="Name of the original document")
    original_document_context: str = Field(
        description="Context of the original document"
    )
    original_document_summary: str = Field(
        description="Summary of the original document"
    )
    tags: List[str] = Field(description="Tags of the chunk", default_factory=list)
    model_config = ConfigDict(
        link_to_chunk=False,
        semantic_search=True,
        text_search=True,
        embedding_template=GRAFA_CHUNK_EMBEDDING_TEMPLATE,
    )

    async def link_node(self, node: "GrafaBaseNode", properties: dict = None) -> bool:
        """Link a node to this chunk.

        Parameters
        ----------
        node : GrafaBaseNode
            The node to link to this chunk
        properties : dict, optional
            Properties to set on the relationship

        Returns
        -------
        bool
            True if the relationship was created successfully, False otherwise
        """
        return await self.create_relationship(
            node, "REFERENCES", properties, merge_if_exists=True
        )


@register_node_type
class GrafaDatabase(GrafaBaseNode):
    """Node that represents metadata for a Grafa database.

    This node stores configuration information for a database, including
    the YAML representation of all node types and relationships.
    """

    description: str = Field(description="Description of the database")
    yaml: str = Field(description="YAML representation of the database configuration")
    language: str = Field(description="Language of the database", default="Spanish")
    node_types: Dict[str, Type[GrafaBaseNode]] = Field(
        default_factory=dict,
        description="Dictionary of node types in the database",
        exclude=True,  # Not stored in Neo4j directly
    )
    allowed_relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of permissible relationships in the database",
        exclude=True,  # Not stored in Neo4j directly
    )
    original_name_to_class_name: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original type names to prefixed class names",
        exclude=True,  # Not stored in Neo4j directly
    )
    model_config = ConfigDict(
        link_to_chunk=False,
        semantic_search=False,
    )

    def __init__(self, **data):
        """Initialize the database metadata and register relationships.

        Parameters
        ----------
        **data : dict
            Keyword arguments to initialize the model
        """
        super().__init__(**data)
        # Register relationships if they are provided
        if hasattr(self, "allowed_relationships") and self.allowed_relationships:
            self._register_relationships()

    @classmethod
    async def database_exists(cls, db_name: str, driver: AsyncGraphDatabase) -> bool:
        """Check if a database with the given name exists in Neo4j.

        Parameters
        ----------
        db_name : str
            The name of the database to check for
        driver : AsyncGraphDatabase
            The Neo4j driver to use for the check

        Returns
        -------
        bool
            True if the database exists, False otherwise
        """
        metadata = await cls.get_by_name(db_name, driver)
        return len(metadata) > 0

    @classmethod
    async def load_from_neo4j(
        cls, db_name: str, driver: AsyncGraphDatabase
    ) -> "GrafaDatabase":
        """Load database configuration from Neo4j.

        Parameters
        ----------
        db_name : str
            The name of the database to load
        driver : AsyncGraphDatabase
            The Neo4j driver to use

        Returns
        -------
        GrafaDatabase
            The loaded database configuration

        Raises
        ------
        RuntimeError
            If the database does not exist in Neo4j
        """
        grafa_database = await cls.get_by_name(db_name, driver)
        if not grafa_database:
            raise RuntimeError(f"Database '{db_name}' does not exist in Neo4j")

        # Load the full configuration from the metadata
        grafa_database[0]._unpack_yaml()

        return grafa_database[0]

    @classmethod
    def from_yaml(
        cls, yaml_str: str, db_name: str | None = None, base64_encode: bool = False
    ) -> "GrafaDatabase":
        """Create a GrafaDatabase from a YAML string.

        Parameters
        ----------
        yaml_str : str
            YAML string containing the database configuration
        db_name : str | None, optional
            Optional database name to override the one in YAML
        base64_encode : bool, optional
            Whether the YAML string is base64 encoded

        Returns
        -------
        GrafaDatabase
            Initialized database metadata

        Raises
        ------
        ValueError
            If the encoded YAML string is invalid
        """
        from grafa.dynamic_models import load_definitions

        try:
            if base64_encode:
                decoded_yaml = base64.b64decode(yaml_str.encode()).decode()
            else:
                decoded_yaml = yaml_str

            # Create metadata instance
            metadata = load_definitions(yaml_str=decoded_yaml, db_name=db_name)

            # Process YAML for storage
            if base64_encode:
                # Already encoded, store as is
                metadata.yaml = yaml_str
            else:
                # Need to encode for storage
                metadata.yaml = base64.b64encode(yaml_str.encode()).decode()

            return metadata
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to create GrafaDatabase from YAML: {str(e)}")

    def _unpack_yaml(self) -> "GrafaDatabase":
        """Get the database configuration from this metadata node.

        Raises
        ------
        ValueError
            If the YAML configuration is invalid
        """
        # The YAML is stored base64 encoded, so we need to decode it
        decoded_yaml = base64.b64decode(self.yaml.encode()).decode()
        unpacked_grafa_database = self.from_yaml(
            yaml_str=decoded_yaml, db_name=self.name
        )
        # Set the private attributes of this class to the one we unpacked
        self.node_types = unpacked_grafa_database.node_types
        self.allowed_relationships = unpacked_grafa_database.allowed_relationships
        self.original_name_to_class_name = (
            unpacked_grafa_database.original_name_to_class_name
        )

    def _register_relationships(self) -> None:
        """
        Register relationships with the appropriate node types.

        This maps each relationship to its corresponding node types in the relationship registry.
        Each node type will only see relationships that are relevant to it.
        """
        # Clear any existing relationships for these node types in this database
        for node_type_name in self.node_types.keys():
            # Validate that this relationship is allowed
            db_name = self.name
            # Use the original type name for validation
            from_type = self.original_name_to_class_name.get(
                node_type_name, node_type_name
            )
            registry_key = (db_name, from_type)
            self._relationship_registry[registry_key] = {}

        # Register each relationship with both its "from" and "to" node types
        for relationship in self.allowed_relationships:
            from_type = relationship.from_type
            to_type = relationship.to_type
            rel_type = relationship.relationship_type

            # Create registry keys with database name
            from_key = (self.name, to_type)
            to_key = (self.name, from_type)

            # Add the relationship to both node types (list)
            if from_key not in self._relationship_registry:
                self._relationship_registry[from_key] = {}
            if to_key not in self._relationship_registry:
                self._relationship_registry[to_key] = {}

            # Add to lookup dictionary for O(1) access
            relationship_key = (from_type, to_type, rel_type)
            self._relationship_registry[from_key][relationship_key] = relationship

            # Only add it to the "to" type if it's different from the "from" type
            if to_type != from_type:
                self._relationship_registry[to_key][relationship_key] = relationship

    def get_index_name(
        self,
        node_type: str | None,
        index_type: Literal["name_constraint", "constraint", "text", "vector"],
    ) -> str:
        """
        Return the index or constraint name for the given node type and index type.

        Parameters
        ----------
        node_type : str | None
            The type of node for which to get the index name. Must be a key in node_types or None for global indices.
        index_type : Literal["name_constraint", "constraint", "text", "vector"]
            The type of index to create.

        Returns
        -------
        str
            The formatted index or constraint name.

        Raises
        ------
        ValueError
            If the node type is not in the database configuration or if index_type is invalid.
        """
        if node_type is not None and node_type not in self.node_types:
            raise ValueError(
                f"Node type {node_type} not found in the database configuration."
            )

        # For global indices (when node_type is None)
        if node_type is None:
            if index_type == "text":
                name = "global_text_idx"
            elif index_type == "vector":
                name = "global_vector_idx"
            else:
                raise ValueError(
                    "Global indices can only be of type 'text' or 'vector'."
                )
        else:
            node_class = self.node_types[node_type]
            node_label = node_class.get_neo4j_label()
            if index_type in ("name_constraint", "constraint"):
                name = f"{node_label}_name_constraint"
            elif index_type == "text":
                name = f"{node_label}_text_idx"
            elif index_type == "vector":
                name = f"{node_label}_vector_idx"
            else:
                raise ValueError(
                    "Invalid index type. Must be one of 'name_constraint', 'text', or 'vector'."
                )

        return name.replace(" ", "_").replace("-", "_")

    async def create_indices(
        self, driver: AsyncGraphDatabase = None
    ) -> Dict[str, List[str]]:
        """Create text and vector indices for all node types in the database.

        Creates fulltext indices for nodes with text_search=True (on text_representation field)
        and vector indices for nodes with semantic_search=True (on embedding field).
        Also creates constraints for unique name per node type and database name.
        Additionally creates global text and vector indices for all non-built-in nodes.

        Parameters
        ----------
        driver : GraphDatabase, optional
            The Neo4j driver to use. If not provided, uses the node's driver.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary of created indices by node type

        Raises
        ------
        RuntimeError
            If no Neo4j driver is provided or set
        """
        # Use provided driver or instance driver
        driver = driver or self._neo4j_driver
        if not driver:
            raise RuntimeError("No Neo4j driver provided or set")

        created_indices = {}

        # Make sure node types are loaded
        if not self.node_types:
            self._unpack_yaml()

        async with driver.session() as session:
            # Create global indices for non-built-in nodes
            created_indices["global"] = []

            # Create global text index
            global_text_index_name = self.get_index_name(None, "text")
            create_query = f"""
                CREATE FULLTEXT INDEX `{global_text_index_name}` IF NOT EXISTS
                FOR (n:USER_DEFINED)
                ON EACH [n.text_representation]
                OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: '{self.language.lower()}' }} }}
            """
            await session.run(create_query)
            created_indices["global"] = [f"Global text index: {global_text_index_name}"]

            # Create global vector index
            global_vector_index_name = self.get_index_name(None, "vector")
            create_query = f"""
                CREATE VECTOR INDEX `{global_vector_index_name}` IF NOT EXISTS
                FOR (n:USER_DEFINED)
                ON n.embedding
                OPTIONS {{indexConfig: {{ `vector.dimensions`: {self._embedding_dimension},
                                   `vector.similarity_function`: '{self._semantic_similarity_function}' }} }}
            """
            await session.run(create_query)
            created_indices["global"].append(
                f"Global vector index: {global_vector_index_name}"
            )

        # For each node type in the database
        for node_type_name, node_class in self.node_types.items():
            created_indices[node_type_name] = []

            # Get the Neo4j node label (without database prefix)
            node_label = node_class.get_neo4j_label()

            async with driver.session() as session:
                # Create constraint for unique name per node type and database
                constraint_name = self.get_index_name(node_type_name, "name_constraint")
                if node_class.model_config.get("unique_name", True):
                    constraint_query = f"""
                            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                            FOR (n:{node_label})
                            REQUIRE (n.name, n.grafa_database_name) IS UNIQUE
                        """
                    await session.run(constraint_query)
                    created_indices[node_type_name].append(
                        f"Constraint: {constraint_name}"
                    )
                if node_class.model_config.get("text_search", False):
                    # Create fulltext index for text_search if enabled
                    fulltext_index_name = self.get_index_name(node_type_name, "text")
                    create_query = f"""
                        CREATE FULLTEXT INDEX `{fulltext_index_name}` IF NOT EXISTS
                        FOR (n:{node_label})
                        ON EACH [n.text_representation]
                        OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: '{self.language.lower()}' }} }}
                    """
                    await session.run(create_query)
                    created_indices[node_type_name].append(
                        f"Text index: {fulltext_index_name}"
                    )

                # Create vector index if semantic_search is enabled
                if node_class.model_config.get("semantic_search", False):
                    vector_index_name = self.get_index_name(node_type_name, "vector")
                    create_query = f"""
                        CREATE VECTOR INDEX `{vector_index_name}` IF NOT EXISTS
                        FOR (n:{node_label})
                        ON n.embedding
                        OPTIONS {{indexConfig: {{ `vector.dimensions`: {self._embedding_dimension},
                                                   `vector.similarity_function`: '{self._semantic_similarity_function}' }} }}
                    """
                    await session.run(create_query)
                    created_indices[node_type_name].append(
                        f"Vector index: {vector_index_name}"
                    )

        return created_indices


class LoadFile(BaseModel):
    """A file to load into Grafa's S3 bucket.

    Attributes
    ----------
    name: Unique name of the file
    path: Path to the file
    content: Content of the file
    context: Context of the file
    author: Author of the file
    source: Source of the file

    Fields populated during ingestion
    --------------------------------
    _s3_bucket: S3 bucket of the file
    _database_name: Database name of the file

    Fields populated during raw file upload
    ------------------------------------
    _raw_object_key: S3 object key of the file
    _raw_version_id: Version ID of the file
    _raw_etag: ETag of the file

    Fields populated during processed file upload
    ------------------------------------------
    _processed_object_key: S3 object key of the processed file
    _processed_version_id: Version ID of the processed file
    _processed_etag: ETag of the processed file
    """

    name: str
    path: str | Path | None
    content: str | None
    context: str | None
    author: str | None
    source: str | None
    # Fields populated during ingestion
    _s3_bucket: str | None = None
    _database_name: str | None = None
    # Fields populated during raw file upload
    _raw_object_key: str | None = None
    _raw_version_id: str | None = None
    _raw_etag: str | None = None
    # Fields populated during processed file upload
    _processed_object_key: str | None = None
    _processed_version_id: str | None = None
    _processed_etag: str | None = None
    # Pydantic object
    _grafa_document: GrafaDocument | None = None

    model_config = ConfigDict(underscore_attrs_are_private=True)

    async def load_content(self) -> str:
        """Load content from S3 if it's not already loaded.

        If content is None and S3 information is available, retrieves the
        file content from S3 and sets the content field.

        Returns
        -------
        str
            The content of the file

        Raises
        ------
        ValueError
            If the content is not available and cannot be loaded from S3
        RuntimeError
            If there's an error retrieving the content from S3
        """
        # If content is already loaded, just return it
        if self.content is not None:
            return self.content

        # Check if we have the necessary S3 information
        if not self._processed_object_key or not self._s3_bucket:
            raise ValueError(
                "Cannot load content: processed object key or S3 bucket not available"
            )
        try:
            session = aioboto3.Session()
            async with session.client("s3") as s3_client:
                params = {"Bucket": self._s3_bucket, "Key": self._processed_object_key}
                if self._processed_version_id:
                    params["VersionId"] = self._processed_version_id
                response = await s3_client.get_object(**params)
                body = await response["Body"].read()
                self.content = body.decode("utf-8")
                return self.content
        except ClientError as e:
            raise RuntimeError(f"Failed to retrieve file from S3: {str(e)}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error retrieving file content: {str(e)}")
