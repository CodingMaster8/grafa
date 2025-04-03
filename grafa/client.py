"""Client for the Grafa Knowledge System."""
import asyncio
import os
from pathlib import Path
from typing import Literal, Self, Type

from grafa.document.chunking import agentic_chunking
from grafa.document.load import process_file, upload_file
from grafa.document.tagging import tag_chunk
from grafa.entity.deduplication import deduplicate_entity
from grafa.entity.extraction import (
    RelationshipOutput,
    extract_entities,
    extract_relationships,
)
from grafa.entity.search import extract_concepts
from grafa.models import (
    _BUILT_IN_NODE_SUBTYPES,
    GrafaBaseNode,
    GrafaChunk,
    GrafaDatabase,
    GrafaDocument,
    GrafaDocumentHistory,
    LoadFile,
    Relationship,
)
from grafa.settings import grafa_password, grafa_uri, grafa_username
from grafa.utils.s3 import create_bucket
from grafa.utils.string import clean_string

import fsspec
import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langfuse.decorators import observe
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, ConfigDict, Field

# Stage 1. Load and transcription
# If there are indexes, go to next stages. If not, treat the documents as S3 files only.
# A database may contain indexed or unindexed documents.
# Unindexed documents may be indexed. Indexes may be deleted.
# Updating an existing document reindexes it.
# There is a getAllDocuments method that obtains all the documents in the database, directly from their transcription.
# Stage 2. Chunk - Construct Nodes, Augment
# Stage 3. Index
# Stage 4. Entity/Relationship Extraction
# Stage 5. Deduplication and Merging/Updating existing Entities/Relationships
# Stage 6. Index new entities and relationships

# At another time
# Stage 7. Retrieval
# There is direct retrieval, lexical retrieval, semantic retrieval, and hybrid.


class GrafaConfig(BaseModel):
    """Configuration for the GrafaClient.

    Parameters
    ----------
    db_name : str
        The name of the database
    embedding_model : Embeddings
        The embedding model to use.
    embedding_dimension : int
        The embedding dimension to use.
    semantic_similarity_function : str
        The similarity function to use for the embedding model.
    llm : Runnable
        The LLM to use for text search.
    grafa_database : GrafaDatabase | None
        The configuration of the database. If None, will attempt to load from Neo4j
    neo4j_driver : AsyncGraphDatabase | None
        The Neo4j database connection. If None, will create new connection
    s3_bucket : str | None
        The S3 bucket to use for storing documents. If None, will use GRAFA_S3_BUCKET env var
    """

    neo4j_driver: AsyncGraphDatabase | None = Field(
        default=None, description="The Neo4j database"
    )
    s3_bucket: str | None = Field(
        default=None, description="The S3 bucket to use for storing documents"
    )
    embedding_model: Embeddings = Field(description="The embedding model to use")
    embedding_dimension: int = Field(description="The embedding dimension to use")
    semantic_similarity_function: str = Field(
        default="cosine",
        description="The similarity function to use for the embedding model",
    )
    llm: Runnable = Field(description="The LLM to use for text search")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, _trusted_instantiation=None, **data):
        """
        Initialize a GrafaConfig instance.

        Parameters
        ----------
        _trusted_instantiation : bool
            Whether the instantiation is trusted
        **data : dict
            Configuration parameters

        Raises
        ------
        RuntimeError
            If direct instantiation is attempted without using create()
        """
        if not _trusted_instantiation:
            raise RuntimeError(
                "Direct instantiation of GrafaConfig is not allowed. Use GrafaConfig.create() instead."
            )
        super().__init__(**data)

    @classmethod
    async def create(cls, **kwargs) -> "GrafaConfig":
        """Create a GrafaConfig instance.

        Returns
        -------
        GrafaConfig
            The fully initialized configuration.
        """
        instance = cls(_trusted_instantiation=True, **kwargs)
        await instance._async_post_init()  # explicitly run async initialization
        return instance

    async def _async_post_init(self) -> None:
        """Post initialization hook for the Pydantic model.

        Automatically connects to the Neo4j database after model initialization.

        Parameters
        ----------
        __context : Any
            Context passed by Pydantic during initialization
        """
        self._verify_s3_bucket()
        await create_bucket(self.s3_bucket)
        await self._connect_to_database()

    @classmethod
    async def default(cls) -> Self:
        """Get the default configuration."""
        from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

        embedding_dimension = 1024
        embedding_model = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
        )
        llm = ChatBedrockConverse(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            temperature=0,
            max_tokens=4096,
        ).with_retry()

        return await cls.create(
            neo4j_driver=None,  # Populated after the initialization
            s3_bucket=None,  # Populated after the initialization
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            semantic_similarity_function="cosine",
            llm=llm,
        )

    def _verify_s3_bucket(self) -> None:
        """Verify that the S3 bucket is set.

        Raises
        ------
        ValueError
            If the S3 bucket is not set
        """
        if self.s3_bucket is None:
            self.s3_bucket = os.getenv("GRAFA_S3_BUCKET")
            if not self.s3_bucket:
                raise ValueError(
                    "Neither s3_bucket nor GRAFA_S3_BUCKET environment variable were provided"
                )

    async def _connect_to_database(self):
        """Connect to the Neo4j database if not already connected.

        Connects to Neo4j using environment variables for URI, username and password
        if not already connected. Verifies connectivity before returning.

        Returns
        -------
        AsyncGraphDatabase
            The connected Neo4j database instance

        Raises
        ------
        RuntimeError
            If any required environment variables (grafa_uri, grafa_username,
            grafa_password) are not set
        """
        if self.neo4j_driver is None:
            if not grafa_uri:
                raise RuntimeError("grafa_uri is not set")
            if not grafa_username:
                raise RuntimeError("grafa_username is not set")
            if not grafa_password:
                raise RuntimeError("grafa_password is not set")
            self.neo4j_driver = AsyncGraphDatabase.driver(
                uri=grafa_uri, auth=(grafa_username, grafa_password)
            )
        try:
            await self.neo4j_driver.verify_connectivity()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not connect to Neo4j database: {str(e)}")


class GrafaClient(BaseModel):
    """Client for the Grafa Knowledge System.

    The GrafaClient can be initialized in two ways:

    1. From a YAML configuration file, creating a new database:
    ```python
    client = await GrafaClient.from_yaml(
        yaml_path="path/to/config.yaml",
        db_name="my_db",  # Optional, defaults to name in YAML
    )
    ```

    2. Directly with an existing database, loading the configuration from Neo4j:
    ```python
    client = await GrafaClient.create(
        db_name="my_db",
        grafa_database=None,  # Will load config from Neo4j
    )
    ```

    Parameters
    ----------
    db_name : str
        The name of the database
    grafa_database : GrafaDatabase | None
        The configuration of the database. If None, will attempt to load from Neo4j
    grafa_config : GrafaConfig | None
        The configuration used for the Grafa Client. If None, will use default configuration
    """

    db_name: str = Field(description="The name of the database")
    grafa_database: GrafaDatabase | None = Field(
        default=None, description="The configuration of the database"
    )
    grafa_config: GrafaConfig | None = Field(
        default=None, description="Configuration used for the Grafa Client."
    )

    def __init__(self, _trusted_instantiation=None, **data):
        """
        Initialize a GrafaClient instance.

        Parameters
        ----------
        _trusted_instantiation : bool
            Whether the instantiation is trusted
        **data : dict
            Configuration parameters

        Raises
        ------
        RuntimeError
            If direct instantiation is attempted without using create() or from_yaml()
        """
        if not _trusted_instantiation:
            raise RuntimeError(
                "Direct instantiation of GrafaClient is not allowed. Use GrafaClient.create() or GrafaClient.from_yaml() instead."
            )
        super().__init__(**data)

    # Add properties to access grafa_config attributes directly
    @property
    def neo4j_driver(self) -> AsyncGraphDatabase:
        """Access the Neo4j database from the config."""
        return self.grafa_config.neo4j_driver

    @property
    def s3_bucket(self) -> str:
        """Access the S3 bucket from the config."""
        return self.grafa_config.s3_bucket

    @property
    def embedding_model(self) -> Embeddings:
        """Access the embedding model from the config."""
        return self.grafa_config.embedding_model

    @property
    def embedding_dimension(self) -> int:
        """Access the embedding dimension from the config."""
        return self.grafa_config.embedding_dimension

    @property
    def semantic_similarity_function(self) -> str:
        """Access the embedding similarity function from the config."""
        return self.grafa_config.semantic_similarity_function

    @property
    def llm(self) -> Runnable:
        """Access the LLM from the config."""
        return self.grafa_config.llm

    @classmethod
    async def create(cls, **kwargs) -> "GrafaClient":
        """
        Asynchronously create and initialize a GrafaClient instance.

        Returns
        -------
        GrafaClient
            The fully initialized client.
        """
        instance = cls(_trusted_instantiation=True, **kwargs)
        await instance._async_post_init()  # explicitly run async initialization
        return instance

    @classmethod
    async def from_yaml(
        cls,
        yaml_path: str | Path,
        db_name: str | None = None,
        grafa_config: GrafaConfig | None = None,
    ) -> Self:
        """Create a GrafaClient instance from a YAML configuration file.

        Parameters
        ----------
        yaml_path : str | Path
            Path to the YAML file containing the database configuration
        db_name : str | None, optional
            Name to use for the database. If not provided, will use name from YAML file
        grafa_config : GrafaConfig | None, optional
            Configuration to use for the Grafa Client. If not provided, will use default configuration

        Returns
        -------
        GrafaClient
            Initialized GrafaClient instance connected to the database

        Raises
        ------
        RuntimeError
            If a database with the given name already exists
        ValueError
            If the YAML configuration is invalid
        """
        # Convert string path to Path object if needed
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        # Read YAML file content
        with fsspec.open(yaml_path, "r") as f:
            yaml_str = f.read()
        grafa_database = GrafaDatabase.from_yaml(yaml_str, db_name=db_name)
        me = await cls.create(
            db_name=grafa_database.name,
            grafa_database=grafa_database,
            grafa_config=grafa_config,
        )
        me.grafa_database._neo4j_driver = me.neo4j_driver
        me.grafa_database._embedding_function = me.embedding_model.embed_query
        me.grafa_database._embedding_dimension = me.embedding_dimension
        me.grafa_database._semantic_similarity_function = (
            me.semantic_similarity_function
        )
        await me.grafa_database.create_indices()

        return me

    async def _async_post_init(self) -> None:
        """Post initialization hook for the Pydantic model.

        Automatically connects to the Neo4j database after model initialization.

        Parameters
        ----------
        __context : Any
            Context passed by Pydantic during initialization
        """
        if self.grafa_config is None:
            self.grafa_config = await GrafaConfig.default()

        # Check if database exists and handle accordingly
        db_exists = await GrafaDatabase.database_exists(self.db_name, self.neo4j_driver)
        if self.grafa_database is None:
            if not db_exists:
                raise RuntimeError(f"Database {self.db_name} does not exist in Neo4j")
            # Load config from Neo4j
            self.grafa_database = await GrafaDatabase.load_from_neo4j(
                self.db_name, self.neo4j_driver
            )
        else:
            if db_exists:
                raise RuntimeError(f"Database {self.db_name} already exists in Neo4j")
            # Save new config to Neo4j
            self.grafa_database._neo4j_driver = self.neo4j_driver
            await self.grafa_database.save_to_neo4j()

        # Set the database name and Neo4j driver for all node types
        if self.grafa_database:
            for node_type in self.grafa_database.node_types.values():
                node_type.database_name = self.db_name
                # Driver will be set per instance when nodes are created

    def claim_node(self, node: GrafaBaseNode) -> GrafaBaseNode:
        """Claim a node for the current database.

        Parameters
        ----------
        node : GrafaBaseNode
            The node to claim

        Returns
        -------
        GrafaBaseNode
            The claimed node
        """
        node._neo4j_driver = self.neo4j_driver
        node._embedding_function = self.embedding_model.embed_query
        node._embedding_dimension = self.embedding_dimension
        node._semantic_similarity_function = self.semantic_similarity_function
        return node

    def _create_unpersisted_node(
        self, node_type: str | type[GrafaBaseNode], **data
    ) -> GrafaBaseNode:
        """Create a new node of the given type.

        Parameters
        ----------
        node_type : str | type[GrafaBaseNode]
            The type of node to create, can be either a string name or the actual node class
        **data
            Data to initialize the node with

        Returns
        -------
        GrafaBaseNode
            The created node
        """
        node_class = (
            node_type if isinstance(node_type, type) else self.get_node_class(node_type)
        )
        data["grafa_database_name"] = self.db_name
        node: GrafaBaseNode = node_class(**data)
        self.claim_node(node)
        return node

    async def create_node(
        self, node_type: str | type[GrafaBaseNode], **data
    ) -> GrafaBaseNode:
        """Create a new node of the given type.

        Parameters
        ----------
        node_type : str | type[GrafaBaseNode]
            The type of node to create, can be either a string name or the actual node class
        **data
            Data to initialize the node with

        Returns
        -------
        GrafaBaseNode
            The created node
        """
        node = self._create_unpersisted_node(node_type, **data)
        await node.save_to_neo4j()
        return node

    async def upload_file(
        self,
        document_name: str,
        document_path: str | Path | None = None,
        document_text: str | None = None,
        context: str | None = None,
        author: str | None = None,
        source: str | None = None,
    ) -> LoadFile:
        """Upload a document to the Grafa database.

        If a document with the same name already exists, it will be updated.
        Otherwise, a new document will be created.

        Parameters
        ----------
        document_name : str
            Name of the document to upload
        document_path : str | Path | None, optional
            Path to the document to upload
        document_text : str | None, optional
            Raw text content to upload
        context : str | None, optional
            Context of the document to upload
        author : str | None, optional
            Author of the document
        source : str | None, optional
            Source of the document

        Returns
        -------
        LoadFile
            The uploaded file object with metadata
        """
        if (document_path is None and document_text is None) or (
            document_path is not None and document_text is not None
        ):
            raise ValueError(
                "Must provide exactly one of document_path or document_text"
            )

        # Create and upload the file
        file = LoadFile(
            name=document_name,
            path=document_path,
            content=document_text,
            context=context,
            author=author,
            source=source,
        )

        await upload_file(file, self.s3_bucket, self.db_name)

        # Determine file extension
        extension = "txt"
        if document_path:
            path_obj = (
                Path(document_path) if isinstance(document_path, str) else document_path
            )
            extension = path_obj.suffix.lstrip(".")

        # Use S3 ETag as the content hash
        content_hash = file._raw_etag

        # Construct S3 URI path
        s3_path = f"s3://{self.s3_bucket}/{file._raw_object_key}"

        # Check for existing document
        document_results = await GrafaDocument.get_by_name(
            name=document_name, driver=self.neo4j_driver, database_name=self.db_name
        )
        document = document_results[0] if len(document_results) > 0 else None
        # Updated document data structure to match new GrafaDocument fields
        document_data = {
            "name": document_name,
            "path_raw": s3_path,
            "context": context or "",
            "author": author,
            "source": source,
            "extension": extension,
            "hash_raw": content_hash,
            "s3_version_raw": file._raw_version_id,
            "grafa_database_name": self.db_name,
        }

        document_history = None
        last_history = None

        if document:
            # Find the last history record to link histories later
            if document.version > 1:
                last_history_results = await GrafaDocumentHistory.get_by_name(
                    name=f"{document_name}_v{document.version - 1}",
                    driver=self.neo4j_driver,
                    database_name=self.db_name,
                )
                if len(last_history_results) == 0:
                    raise ValueError(
                        f"Document history {document_name}_v{document.version - 1} not found"
                    )
                last_history = last_history_results[0]

            document_history = await self.create_node(
                GrafaDocumentHistory,
                name=f"{document_name}_v{document.version}",
                document_uuid=document.uuid,
                path_raw=document.path_raw,
                path_processed=document.path_processed,
                author=document.author,
                source=document.source,
                context=document.context,
                summary=document.summary,
                extension=document.extension,
                hash_raw=document.hash_raw,
                hash_processed=document.hash_processed,
                s3_version_raw=document.s3_version_raw,
                s3_version_processed=document.s3_version_processed,
                document_version=document.version,
                grafa_database_name=self.db_name,
            )

            # Update existing document with new data
            for key, value in document_data.items():
                if value is not None:  # Only update fields that are provided
                    setattr(document, key, value)

            # Increment the version number when updating
            document.version += 1
        else:
            # Create new document
            document = await self.create_node(GrafaDocument, **document_data)
            # version defaults to 1 for new documents

        # Now create all relationships after everything is saved
        if document_history:
            # Create relationship between document and history
            await document.create_relationship(document_history, "HAS_HISTORY")

            # If there's a previous history record, link them with NEXT_VERSION relationship
            if last_history and last_history.uuid != document_history.uuid:
                await last_history.create_relationship(document_history, "NEXT_VERSION")

        # Create relationship between database and document
        await self.grafa_database.create_relationship(document, "HAS_DOCUMENT")

        file._grafa_document = document

        return file

    async def process_file(self, file: LoadFile) -> LoadFile:
        """Process a file to transcribe it.

        Parameters
        ----------
        file : LoadFile
            The file to process

        Returns
        -------
        LoadFile
            The processed file object with metadata
        """
        await process_file(file)

        if file._grafa_document is None:
            raise ValueError("LoadFile has not been loaded into a GrafaDocument")

        s3_path = f"s3://{self.s3_bucket}/{file._processed_object_key}"

        document_data = {
            "path_processed": s3_path,
            "hash_processed": file._processed_etag,
            "s3_version_processed": file._processed_version_id,
        }

        for key, value in document_data.items():
            if value is not None:
                setattr(file._grafa_document, key, value)

        await file._grafa_document.save_to_neo4j()

        return file

    async def chunk_document(
        self,
        grafa_document: GrafaDocument,
        llm: Runnable | None = None,
        max_token_chunk_size: int = 100,
        verbose: bool = False,
        output_language: str | None = None,
    ) -> list[GrafaChunk]:
        """Chunk a document into smaller chunks.

        Parameters
        ----------
        grafa_document : GrafaDocument
            The document to chunk
        llm : Runnable | None, optional
            The LLM to use for chunking. If None, will use the default LLM
        max_token_chunk_size : int, optional
            The maximum number of tokens per chunk
        verbose : bool, optional
            Whether to print verbose output
        output_language : str | None, optional
            The language of the output summary. If None, will use the language of the GrafaDatabase

        Returns
        -------
        list[GrafaChunk]
            The chunks of the document
        """
        # Make sure the document has a Neo4j driver
        if not grafa_document._neo4j_driver:
            self.claim_node(grafa_document)

        if llm is None:
            llm = self.llm

        if output_language is None:
            output_language = self.grafa_database.language

        old_chunks = await grafa_document.get_chunks()
        new_chunks = []
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        for chunk in old_chunks:
            await chunk.delete_from_neo4j()
        try:
            chunk_output = await agentic_chunking(
                await grafa_document.get_processed_content(),
                llm,
                max_token_chunk_size,
                verbose,
                output_language,
            )

            if chunk_output.summary:
                grafa_document.summary = chunk_output.summary
                await grafa_document.save_to_neo4j()

            last_chunk = None
            for i, chunk in enumerate(chunk_output.chunks):
                tags = (await tag_chunk(chunk, llm, output_language)).tags
                grafa_chunk = await self.create_node(
                    GrafaChunk,
                    name=f"{grafa_document.name}_chunk_{i+1}",
                    content=chunk,
                    tags=tags,
                    chunk_sequence_id=i + 1,
                    chunk_length=len(tokenizer.encode(chunk)),
                    grafa_database_name=grafa_document.grafa_database_name,
                    original_document_name=grafa_document.name,
                    original_document_context=grafa_document.context,
                    original_document_summary=grafa_document.summary,
                )
                await grafa_document.create_relationship(grafa_chunk, "HAS_CHUNK")
                new_chunks.append(grafa_chunk)
                if last_chunk:
                    await last_chunk.create_relationship(grafa_chunk, "NEXT_CHUNK")
                last_chunk = grafa_chunk
        except Exception as e:  # noqa: BLE001
            for chunk in new_chunks:
                await chunk.delete_from_neo4j()
            for chunk in old_chunks:
                await chunk.save_to_neo4j()
            raise RuntimeError(f"Error chunking document: {str(e)}")

        return new_chunks

    async def get_document_content(self, document_name: str) -> str:
        """Get the content of a document.

        Parameters
        ----------
        document_name : str
            The name of the document to get the content of

        Returns
        -------
        str
            The content of the document
        """
        document_results = await GrafaDocument.get_by_name(
            name=document_name,
            driver=self.neo4j_driver,
            database_name=self.db_name,
        )
        if len(document_results) == 0:
            raise ValueError(f"Document {document_name} not found")
        document = document_results[0]
        return await document.get_processed_content()

    def get_node_class(
        self, node_type: str | type[GrafaBaseNode]
    ) -> Type[GrafaBaseNode]:
        """Get the node class for a given node type.

        Parameters
        ----------
        node_type : str | type[GrafaBaseNode]
            The type of node to get the class for, or a class that inherits from GrafaBaseNode

        Returns
        -------
        Type[GrafaBaseNode]
            The node class

        Raises
        ------
        ValueError
            If the node type does not exist
        """
        if isinstance(node_type, GrafaBaseNode):
            return node_type

        if self.grafa_database is None:
            raise RuntimeError("Database config not initialized")

        # Check if node_type is a built-in type
        if node_type in _BUILT_IN_NODE_SUBTYPES:
            return self.grafa_database.node_types[node_type]

        # Check if node_type is a key in node_types dictionary
        if node_type in self.grafa_database.node_types:
            return self.grafa_database.node_types[node_type]

        # Check if node_type is a value in node_types dictionary
        for key, value in self.grafa_database.node_types.items():
            if value == node_type:
                return node_type

        # If node_type is not found anywhere
        raise ValueError(f"Node type {node_type} does not exist")

    async def get_node(
        self, node_type: str | type[GrafaBaseNode], uuid: str
    ) -> GrafaBaseNode:
        """Get a node by its UUID.

        Parameters
        ----------
        node_type : str | type[GrafaBaseNode]
            The type of node to get
        uuid : str
            The UUID of the node to get

        Returns
        -------
        GrafaBaseNode
            The retrieved node

        Raises
        ------
        ValueError
            If the node type does not exist or the node is not found
        """
        node_class = self.get_node_class(node_type)
        node = await node_class.get_by_uuid(uuid, self.neo4j_driver, self.db_name)

        if node is None:
            raise ValueError(f"Node with UUID {uuid} not found")

        self.claim_node(node)

        return node

    async def get_all_nodes(
        self,
        node_type: str | type[GrafaBaseNode],
        limit: int = 100,
        offset: int = 0,
        filters: dict = None,
    ) -> list[GrafaBaseNode]:
        """Get all nodes of a given type.

        Parameters
        ----------
        node_type : str | type[GrafaBaseNode]
            The type of nodes to get
        limit : int, optional
            Maximum number of nodes to return
        offset : int, optional
            Number of nodes to skip
        filters : dict, optional
            Properties to filter nodes by

        Returns
        -------
        list[GrafaBaseNode]
            List of nodes
        """
        node_class = self.get_node_class(node_type)
        nodes = await node_class.get_all(
            self.neo4j_driver, self.db_name, limit, offset, filters
        )

        # Set the driver for all node instances
        for node in nodes:
            self.claim_node(node)

        return nodes

    async def get_node_by_name(
        self, node_type: str | type[GrafaBaseNode], name: str
    ) -> list[GrafaBaseNode]:
        """Get a node by its name.

        Parameters
        ----------
        node_type : str | type[GrafaBaseNode]
            The type of node to get
        name : str
            The name of the node to get

        Returns
        -------
        list[GrafaBaseNode]
            The nodes with the given name
        """
        node_class = self.get_node_class(node_type)
        nodes = await node_class.get_by_name(name, self.neo4j_driver, self.db_name)
        if len(nodes) == 0:
            raise ValueError(f"Node of type {node_type} with name {name} not found")
        for node in nodes:
            self.claim_node(node)
        return nodes

    def get_user_defined_node_types(self) -> list[Type[GrafaBaseNode]]:
        """Get all user defined node types.

        Returns
        -------
        list[Type[GrafaBaseNode]]
            The user defined node types
        """
        return [
            cls
            for name, cls in self.grafa_database.node_types.items()
            if cls.model_config.get("user_defined", False)
        ]

    def get_user_defined_node_type_templates(self) -> list[BaseModel]:
        """Get all user defined node type templates.

        Returns
        -------
        list[BaseModel]
            The user defined node type templates
        """
        return [
            cls.get_pydantic_template() for cls in self.get_user_defined_node_types()
        ]

    def get_user_defined_relationship_types(self) -> list[Relationship]:
        """Get all user defined relationship types.

        Returns
        -------
        list[Type[Relationship]]
            The user defined relationship types
        """
        return [
            relationship
            for relationship in self.grafa_database.allowed_relationships
            if relationship.user_defined
        ]

    @observe
    async def similarity_search(
        self,
        query: str,
        node_types: str | GrafaBaseNode | list[str | GrafaBaseNode] | None = None,
        limit: int = 100,
        semantic_threshold: float | None = None,
        text_threshold: float | None = None,
        name_edit_distance: int | None = 5,
        search_mode: Literal["semantic", "text", "hybrid", "allowed"] = "allowed",
        include_name_matching: bool = True,
        uuid_only: bool = False,
    ) -> list[dict]:
        """
        Perform a similarity search on nodes in the database.

        Searches can use vector embeddings (semantic), text index (full-text), name matching,
        or a combination of these approaches.

        Parameters
        ----------
        query : str
            The text query to search for similar nodes.
        node_types : str | GrafaBaseNode | list[str | GrafaBaseNode] | None, optional
            The type(s) of nodes to search. Can be a single node type or a list of node types.
            If None, all user defined node types will be queried, by default None.
        limit : int, optional
            Maximum number of results to return, by default 100.
        semantic_threshold : float | None, optional
            Minimum similarity score (0.0-1.0) for semantic matches. Set to None to disable
            semantic threshold filtering, by default None.
        text_threshold : float | None, optional
            Minimum similarity score (0.0-1.0) for text index matches. Set to None to disable
            text threshold filtering, by default None.
        name_edit_distance : int | None, optional
            Maximum Levenshtein edit distance for name matching. Lower values require closer
            matches. Set to None to disable name matching, by default 5.
        search_mode : Literal["semantic", "text", "hybrid", "allowed"], optional
            The search strategy:
            - "semantic": Use only vector embeddings for search.
            - "text": Use only text-index-based matching.
            - "hybrid": Combine both semantic and text approaches
            - "allowed": Automatically use available indexes for each node type, by default "allowed".
        include_name_matching : bool, optional
            Whether to include name matching (checking node names and synonyms),
            independent of the search_mode, by default True.
        uuid_only : bool, optional
            If True, return only basic node properties (uuid, name, description, synonyms).
            If False, return all node properties under a 'node' key, by default True.

        Returns
        -------
        list[dict]
            List of matching nodes. If uuid_only=True, each containing:
            - 'name': Node name
            - 'description': Node description
            - 'synonyms': List of alternative names
            - 'semantic_score': Semantic similarity score (if semantic search was used)
            - 'text_score': Text similarity score (if text search was used)
            - 'name_match': Whether the node matched by name (if name matching was used)
            - 'index': Result position in the sorted list

            If uuid_only=False, each containing:
            - 'node': Dict containing all node properties
            - 'semantic_score': Semantic similarity score (if semantic search was used)
            - 'text_score': Text similarity score (if text search was used)
            - 'name_match': Whether the node matched by name (if name matching was used)
            - 'index': Result position in the sorted list

        Raises
        ------
        ValueError
            If no node_types are provided, if an invalid search_mode is specified,
            or if semantic search is requested for node types that don't support it.
        Exception
            If database query execution fails.
        """
        # Normalize search mode
        search_mode = search_mode.lower()

        # Determine search modes based on 'allowed' setting
        if search_mode == "allowed":
            # Will be set to True if any node supports the search type
            semantic_enabled = False
            text_enabled = False
        else:
            semantic_enabled = search_mode in ["semantic", "hybrid"]
            text_enabled = search_mode in ["text", "hybrid"]

        # Generate embedding for the query only if we might need it
        query_embedding = None

        # Convert single values to lists for consistent processing
        if node_types is not None and (
            isinstance(node_types, str) or not isinstance(node_types, list)
        ):
            node_types = [node_types]

        # Determine the index names from node_types
        vector_index_names = []
        text_index_names = []

        # If no node types are provided, use all user defined node types
        if not node_types:
            node_types_to_query = self.get_user_defined_node_types()
        else:
            node_types_to_query = [self.get_node_class(nt) for nt in node_types]

        unsupported_nodes_semantic = []
        unsupported_nodes_text = []

        for node_class in node_types_to_query:
            node_label = node_class.get_neo4j_label()
            can_semantic_search = node_class.model_config.get("semantic_search", False)
            can_text_search = node_class.model_config.get("text_search", False)
            # Handle index availability differently for 'allowed' mode
            if search_mode == "allowed":
                if can_semantic_search:
                    semantic_enabled = True
                    vector_index_names.append(
                        self.grafa_database.get_index_name(node_label, "vector")
                    )
                if can_text_search:
                    text_enabled = True
                    text_index_names.append(
                        self.grafa_database.get_index_name(node_label, "text")
                    )
            else:
                # Original logic for other modes
                if semantic_enabled:
                    if can_semantic_search:
                        vector_index_names.append(
                            self.grafa_database.get_index_name(node_label, "vector")
                        )
                    else:
                        unsupported_nodes_semantic.append(node_label)
                if text_enabled:
                    if can_text_search:
                        text_index_names.append(
                            self.grafa_database.get_index_name(node_label, "text")
                        )
                    else:
                        unsupported_nodes_text.append(node_label)

        # Generate query embedding only if semantic search will be used
        if semantic_enabled:
            query_embedding = await self.embedding_model.aembed_query(query)

        # Modify error handling for different modes
        if search_mode != "allowed" and node_types is not None:
            # Keep original error checking for non-allowed modes
            if semantic_enabled and unsupported_nodes_semantic:
                raise ValueError(
                    f"Semantic search requested but the following node types don't support it: {', '.join(unsupported_nodes_semantic)}"
                )
            if text_enabled and unsupported_nodes_text:
                raise ValueError(
                    f"Text search requested but the following node types don't support it: {', '.join(unsupported_nodes_text)}"
                )

        # Ensure we have appropriate indexes for the requested search mode
        if semantic_enabled and not vector_index_names and node_types is not None:
            raise ValueError(
                "Semantic search requested but no vector indexes available for the specified node types"
            )
        # If text search is required but no text indexes are available, raise an error
        if text_enabled and not text_index_names and node_types is not None:
            raise ValueError(
                "Text search requested but no text indexes available for the specified node types"
            )

        if node_types is None:
            if semantic_enabled:
                vector_index_names = [
                    self.grafa_database.get_index_name(None, "vector")
                ]
            if text_enabled:
                text_index_names = [self.grafa_database.get_index_name(None, "text")]

        # Prepare name matching condition for use in queries
        name_match_condition = ""
        if include_name_matching and name_edit_distance is not None:
            name_match_condition = """(
                toLower(node.name) CONTAINS toLower($query)
                OR toLower($query) CONTAINS toLower(node.name)
                OR apoc.text.distance(toLower(node.name), toLower($query)) < $name_edit_distance
                OR ANY(synonym IN node.synonyms WHERE
                    toLower(synonym) CONTAINS toLower($query)
                    OR toLower($query) CONTAINS toLower(synonym)
                    OR apoc.text.distance(toLower(synonym), toLower($query)) < $name_edit_distance
                )
            )"""

        all_records = {}  # Use dict to avoid duplicates by node ID

        # Unified search execution (combining semantic and text query subqueries)
        subqueries = []
        params_union = {
            "database_name": self.db_name,
            "query": clean_string(query),  # Use cleaned query here
            "name_edit_distance": name_edit_distance,
            "limit": limit,
        }

        # Semantic search subquery construction.
        if semantic_enabled and vector_index_names:
            params_union["vector_index_names"] = vector_index_names
            params_union["query_embedding"] = query_embedding
            params_union["semantic_threshold"] = semantic_threshold

            semantic_conditions = ["node.grafa_database_name = $database_name"]
            if semantic_threshold is not None:
                semantic_conditions.append("score > toFloat($semantic_threshold)")
            # Remove name matching from semantic search in hybrid mode
            if (
                include_name_matching
                and name_match_condition
                and search_mode != "hybrid"
            ):
                semantic_conditions.append(name_match_condition)
            semantic_where_clause = "WHERE " + " OR ".join(
                f"({cond})" for cond in semantic_conditions
            )

            # Modify the RETURN clause in semantic subquery based on uuid_only
            return_clause = (
                """
                node.uuid AS uuid,
                node.name AS name,
                node.description AS description,
                node.synonyms AS synonyms,
                labels(node) AS labels,
            """
                if uuid_only
                else """
                node AS node,
                labels(node) AS labels,
            """
            )

            subquery_sem = f"""
            UNWIND $vector_index_names AS index_name
            CALL db.index.vector.queryNodes(index_name, $limit, $query_embedding)
            YIELD node, score
            {semantic_where_clause}
            RETURN
                {return_clause}
                score AS semantic_score,
                null AS text_score,
                false AS name_match,
                score AS score
            """
            subqueries.append(subquery_sem)

        # Text search subquery construction.
        if text_enabled and text_index_names:
            params_union["text_index_names"] = text_index_names
            params_union["text_threshold"] = text_threshold

            text_conditions = ["node.grafa_database_name = $database_name"]
            if text_threshold is not None:
                text_conditions.append("score > toFloat($text_threshold)")
            if include_name_matching and name_match_condition:
                text_conditions.append(name_match_condition)
            text_where_clause = "WHERE " + " OR ".join(
                f"({cond})" for cond in text_conditions
            )

            # Modify the RETURN clause in text subquery based on uuid_only
            return_clause = (
                """
                node.uuid AS uuid,
                node.name AS name,
                node.description AS description,
                node.synonyms AS synonyms,
                labels(node) AS labels,
            """
                if uuid_only
                else """
                node AS node,
                labels(node) AS labels,
            """
            )

            subquery_text = f"""
            UNWIND $text_index_names AS index_name
            CALL db.index.fulltext.queryNodes(index_name, $query)
            YIELD node, score
            {text_where_clause}
            RETURN
                {return_clause}
                null AS semantic_score,
                score AS text_score,
                {name_match_condition} AS name_match,
                score AS score
            """
            subqueries.append(subquery_text)

        if not subqueries:
            raise Exception(
                "No valid indexes available for the requested search modes."
            )

        # If more than one subquery is enabled, combine them using UNION ALL.
        # Otherwise, use the single query directly.
        if len(subqueries) > 1:
            combined_query = (
                "\nUNION ALL\n".join(subqueries) + "\nORDER BY score DESC LIMIT $limit"
            )
        else:
            combined_query = (
                subqueries[0].strip() + "\nORDER BY score DESC LIMIT $limit"
            )
        # Execute the combined Cypher query using the Neo4j driver.
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(combined_query, parameters=params_union)
                records = await result.data()

                # Process and aggregate records by node UUID.
                for record in records:
                    # Get the node type from labels
                    node_labels = record["labels"]
                    # Find the most specific node type (excluding base labels like 'Node')
                    node_type = next(
                        (
                            label
                            for label in node_labels
                            if label not in ["Node", "BaseNode", "USER_DEFINED"]
                        ),
                        node_labels[0],
                    )

                    if not uuid_only:
                        # Convert node dict to Pydantic object
                        node_data = record["node"]
                        pydantic_node = self._create_unpersisted_node(
                            node_type, **node_data
                        )
                        record["node"] = pydantic_node
                    else:
                        record["node_type"] = node_type

                    node_uuid = record.get("uuid") if uuid_only else record["node"].uuid
                    if node_uuid not in all_records:
                        # Store record with both scores
                        all_records[node_uuid] = record
                    else:
                        # Update existing record with maximum scores
                        if record.get("semantic_score") is not None:
                            existing_score = all_records[node_uuid].get(
                                "semantic_score", 0
                            )
                            if existing_score is None:
                                existing_score = 0
                            all_records[node_uuid]["semantic_score"] = max(
                                record.get("semantic_score"), existing_score
                            )
                        if record.get("text_score") is not None:
                            existing_score = all_records[node_uuid].get("text_score", 0)
                            if existing_score is None:
                                existing_score = 0
                            all_records[node_uuid]["text_score"] = max(
                                record.get("text_score"), existing_score
                            )
                        if record.get("name_match"):
                            all_records[node_uuid]["name_match"] = True
        except Exception as e:
            mode = (
                "semantic and text"
                if semantic_enabled and text_enabled
                else "semantic"
                if semantic_enabled
                else "text"
            )
            raise Exception(f"Error during {mode} search: {str(e)}") from e

        # Convert dictionary back to list and sort by highest score of either type
        result_records = list(all_records.values())
        result_records.sort(
            key=lambda x: (
                not x.get(
                    "name_match", False
                ),  # Name matches first (False sorts after True)
                -max(  # Negative for reverse sort by score
                    x.get("semantic_score", 0) or 0, x.get("text_score", 0) or 0
                ),
            )
        )

        # Limit to top results across all indexes if needed
        if len(result_records) > limit:
            result_records = result_records[:limit]

        # Append an 'index' field to each returned record for easier reference
        for i, record in enumerate(result_records):
            record["index"] = i

            # Ensure all score fields exist
            if "semantic_score" not in record:
                record["semantic_score"] = 0
            if "text_score" not in record:
                record["text_score"] = 0
            if "name_match" not in record:
                record["name_match"] = False
            record.pop("score", None)
        return result_records

    @observe
    async def process_chunk(
        self,
        chunk: GrafaChunk,
        deduplication_similarity_threshold: float | None = 0.6,
        deduplication_text_threshold: float | None = 0.4,
        deduplication_word_edit_distance: int | None = 5,
        deduplication_query_limit: int = 100,
    ) -> tuple[list[GrafaBaseNode], RelationshipOutput]:
        """Process a chunk.

        This method extracts entities and relationships from a chunk.
        Then deduplicates entities and creates relationships between them.

        Parameters
        ----------
        chunk : GrafaChunk
            The chunk to process
        deduplication_similarity_threshold : float, optional
            The minimum similarity score threshold to qualify a node, by default 0.9.
        deduplication_text_threshold : float, optional
            The minimum text similarity score threshold to qualify a node, by default 0.9.
        deduplication_word_edit_distance : int, optional
            The maximum allowed edit distance for name similarity checks, by default 5.
        deduplication_query_limit : int, optional
            The maximum number of entities to query, by default 100.

        Returns
        -------
        list[GrafaBaseNode]
            The final entities
        list[Relationship]
            The final relationships
        """
        # Step 1: Extract entities
        entities = await extract_entities(
            chunk.get_embedding_text(),
            self.get_user_defined_node_type_templates(),
            self.llm,
            self.grafa_database.language,
        )

        processed_entities = []
        # Step 2: Deduplicate entities
        for entity in entities.entities:
            entity_node = entity.to_original_class(self)
            similar_entity_results = await self.similarity_search(
                entity_node.get_embedding_text(),
                node_types=type(entity_node),
                limit=deduplication_query_limit,
                semantic_threshold=deduplication_similarity_threshold,
                text_threshold=deduplication_text_threshold,
                name_edit_distance=deduplication_word_edit_distance,
                uuid_only=False,
                include_name_matching=True,
                search_mode="allowed",
            )
            similar_entities = [e["node"] for e in similar_entity_results]
            deduplicated_entity = await deduplicate_entity(
                entity_node,
                similar_entities,
                self.llm,
                self.grafa_database.language,
            )
            await deduplicated_entity.save_to_neo4j()
            if deduplicated_entity.model_config.get("link_to_chunk", False):
                await chunk.link_node(deduplicated_entity)
            processed_entities.append(deduplicated_entity)

        # Step 3: Extract relationships
        relationships = await extract_relationships(
            chunk.get_embedding_text(),
            processed_entities,
            self.get_user_defined_relationship_types(),
            self.llm,
        )
        for r in relationships.relationships:
            if r.from_entity_index in range(
                len(processed_entities)
            ) and r.to_entity_index in range(len(processed_entities)):
                try:
                    await processed_entities[r.from_entity_index].create_relationship(
                        processed_entities[r.to_entity_index], r.type
                    )
                except ValueError:
                    pass
            else:
                raise ValueError(
                    f"Entity at index {r.from_entity_index} or {r.to_entity_index} not found in processed_entities"
                )

        return processed_entities, relationships

    @observe
    async def ingest_file(
        self,
        document_name: str,
        document_path: str | Path | None = None,
        document_text: str | None = None,
        context: str | None = None,
        author: str | None = None,
        source: str | None = None,
        max_token_chunk_size: int = 500,
        chunk_verbose: bool = False,
        chunk_output_language: str | None = None,
        deduplication_similarity_threshold: float | None = 0.6,
        deduplication_text_threshold: float | None = 0.4,
        deduplication_word_edit_distance: int | None = 5,
        deduplication_query_limit: int = 100,
    ) -> tuple[
        GrafaDocument,
        list[GrafaChunk],
        list[list[GrafaBaseNode]],
        list[RelationshipOutput],
    ]:
        """Ingest a file into the Grafa knowledge system.

        This method performs the complete ingestion pipeline:
        1. Uploads and processes the file
        2. Chunks the document
        3. Processes each chunk to extract and deduplicate entities and relationships

        Parameters
        ----------
        document_name : str
            Name of the document to upload
        document_path : str | Path | None, optional
            Path to the document to upload. Must provide either document_path or document_text
        document_text : str | None, optional
            Raw text content to upload. Must provide either document_path or document_text
        context : str | None, optional
            Context of the document to upload
        author : str | None, optional
            Author of the document
        source : str | None, optional
            Source of the document
        max_token_chunk_size : int, optional
            Maximum number of tokens per chunk, by default 500
        chunk_verbose : bool, optional
            Whether to print verbose output during chunking, by default False
        chunk_output_language : str | None, optional
            Language for chunk summaries. If None, uses database default language
        deduplication_similarity_threshold : float | None, optional
            Minimum semantic similarity score (0-1) for entity deduplication, by default 0.6
        deduplication_text_threshold : float | None, optional
            Minimum text similarity score (0-1) for entity deduplication, by default 0.4
        deduplication_word_edit_distance : int | None, optional
            Maximum edit distance for name matching in deduplication, by default 5
        deduplication_query_limit : int, optional
            Maximum number of similar entities to query for deduplication, by default 100

        Returns
        -------
        tuple[GrafaDocument, list[GrafaChunk], list[list[GrafaBaseNode]], list[RelationshipOutput]]
            - The processed document
            - List of document chunks
            - List of processed entities for each chunk
            - List of relationship outputs for each chunk

        Raises
        ------
        ValueError
            If neither document_path nor document_text is provided
        RuntimeError
            If file processing or chunking fails
        """
        load_file = await self.upload_file(
            document_name=document_name,
            document_path=document_path,
            document_text=document_text,
            context=context,
            author=author,
            source=source,
        )
        await self.process_file(load_file)
        document: GrafaDocument = load_file._grafa_document

        chunks = await self.chunk_document(
            document,
            max_token_chunk_size=max_token_chunk_size,
            verbose=chunk_verbose,
            output_language=chunk_output_language,
        )

        processed_entities = []
        relationship_outputs = []
        for chunk in chunks:
            entities, relationships = await self.process_chunk(
                chunk,
                deduplication_similarity_threshold=deduplication_similarity_threshold,
                deduplication_text_threshold=deduplication_text_threshold,
                deduplication_word_edit_distance=deduplication_word_edit_distance,
                deduplication_query_limit=deduplication_query_limit,
            )
            processed_entities.append(entities)
            relationship_outputs.append(relationships)

        return document, chunks, processed_entities, relationship_outputs

    @observe
    async def knowledgebase_query(
        self,
        query: str,
        limit: int = 100,
        semantic_threshold: float | None = 0.6,
        text_threshold: float | None = 0.4,
        name_edit_distance: int | None = 5,
        search_mode: Literal["semantic", "text", "hybrid", "allowed"] = "allowed",
        drill_down_threshold: int = 300,
        include_name_matching: bool = True,
        max_hops: int = 1,
        return_formatted: bool = True,
    ) -> str | dict[str, list[dict]]:
        """Query the knowledgebase.

        This method queries the knowledgebase for the most relevant documents, based on the matched nodes.
        It retrieves the chunks containing these nodes and their surrounding context.

        Parameters
        ----------
        query : str
            The query to search for
        limit : int, optional
            The maximum number of results to return, by default 100
        semantic_threshold : float | None, optional
            The minimum semantic similarity score to return, by default 0.6
        text_threshold : float | None, optional
            The minimum text similarity score to return, by default 0.4
        name_edit_distance : int | None, optional
            The maximum edit distance for name matching, by default 5
        search_mode : Literal["semantic", "text", "hybrid", "allowed"], optional
            The search mode to use, by default "allowed"
        include_name_matching : bool, optional
            Whether to include name matching in the results, by default True
        drill_down_threshold : int, optional
            Query length (in characters) above which drill down search is used, by default 300
        max_hops : int, optional
            Number of chunks to retrieve before and after the matched chunk, by default 1
        return_formatted : bool, optional
            If True, returns a formatted string. If False, returns raw chunk data grouped by document,
            by default True

        Returns
        -------
        str | dict[str, list[dict]]
            If return_formatted is True: A formatted string containing the query results
            If return_formatted is False: A dictionary mapping document names to lists of chunk data
        """
        queries = [query]

        if len(query) > drill_down_threshold:
            queries.extend((await extract_concepts(query, self.llm)).entities)

        search_tasks = []
        for query in queries:
            # Create a task for each similarity search
            task = self.similarity_search(
                query,
                limit=limit,
                semantic_threshold=semantic_threshold,
                text_threshold=text_threshold,
                name_edit_distance=name_edit_distance,
                search_mode=search_mode,
                include_name_matching=include_name_matching,
                uuid_only=True,
            )
            search_tasks.append(task)

        search_results = await asyncio.gather(*search_tasks)

        # Flatten the search results and extract unique UUIDs
        all_uuids = []
        for result_list in search_results:
            for result in result_list:
                all_uuids.append(result["uuid"])

        # Remove duplicates while preserving order
        uuids = []
        for uuid in all_uuids:
            if uuid not in uuids:
                uuids.append(uuid)

        return await self._find_chunks_from_uuids(
            uuids, max_hops=max_hops, return_formatted=return_formatted
        )

    async def _find_chunks_from_uuids(
        self,
        uuids: list[str],
        max_hops: int = 1,
        return_formatted: bool = True,
    ) -> str | dict[str, list[dict]]:
        """Find chunks from a list of uuids.

        Will find all chunks linked to the nodes with the given uuids.
        If max_hops is 0, will only return the chunks linked to the nodes with the given uuids.
        If max_hops is greater than 0, will return the chunks linked to the nodes with the given uuids,
        and the chunks linked to the nodes that are linked to the nodes with the given uuids, and so on.

        Parameters
        ----------
        uuids : list[str]
            The uuids found through similarity_search
        max_hops : int, optional
            Number of chunks to retrieve before and after the matched chunk, by default 1
        return_formatted : bool, optional
            If True, returns a formatted string. If False, returns raw chunk data grouped by document,
            by default True

        Returns
        -------
        str | dict[str, list[dict]]
            If return_formatted is True: A formatted string containing the query results
            If return_formatted is False: A dictionary mapping document names to lists of chunk data
        """
        # Early return if no results found
        if not uuids:
            return "No relevant information found." if return_formatted else {}

        # Updated query using an f-string to embed max_hops literal in the MATCH pattern.
        cypher_query = f"""
        // Match nodes and their connected chunks
        MATCH (n)
        WHERE n.uuid IN $uuids
        MATCH (n)-[*0..1]-(chunk:GrafaChunk)
        WITH DISTINCT chunk

        // Get surrounding chunks within max_hops
        CALL {{
            WITH chunk
            MATCH path = (start)-[r:NEXT_CHUNK*0..{max_hops}]-(chunk)-[r2:NEXT_CHUNK*0..{max_hops}]-(end)
            WHERE all(rel in r WHERE rel.grafa_database_name = $database_name)
              AND all(rel in r2 WHERE rel.grafa_database_name = $database_name)
            RETURN collect(DISTINCT nodes(path)) as context_paths
        }}

        // Unwind paths to get individual chunks with their sequence
        UNWIND context_paths as path
        UNWIND path as context_chunk
        WITH DISTINCT context_chunk
        ORDER BY context_chunk.chunk_sequence_id

        // Return chunk data
        RETURN
            context_chunk.name as chunk_name,
            context_chunk.content as content,
            context_chunk.original_document_name as doc_name,
            context_chunk.chunk_sequence_id as sequence_id,
            context_chunk.original_document_context as doc_context,
            context_chunk.original_document_summary as doc_summary
        ORDER BY doc_name, sequence_id
        """

        # Execute query: note that max_hops is now in the query string, so it is not passed as a parameter.
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                cypher_query,
                parameters={
                    "uuids": uuids,
                    "database_name": self.db_name,
                },
            )
            chunks_data = await result.data()

        # Return empty if no chunks found
        if not chunks_data:
            return "No relevant chunks found." if return_formatted else {}

        # Group chunks by document and deduplicate by chunk name
        doc_chunks = {}
        seen_chunks = set()
        for chunk in chunks_data:
            doc_name = chunk["doc_name"]
            chunk_name = chunk["chunk_name"]

            # Skip if we've already seen this chunk
            if chunk_name in seen_chunks:
                continue

            if doc_name not in doc_chunks:
                doc_chunks[doc_name] = []
            doc_chunks[doc_name].append(chunk)
            seen_chunks.add(chunk_name)

        # Sort chunks within each document by sequence ID
        for chunks in doc_chunks.values():
            chunks.sort(key=lambda x: x["sequence_id"])

        if not return_formatted:
            return doc_chunks

        # Build formatted output
        output = []
        for doc_name, chunks in doc_chunks.items():
            output.append(f"\nDocument: {doc_name}")
            output.append("-" * (len(doc_name) + 10))

            # Add document context and summary if available
            if chunks[0]["doc_context"]:
                output.append(f"\nContext: {chunks[0]['doc_context']}")
            if chunks[0]["doc_summary"]:
                output.append(f"\nSummary: {chunks[0]['doc_summary']}")

            # Add chunks
            for chunk in chunks:
                output.append(f"\nChunk {chunk['sequence_id']}: {chunk['chunk_name']}")
                output.append(chunk["content"])
                output.append("")

        return "\n".join(output)
