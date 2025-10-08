# Grafa

<p align="center">
    <em>Knowledge Graph Generation Library</em>
</p>

[![build](https://github.com/codingmaster8/grafa/workflows/Build/badge.svg)](https://github.com/codingmaster8/grafa/actions)
[![codecov](https://codecov.io/gh/codingmaster8/grafa/branch/master/graph/badge.svg)](https://codecov.io/gh/codingmaster8/grafa)
[![PyPI version](https://badge.fury.io/py/grafa.svg)](https://badge.fury.io/py/grafa)

---

**Documentation**: <a href="https://pablo.vargas.github.io/grafa/" target="_blank">https://codingmaster8.github.io/grafa/</a>

**Source Code**: <a href="https://github.com/codingmaster8/grafa" target="_blank">https://github.com/codingmaster8/grafa</a>

---

## What is Grafa?

Grafa is a comprehensive Python library for building, managing, and querying knowledge graphs. It provides an end-to-end solution for:

- **Document Ingestion**: Upload and process documents (text files, PDFs, etc.)
- **Intelligent Chunking**: Break documents into meaningful chunks using agentic chunking strategies
- **Entity Extraction**: Automatically extract entities and relationships from text using LLMs
- **Knowledge Graph Construction**: Build structured knowledge graphs in Neo4j
- **Smart Search**: Perform semantic, text-based, and hybrid searches across your knowledge base
- **Deduplication**: Automatically merge similar entities to maintain graph quality

## Key Features

### 🚀 Easy Setup
- Schema-driven approach using YAML configuration
- Automatic Neo4j index creation (vector and text indexes)
- Built-in support for AWS S3 storage and local file storage

### 🧠 AI-Powered Processing
- LLM-based entity and relationship extraction
- Semantic similarity search using embeddings
- Intelligent entity deduplication and merging

### 🔍 Advanced Search Capabilities
- **Semantic Search**: Vector-based similarity search
- **Text Search**: Full-text search with fuzzy matching
- **Hybrid Search**: Combines semantic and text approaches
- **Name Matching**: Edit distance-based name matching

### 📊 Flexible Node Types
- Built-in node types: Documents, Chunks, Document History
- Custom node types defined via YAML schema
- Support for metadata, embeddings, and relationships

## Installation

```bash
pip install grafa
```

## Quick Start

### 1. Define Your Schema

Create a YAML file ([schema.yaml](schema.yaml)) to define your knowledge graph structure:

```yaml
database:
  name: "Business Concepts"
  description: "A knowledge graph for business concepts"

node_types:
  Person:
    description: "A person"
    fields:
      occupation:
        type: STRING
        description: "Occupation of the person"
    options:
      link_to_chunk: false
      embed: false

  Company:
    description: "A company"
    fields:
      description:
        type: STRING
        description: "Description of the company"
    options:
      link_to_chunk: false
      embed: false

  Concept:
    description: "A business concept"
    fields:
      description:
        type: STRING
        description: "Description of the concept"
    options:
      link_to_chunk: true
      semantic_search: true
      text_search: true

relationships:
  - from_type: Person
    to_type: Company
    relationship_type: WORKS_AT
    description: "A person works at a company"
  
  - from_type: Company
    to_type: Concept
    relationship_type: IS_RELATED_TO
    description: "A company is related to a concept"
```

### 2. Initialize the Client

```python
from grafa import GrafaClient

# Create client from YAML schema
client = await GrafaClient.from_yaml(
    yaml_path="schema.yaml",
    db_name="my_knowledge_base"
)

# Or connect to existing database
client = await GrafaClient.create(db_name="existing_db")
```

### 3. Ingest Documents

```python
# Upload and process a document
document, chunks, entities, relationships = await client.ingest_file(
    document_name="business_guide",
    document_path="path/to/document.txt",
    context="Business processes and concepts",
    author="John Doe",
    max_token_chunk_size=500,
    deduplication_similarity_threshold=0.6
)

print(f"Created {len(chunks)} chunks")
print(f"Extracted {sum(len(e) for e in entities)} entities")
```

### 4. Search Your Knowledge Base

```python
# Semantic search
results = await client.similarity_search(
    query="What is revenue management?",
    node_types=["Concept"],
    search_mode="semantic",
    limit=10
)

# Hybrid search (semantic + text)
results = await client.similarity_search(
    query="company revenue strategies",
    search_mode="hybrid",
    semantic_threshold=0.7,
    text_threshold=0.5
)

# Knowledge base query (returns formatted context)
answer = await client.knowledgebase_query(
    query="How do we measure promotional effectiveness?",
    max_hops=2,
    return_formatted=True
)
print(answer)
```

## Configuration

### Environment Variables

Set these environment variables for database and storage configuration:

```bash
# Neo4j Configuration
export GRAFA_URI="neo4j+s://your-database.neo4j.io"
export GRAFA_USERNAME="neo4j"
export GRAFA_PASSWORD="your-password"

# Storage Configuration (choose one)
export GRAFA_S3_BUCKET="your-s3-bucket"        # For S3 storage
export GRAFA_LOCAL_STORAGE_PATH="/local/path"  # For local storage
```

### Custom Configuration

```python
from grafa import GrafaClient, GrafaConfig
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Create custom configuration
config = await GrafaConfig.create(
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    embedding_dimension=1536,
    llm=ChatOpenAI(model="gpt-4"),
    s3_bucket="my-documents-bucket"
)

client = await GrafaClient.create(
    db_name="my_db",
    grafa_config=config
)
```

## Schema Definition

### Node Types

Define custom node types with fields and options:

```yaml
node_types:
  Product:
    description: "A product in our catalog"
    fields:
      price:
        type: FLOAT
        description: "Product price"
      category:
        type: STRING
        description: "Product category"
      features:
        type: LIST
        description: "List of product features"
    options:
      link_to_chunk: true      # Link to source chunks
      semantic_search: true    # Enable vector search
      text_search: true        # Enable full-text search
      unique_name: true        # Enforce unique names
```

### Field Types
- `STRING`: Text fields
- `INTEGER`: Numeric integers
- `FLOAT`: Numeric floats
- `BOOLEAN`: True/false values
- `LIST`: Arrays of values
- `DATETIME`: Date and time values

### Node Options
- `link_to_chunk`: Whether nodes link back to source chunks
- `semantic_search`: Enable vector-based semantic search
- `text_search`: Enable full-text search indexing
- `unique_name`: Enforce unique names for this node type
- `embed`: Whether to generate embeddings for this node type

## Advanced Features

### Entity Deduplication

Grafa automatically deduplicates similar entities during ingestion:

```python
# Configure deduplication thresholds
await client.ingest_file(
    document_name="document.txt",
    deduplication_similarity_threshold=0.8,  # Semantic similarity
    deduplication_text_threshold=0.6,        # Text similarity
    deduplication_word_edit_distance=3       # Name edit distance
)
```

### Custom Chunking

Use different chunking strategies:

```python
from grafa.document.chunking import agentic_chunking

# Create document first
document = await client.upload_file(
    document_name="guide.txt",
    document_path="path/to/guide.txt"
)

# Custom chunking with specific parameters
chunks = await client.chunk_document(
    document,
    max_token_chunk_size=800,
    verbose=True,
    output_language="en"
)
```

### Search Modes

Different search strategies for different use cases:

```python
# Pure semantic search (vector embeddings)
semantic_results = await client.similarity_search(
    query="machine learning algorithms",
    search_mode="semantic",
    semantic_threshold=0.75
)

# Pure text search (full-text index)
text_results = await client.similarity_search(
    query="revenue management strategies",
    search_mode="text",
    text_threshold=0.6
)

# Hybrid search (combines both)
hybrid_results = await client.similarity_search(
    query="customer segmentation",
    search_mode="hybrid",
    semantic_threshold=0.7,
    text_threshold=0.5
)

# Automatic mode (uses available indexes)
auto_results = await client.similarity_search(
    query="business metrics",
    search_mode="allowed"  # Default
)
```

## Examples

The [examples/](examples/) directory contains comprehensive examples:

- [`client.ipynb`](examples/client.ipynb): Basic client usage
- [`graphrag.ipynb`](examples/graphrag.ipynb): Complete GraphRAG implementation
- [`search.ipynb`](examples/search.ipynb): Advanced search examples
- [`chunking.ipynb`](examples/chunking.ipynb): Document chunking strategies
- [`database_info.ipynb`](examples/database_info.ipynb): Database schema exploration

## Core Components

### GrafaClient
The main interface for all operations ([grafa/client.py](grafa/client.py)):
- Document ingestion and processing
- Entity extraction and relationship building
- Search and retrieval operations
- Database management

### Node Types
Built-in node types ([grafa/models.py](grafa/models.py)):
- **GrafaDocument**: Represents uploaded documents
- **GrafaChunk**: Document chunks with content and metadata
- **GrafaDocumentHistory**: Version history for documents
- **GrafaDatabase**: Database schema and configuration

### Dynamic Models
Custom node types generated from YAML ([grafa/dynamic_models.py](grafa/dynamic_models.py)):
- Runtime model creation from schema
- Automatic relationship validation
- Field type mapping and validation

## Development

### Setup environment

We use [Hatch](https://hatch.pypa.io/latest/install/) to manage the development environment and production build. Ensure it's installed on your system.

### Run unit tests

You can run all the tests with:

```bash
hatch run test
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
hatch run lint
```

### Publish a new version

You can bump the version, create a commit and associated tag with one command:

```bash
hatch version patch
```

```bash
hatch version minor
```

```bash
hatch version major
```

Your default Git text editor will open so you can add information about the release.

When you push the tag on GitHub, the workflow will automatically publish it on PyPi and a GitHub release will be created as draft.

## Serve the documentation

You can serve the Mkdocs documentation with:

```bash
hatch run docs-serve
```

It'll automatically watch for changes in your code.

## Requirements

- Python 3.8+
- Neo4j database (local or cloud)
- OpenAI API key (for embeddings and LLM operations)
- AWS credentials (if using S3 storage)

## License

This project is licensed under the terms of the Not open source.
