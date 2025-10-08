# Grafa Client Logging Guide

The Grafa client now includes comprehensive logging to help you track document ingestion progress, entity identification, and relationship extraction. This guide explains how to set up and use the logging features.

## Quick Start

```python
import logging
from grafa import GrafaClient  # Adjust import as needed

# Basic logging setup
logging.basicConfig(level=logging.INFO)

# For detailed entity/relationship info, set DEBUG level
grafa_logger = logging.getLogger('grafa.client')
grafa_logger.setLevel(logging.DEBUG)

# Now use GrafaClient as usual
# client = await GrafaClient.create(...)
```

## What Gets Logged

### Document Ingestion (`ingest_file`)
- **INFO**: Document ingestion start/completion
- **INFO**: Upload and processing milestones  
- **INFO**: Chunking progress and chunk counts
- **INFO**: Total entity and relationship counts per document

### Entity Extraction (`process_chunk`)
- **DEBUG**: Raw entity extraction from each chunk
- **DEBUG**: Individual entity names and types
- **INFO**: Final processed entities with UUIDs after deduplication
- **INFO**: Entity processing progress (entity X of Y)

### Relationship Extraction
- **DEBUG**: Relationship extraction initiation
- **INFO**: Individual relationships with entity names and types
- **INFO**: Relationship counts per chunk
- **WARNING**: Failed relationship creation attempts

### Upload and Processing
- **DEBUG**: File upload initiation
- **INFO**: Document upload completion
- **INFO**: Document processing completion

## Logging Levels

### INFO Level (Recommended for Production)
Shows major milestones and summaries:
```
2024-09-30 10:15:23 - grafa.client - INFO - Starting ingestion of document: my_document
2024-09-30 10:15:25 - grafa.client - INFO - Successfully uploaded document: my_document  
2024-09-30 10:15:27 - grafa.client - INFO - Document my_document chunked into 3 chunks
2024-09-30 10:15:30 - grafa.client - INFO - Final entity: Person - John Smith (UUID: abc123...)
2024-09-30 10:15:31 - grafa.client - INFO - Relationship 1: John Smith --[WORKS_AT]--> TechCorp
2024-09-30 10:15:35 - grafa.client - INFO - Document my_document ingestion completed. Total: 15 entities, 8 relationships across 3 chunks
```

### DEBUG Level (For Development/Debugging)  
Shows detailed extraction process:
```
2024-09-30 10:15:28 - grafa.client - DEBUG - Extracting entities from chunk: my_document_chunk_1
2024-09-30 10:15:29 - grafa.client - DEBUG - Entity 1: Person - John Smith
2024-09-30 10:15:29 - grafa.client - DEBUG - Entity 2: Organization - TechCorp
2024-09-30 10:15:29 - grafa.client - DEBUG - Processing entity 1/2: Person
2024-09-30 10:15:30 - grafa.client - DEBUG - Found 2 similar entities for deduplication
```

## Advanced Logging Configuration

### Log to File
```python
import logging

# Create file handler
file_handler = logging.FileHandler('grafa_ingestion.log')
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

# Add to Grafa logger
grafa_logger = logging.getLogger('grafa.client')
grafa_logger.addHandler(file_handler)
```

### Filter by Operation
You can filter logs by searching for specific patterns:
- `"Starting ingestion"` - Document ingestion starts
- `"Final entity:"` - Successfully processed entities
- `"Relationship"` - Created relationships
- `"ingestion completed"` - Ingestion summaries

### Custom Log Format
```python
# Detailed format with function names
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
)

# Minimal format for production
formatter = logging.Formatter('%(levelname)s: %(message)s')
```

## Example Output

Here's what you'll see during a typical document ingestion:

```
INFO - Starting ingestion of document: research_paper.pdf
DEBUG - Starting upload for document: research_paper.pdf  
INFO - Successfully uploaded document: research_paper.pdf
INFO - Successfully processed document: research_paper.pdf
INFO - Starting chunking for document: research_paper.pdf
INFO - Document research_paper.pdf chunked into 5 chunks
INFO - Processing chunk 1/5 for document: research_paper.pdf
DEBUG - Extracting entities from chunk: research_paper.pdf_chunk_1
INFO - Extracted 8 raw entities from chunk research_paper.pdf_chunk_1
DEBUG - Entity 1: Person - Dr. Sarah Chen
DEBUG - Entity 2: Organization - Stanford University  
DEBUG - Entity 3: Concept - Machine Learning
INFO - Final entity: Person - Dr. Sarah Chen (UUID: abc123...)
INFO - Final entity: Organization - Stanford University (UUID: def456...)
DEBUG - Extracting relationships for 8 entities in chunk research_paper.pdf_chunk_1
INFO - Extracted 5 relationships from chunk research_paper.pdf_chunk_1
INFO - Relationship 1: Dr. Sarah Chen --[AFFILIATED_WITH]--> Stanford University
INFO - Relationship 2: Dr. Sarah Chen --[RESEARCHES]--> Machine Learning
INFO - Chunk 1 processed: 8 entities, 5 relationships
...
INFO - Document research_paper.pdf ingestion completed. Total: 35 entities, 22 relationships across 5 chunks
```

## Troubleshooting

### Too Verbose?
- Set level to `logging.WARNING` to see only errors
- Set level to `logging.INFO` for key milestones only

### Missing Logs?
- Ensure you've set the logger level: `grafa_logger.setLevel(logging.DEBUG)`
- Check that logging is configured before creating the GrafaClient

### Performance Impact
- DEBUG level logging may slow down ingestion slightly
- Consider using INFO level for production workloads
- File logging is generally faster than console logging

## Integration Tips

### With Jupyter Notebooks
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
grafa_logger = logging.getLogger('grafa.client')
grafa_logger.setLevel(logging.INFO)  # Avoid too much output in notebooks
```

### With Production Applications
```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'grafa.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
        }
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'loggers': {
        'grafa.client': {
            'level': 'INFO',
            'handlers': ['file']
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```
