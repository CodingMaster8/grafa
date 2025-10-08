"""
Example script to test the new logging functionality in Grafa client.

This script demonstrates how to set up logging to see entity and relationship 
identification during document ingestion.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Set up logging configuration
def setup_logging():
    """Configure logging to show Grafa client activity."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler]
    )
    
    # Set Grafa client to DEBUG level for detailed entity/relationship info
    grafa_logger = logging.getLogger('grafa.client')
    grafa_logger.setLevel(logging.DEBUG)
    
    # You can also log to a file
    file_handler = logging.FileHandler('grafa_ingestion.log')
    file_handler.setFormatter(formatter)
    grafa_logger.addHandler(file_handler)


async def test_ingestion():
    """Test document ingestion with logging."""
    
    setup_logging()
    
    # Sample text to ingest
    sample_text = """
    John Smith is a software engineer at TechCorp. He works closely with Sarah Johnson, 
    who is the project manager for the AI initiative. The company, TechCorp, was founded 
    in 2010 and specializes in artificial intelligence solutions. John has been working 
    on a machine learning project that uses natural language processing to analyze 
    customer feedback. Sarah oversees this project and reports to Mike Davis, the 
    VP of Engineering.
    """
    
    try:
        # Note: You would need to create a GrafaClient instance here
        # This is just an example of how the logging would work
        print("=== Starting Grafa Document Ingestion with Logging ===")
        print("Sample text:", sample_text[:100] + "...")
        
        # Example of what the log output would look like:
        logger = logging.getLogger('grafa.client')
        
        logger.info("Starting ingestion of document: sample_document")
        logger.debug("Starting upload for document: sample_document")
        logger.info("Successfully uploaded document: sample_document")
        logger.info("Successfully processed document: sample_document")
        logger.info("Starting chunking for document: sample_document")
        logger.info("Document sample_document chunked into 1 chunks")
        logger.info("Processing chunk 1/1 for document: sample_document")
        
        # Entity extraction logging
        logger.debug("Extracting entities from chunk: sample_document_chunk_1")
        logger.info("Extracted 4 raw entities from chunk sample_document_chunk_1")
        logger.debug("  Entity 1: Person - John Smith")
        logger.debug("  Entity 2: Person - Sarah Johnson") 
        logger.debug("  Entity 3: Organization - TechCorp")
        logger.debug("  Entity 4: Person - Mike Davis")
        
        # Entity processing logging
        logger.debug("Processing entity 1/4: Person")
        logger.debug("  Found 0 similar entities for deduplication")
        logger.info("  Final entity: Person - John Smith (UUID: abc123...)")
        
        logger.debug("Processing entity 2/4: Person")
        logger.debug("  Found 0 similar entities for deduplication")
        logger.info("  Final entity: Person - Sarah Johnson (UUID: def456...)")
        
        logger.debug("Processing entity 3/4: Organization")
        logger.debug("  Found 0 similar entities for deduplication")
        logger.info("  Final entity: Organization - TechCorp (UUID: ghi789...)")
        
        logger.debug("Processing entity 4/4: Person")
        logger.debug("  Found 0 similar entities for deduplication")
        logger.info("  Final entity: Person - Mike Davis (UUID: jkl012...)")
        
        # Relationship extraction logging
        logger.debug("Extracting relationships for 4 entities in chunk sample_document_chunk_1")
        logger.info("Extracted 3 relationships from chunk sample_document_chunk_1")
        logger.info("  Relationship 1: John Smith --[WORKS_AT]--> TechCorp")
        logger.info("  Relationship 2: Sarah Johnson --[MANAGES]--> John Smith")
        logger.info("  Relationship 3: Mike Davis --[SUPERVISES]--> Sarah Johnson")
        
        # Final summary logging
        logger.info("Chunk 1 processed: 4 entities, 3 relationships")
        logger.info("Document sample_document ingestion completed. Total: 4 entities, 3 relationships across 1 chunks")
        
        print("\n=== Logging Example Complete ===")
        print("Check 'grafa_ingestion.log' file for detailed logs")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise


def main():
    """Main function to run the logging test."""
    print("Grafa Client Logging Test")
    print("This script demonstrates the logging output you'll see during document ingestion.")
    print()
    
    # Run the async test
    asyncio.run(test_ingestion())
    
    print("\nLogging Configuration Tips:")
    print("1. Set logging level to INFO to see major steps and entity/relationship counts")
    print("2. Set logging level to DEBUG to see individual entity names and relationship details")
    print("3. Use file handlers to save logs for later analysis")
    print("4. Filter logs by logger name 'grafa.client' for client-specific activity")


if __name__ == "__main__":
    main()
