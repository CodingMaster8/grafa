"""Module for loading documents into the knowledge graph."""
import os
from pathlib import Path

from grafa.models import LoadFile

from .input_format import FILE_PROCESSORS
from .utils import get_object_metadata, upload_to_s3


def _get_object_key(db_name: str, file_name: str, source: bool = True) -> str:
    """Get the object key for a file in the S3 bucket.

    Parameters
    ----------
    db_name : str
        The name of the database to load the files into
    """
    return (
        f"grafa/{db_name}/source/{file_name}"
        if source
        else f"grafa/{db_name}/processed/{file_name}"
    )


async def upload_file(file: LoadFile, s3_bucket: str, db_name: str) -> LoadFile:
    """Upload a file to Grafa's S3 bucket.

    Parameters
    ----------
    file : LoadFile
        The file to upload
    s3_bucket : str
        The name of the S3 bucket to load the files into
    db_name : str
        The name of the database to load the files into

    Returns
    -------
    LoadFile
        The file with the populated attributes
    """

    # If s3 bucket is a local path, save locally
    if not s3_bucket.startswith("s3://"):
        local_dir = Path(s3_bucket)
        local_dir.mkdir(parents=True, exist_ok=True)
        file_path = local_dir / f"{db_name}_{file.name}"
        file_path.write_text(file.content or "")
        file._raw_object_key = str(file_path)
        file._raw_etag = str(hash(file.content or ""))
        file._raw_version_id = "local"
        return file
    
    object_key = _get_object_key(db_name, file.name)
    # Populate file attributes
    file._s3_bucket = s3_bucket
    file._database_name = db_name

    if file.content is not None:
        content = file.content
    elif file.path is not None:
        content = file.path
    else:
        raise ValueError(f"File {file.name} has no content or path")

    metadata = {}
    if file.context:
        metadata["context"] = file.context
    if file.author:
        metadata["author"] = file.author
    if file.source:
        metadata["source"] = file.source

    await upload_to_s3(s3_bucket, content, object_key, metadata)
    version_id, etag = await get_object_metadata(s3_bucket, object_key)
    file._raw_object_key = object_key
    file._raw_version_id = version_id
    file._raw_etag = etag

    return file


async def process_file(file: LoadFile) -> LoadFile:
    """Process a file and return a dictionary with the file paths as keys and the processed content as values.

    Parameters
    ----------
    file : LoadFile
        The file to process

    Returns
    -------
    LoadFile
        The file with the populated attributes
    """
    if file._raw_object_key is None:
        raise ValueError(f"File {file.name} has no raw object key")
    if file._s3_bucket is None:
        raise ValueError(f"File {file.name} has no S3 bucket")
    if file._database_name is None:
        raise ValueError(f"File {file.name} has no database name")
    if file._raw_version_id is None:
        raise ValueError(f"File {file.name} has no raw version ID")
    if file._raw_etag is None:
        raise ValueError(f"File {file.name} has no raw ETag")
    if file._grafa_document is None:
        raise ValueError(f"File {file.name} has no GrafaDocument")

    # Extract file content
    ext = file._grafa_document.extension
    if ext not in FILE_PROCESSORS:
        raise ValueError(f"Unsupported file type: {ext}")
    file_content = await FILE_PROCESSORS[ext](file.path)

    metadata = {}
    if file.context:
        metadata["context"] = file.context
    
    # Check if using local storage (when _s3_bucket doesn't start with s3://)
    if file._s3_bucket and not file._s3_bucket.startswith("s3://"):
        # Local storage handling
        processed_file_path = Path(file._s3_bucket) / f"processed_{file.name}.txt"
        processed_file_path.write_text(file_content)
        
        file._processed_object_key = str(processed_file_path)
        file._processed_version_id = "local"
        file._processed_etag = str(hash(file_content))
        
        metadata["origin"] = f"file://{file._raw_object_key}"
    else:
        # S3 storage handling
        metadata["origin"] = f"s3://{file._s3_bucket}/{file._raw_object_key}"
        
        processed_object_key = _get_object_key(file._database_name, file.name, source=False)

        await upload_to_s3(
            file._s3_bucket, file_content, processed_object_key, metadata=metadata
        )
        version_id, etag = await get_object_metadata(file._s3_bucket, processed_object_key)

        file._processed_object_key = processed_object_key
        file._processed_version_id = version_id
        file._processed_etag = etag

    file.content = file_content

    return file
