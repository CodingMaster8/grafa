"""Utilities for loading documents into the knowledge graph."""
import base64
from io import BytesIO
from pathlib import Path

import aioboto3
import fsspec
from PIL import Image


async def upload_to_s3(
    bucket_name: str,
    content: str | bytes,
    object_key: str,
    metadata: dict[str, str] = {},
) -> str:
    """Upload content to S3 bucket, handling different input types.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to upload to
    content : str | bytes
        Content to upload - can be a string (file path or text content) or bytes
    object_key : str
        Key (path) where the object will be stored in S3
    metadata : dict[str, str]
        Metadata to attach to the object

    Notes
    -----
    If content is a string, it first checks if it's a valid file path.
    If so, uploads the file contents. If not, uploads the string content directly.
    If content is bytes, uploads the binary data directly.
    """
    session = aioboto3.Session()
    async with session.client("s3") as s3_client:
        # If content is a string, check if it's a file path
        match content:
            case Path() | str() if Path(content).is_file():
                with fsspec.open(content, "rb") as f:
                    await s3_client.put_object(
                        Bucket=bucket_name,
                        Key=object_key,
                        Body=f.read(),
                        Metadata=metadata,
                    )
            case str():
                await s3_client.put_object(
                    Bucket=bucket_name,
                    Key=object_key,
                    Body=content.encode("utf-8"),
                    Metadata=metadata,
                )
            case bytes() | bytearray():
                await s3_client.put_object(
                    Bucket=bucket_name,
                    Key=object_key,
                    Body=content,
                    Metadata=metadata,
                )
            case _:
                raise ValueError(f"Unsupported content type: {type(content)}")


async def get_object_metadata(bucket_name: str, object_key: str) -> tuple[str, str]:
    """Get version ID and hash for an S3 object.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket
    object_key : str
        Key (path) of the S3 object

    Returns
    -------
    tuple[str, str]
        Version ID and MD5 hash of the S3 object
    """
    session = aioboto3.Session()
    async with session.client("s3") as s3_client:
        response = await s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return response["VersionId"], response["ETag"].strip('"')


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 and return as a data URL.

    Parameters
    ----------
    image : Image.Image
        PIL Image object to convert

    Returns
    -------
    str
        The base64 encoded string of the image, prefixed with the appropriate data URL scheme.
    """
    try:
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format or "PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Infer MIME type from image format
        mime_type = f"image/{(image.format or 'png').lower()}"

        # Encode the image to base64
        image_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

        # Return the base64 string with the data URL scheme
        return f"data:{mime_type};base64,{image_base64}"
    except Exception:  # noqa: BLE001
        return ""
