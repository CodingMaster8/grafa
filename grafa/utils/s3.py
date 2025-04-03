"""Utilities for interacting with S3 using an async LRU cache."""

import aioboto3
from async_lru import alru_cache


@alru_cache(maxsize=None)
async def create_bucket(bucket_name: str):
    """
    Create a versioned S3 bucket if it doesn't exist.

    This function checks for the bucket's existence, creates it if necessary,
    and configures versioning. It uses an asynchronous LRU cache to avoid
    repeated calls for the same bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to create

    Returns
    -------
    None

    Notes
    -----
    - Uses aioboto3 for asynchronous S3 client operations.
    - The cache stores the result of the bucket creation operation by bucket name.
    """
    # Create an asynchronous S3 client using aioboto3 session manager.
    session = aioboto3.Session()
    async with session.client("s3") as s3_client:
        # Try to fetch the bucket metadata. If it doesn't exist, an exception is raised.
        try:
            await s3_client.head_bucket(Bucket=bucket_name)
        except s3_client.exceptions.ClientError:
            # Bucket not found; create the bucket.
            await s3_client.create_bucket(
                Bucket=bucket_name,
            )

        # Check the versioning configuration of the bucket.
        versioning = await s3_client.get_bucket_versioning(Bucket=bucket_name)
        if "Status" not in versioning or versioning["Status"] != "Enabled":
            # If versioning is not enabled, enable it.
            await s3_client.put_bucket_versioning(
                Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
            )
