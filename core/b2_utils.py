import logging
from typing import cast

import boto3  # https://pypi.org/project/boto3/
from botocore.config import Config
from botocore.exceptions import ClientError
from environ import Env
from mypy_boto3_s3 import S3ServiceResource
from mypy_boto3_s3.type_defs import ObjectIdentifierTypeDef

logger = logging.getLogger(__name__)

env = Env()
Env.read_env()
ENVIRONMENT = env("ENVIRONMENT", default="production")

# django-environ ships no type stubs -- env() without a default is
# fail-secure (raises if missing) and always returns str at runtime (see
# tests/webibex/test_infra.py T02 for the equivalent runtime assertion on a
# sibling module). cast() narrows the Unknown/NoValue union pyright infers
# without a stub, per python-types.md.
AWS_ACCESS_KEY_ID = cast(str, env("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = cast(str, env("AWS_SECRET_ACCESS_KEY"))
AWS_S3_ENDPOINT_URL = cast(str, env("AWS_S3_ENDPOINT_URL"))
AWS_STORAGE_BUCKET_NAME = cast(str, env("AWS_STORAGE_BUCKET_NAME"))


# Return a boto3 resource object for B2 service
def get_b2_resource(
    endpoint: str = AWS_S3_ENDPOINT_URL,
    key_id: str = AWS_ACCESS_KEY_ID,
    application_key: str = AWS_SECRET_ACCESS_KEY,
) -> S3ServiceResource:
    b2 = boto3.resource(
        service_name="s3",
        endpoint_url=endpoint,  # Backblaze endpoint
        aws_access_key_id=key_id,  # Backblaze keyID
        aws_secret_access_key=application_key,  # Backblaze applicationKey
        config=Config(
            signature_version="s3v4",
        ),
    )
    logger.debug("b2 resource created: '%s'", b2)
    return b2


# return a file object from a bucket
def download_file(
    bucket_file_path: str, bucket_name: str = AWS_STORAGE_BUCKET_NAME
) -> bytes | None:
    b2_resource = get_b2_resource()
    # create a client
    s3_client = b2_resource.meta.client
    try:
        # Use the client to get the object (file) from the bucket
        response = s3_client.get_object(Bucket=bucket_name, Key=bucket_file_path)
        # Read the file content
        file_content = response["Body"].read()
        return file_content
    except ClientError as e:
        logger.error("Error occurred while downloading the file: %s", e)
        return None


# Delete the specified objects from B2
def delete_files(
    bucket_file_path_list: list[str], bucket_name: str = AWS_STORAGE_BUCKET_NAME
) -> None:
    b2_resource = get_b2_resource()
    objects: list[ObjectIdentifierTypeDef] = [
        {"Key": key} for key in bucket_file_path_list
    ]
    try:
        b2_resource.Bucket(bucket_name).delete_objects(Delete={"Objects": objects})
    except ClientError as ce:
        logger.error("Error deleting files from B2: %s", ce)


# check if file exists in the b2 bucket
def check_file_exists(
    bucket_file_path: str, bucket_name: str = AWS_STORAGE_BUCKET_NAME
) -> bool | None:
    b2_resource = get_b2_resource()
    try:
        b2_resource.meta.client.head_object(Bucket=bucket_name, Key=bucket_file_path)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            # KNOWN BUG (out of scope for this CR, see the B2 S3-mock test
            # tier's T08 case): falls through to `return None` here instead
            # of `return False`. Pinned by test, not fixed as part of this
            # lint-baseline rollout.
            logger.info("File not found in Backblaze B2 bucket.")
        else:
            logger.error("Unexpected error occurred: %s", e)
        return None
