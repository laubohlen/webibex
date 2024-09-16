import boto3 # https://pypi.org/project/boto3/
import numpy as np
from environ import Env
from botocore.config import Config
from botocore.exceptions import ClientError

env = Env()
Env.read_env()
ENVIRONMENT = env("ENVIRONMENT", default="production")

AWS_ACCESS_KEY_ID = env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env("AWS_SECRET_ACCESS_KEY")
AWS_S3_ENDPOINT_URL = env("AWS_S3_ENDPOINT_URL")
AWS_STORAGE_BUCKET_NAME = env("AWS_STORAGE_BUCKET_NAME") 

# Return a boto3 resource object for B2 service
def get_b2_resource(endpoint=AWS_S3_ENDPOINT_URL, key_id=AWS_ACCESS_KEY_ID, application_key=AWS_SECRET_ACCESS_KEY):
    b2 = boto3.resource(service_name='s3',
                        endpoint_url=endpoint,                 # Backblaze endpoint
                        aws_access_key_id=key_id,              # Backblaze keyID
                        aws_secret_access_key=application_key, # Backblaze applicationKey
                        config = Config(
                            signature_version='s3v4',
                    ))
    print("b2", b2)
    return b2


# return a file object from a bucket
def download_file(bucket_file_path, bucket_name=AWS_STORAGE_BUCKET_NAME):
    b2_resource = get_b2_resource()
    # create a client 
    s3_client = b2_resource.meta.client
    try:
        # Use the client to get the object (file) from the bucket
        response = s3_client.get_object(Bucket=bucket_name, Key=bucket_file_path)
        # Read the file content
        file_content = response['Body'].read()
        return file_content
    except ClientError as e:
        print(f"Error occurred while downloading the file: {e}")
        return None


# Delete the specified objects from B2
def delete_files(bucket_file_path_list, bucket_name=AWS_STORAGE_BUCKET_NAME):
    b2_resource = get_b2_resource()
    objects = [{'Key': key} for key in bucket_file_path_list]
    try:
        b2_resource.Bucket(bucket_name).delete_objects(Delete={'Objects': objects})
    except ClientError as ce:
        print('error', ce)
    

# check if file exists in the b2 bucket
def check_file_exists(bucket_file_path, bucket_name=AWS_STORAGE_BUCKET_NAME):
    b2_resource = get_b2_resource()
    try:
        b2_resource.meta.client.head_object(Bucket=bucket_name, Key=bucket_file_path)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            file_exists = False
            print("File not found in Backblaze B2 bucket.")
        else:
            print(f"Unexpected error occurred: {e}")
            return