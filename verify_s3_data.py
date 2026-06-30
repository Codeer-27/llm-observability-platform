"""
Quick script to download and inspect one Parquet file from S3
to verify our pipeline is writing correct data.
"""

import boto3
import pandas as pd
import io
from dotenv import load_dotenv
import os

load_dotenv()

s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
bucket = os.getenv("S3_BUCKET_NAME")

# List files in today's partition
prefix = "raw/year=2026/month=06/day=30/"
response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

print(f"Found {len(response['Contents'])} files:\n")
for obj in response["Contents"]:
    print(f"  {obj['Key']}  ({obj['Size']} bytes)")

# Download and read the first file
first_file_key = response["Contents"][0]["Key"]
print(f"\nReading: {first_file_key}\n")

obj = s3_client.get_object(Bucket=bucket, Key=first_file_key)
parquet_bytes = obj["Body"].read()

# pandas can read Parquet directly from bytes in memory
df = pd.read_parquet(io.BytesIO(parquet_bytes))

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

print("\nSample trace (full record):")
print(df.iloc[0].to_dict())