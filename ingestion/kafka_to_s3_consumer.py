"""
Kafka -> S3 Consumer

Concept: This consumer reads trace events from Kafka, batches them
in memory, and writes them as Parquet files to S3 — organized by
date partitions (Hive-style).

This is the "Bronze" landing zone of our medallion architecture:
raw data, exactly as received, no transformations applied yet.
"""

import json
import os
import io
from datetime import datetime, timezone
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv
import pandas as pd
import boto3

load_dotenv()

# ─── Configuration ───────────────────────────────────────────
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_RAW", "llm-traces-raw")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# Two flush conditions — whichever happens first
BATCH_SIZE = 50          # flush after collecting this many messages
BATCH_TIMEOUT_SEC = 60   # OR flush after this many seconds


# ─── S3 client setup ─────────────────────────────────────────
# boto3 automatically reads AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# from the environment (which load_dotenv() populated from .env).
# We never hardcode credentials directly in code.
s3_client = boto3.client("s3", region_name=AWS_REGION)


# ─── Kafka consumer setup ────────────────────────────────────
consumer = Consumer({
    "bootstrap.servers":  BOOTSTRAP_SERVERS,
    "group.id":           "s3-writer-group",
    "auto.offset.reset":  "earliest",
    "enable.auto.commit": True,
})
consumer.subscribe([TOPIC])


def write_batch_to_s3(records: list[dict]):
    """
    Takes a list of trace dicts, converts to Parquet, uploads to S3
    using Hive-style date partitioning.
    
    Example: 50 trace dicts in -> one file like
    s3://bucket/raw/year=2026/month=06/day=30/traces_143205.parquet
    """
    if not records:
        return

    # Convert list of dicts into a pandas DataFrame (a table structure)
    # Example: [{"trace_id": "a1", "latency_ms": 500}, {"trace_id": "a2", "latency_ms": 800}]
    # becomes a table with columns trace_id, latency_ms and 2 rows
    df = pd.DataFrame(records)

    # Build today's date-based partition path
    now = datetime.now(timezone.utc)
    partition_path = (
        f"raw/year={now.year}/month={now.month:02d}/day={now.day:02d}"
    )

    # Unique filename so concurrent batches in the same minute don't collide
    filename = f"traces_{now.strftime('%H%M%S')}.parquet"
    s3_key = f"{partition_path}/{filename}"

    # Write Parquet bytes into an in-memory buffer (RAM, not disk)
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False, compression="snappy")
    buffer.seek(0)  # rewind buffer to the start before reading from it

    # Upload directly to S3
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=buffer.getvalue(),
    )

    print(f"  Wrote {len(records)} records -> s3://{S3_BUCKET}/{s3_key}")


def main():
    print(f"Consumer listening on topic: {TOPIC}")
    print(f"Writing to S3 bucket: {S3_BUCKET}")
    print(f"Batch size: {BATCH_SIZE} | Batch timeout: {BATCH_TIMEOUT_SEC}s")
    print("Waiting for messages... (Ctrl+C to stop)\n")

    batch = []
    last_flush_time = datetime.now(timezone.utc)
    total_written = 0

    try:
        while True:
            # Wait up to 1 second for a new message
            msg = consumer.poll(timeout=1.0)

            # Check the time-based flush condition on every loop,
            # even if no new message arrived this second
            seconds_since_flush = (
                datetime.now(timezone.utc) - last_flush_time
            ).total_seconds()

            if msg is None:
                # No new message right now — but should we flush due to timeout?
                if batch and seconds_since_flush >= BATCH_TIMEOUT_SEC:
                    write_batch_to_s3(batch)
                    total_written += len(batch)
                    batch = []
                    last_flush_time = datetime.now(timezone.utc)
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"  Kafka error: {msg.error()}")
                continue

            # Decode the message and add it to our in-memory batch
            trace = json.loads(msg.value().decode("utf-8"))
            batch.append(trace)

            print(f"  Buffered #{len(batch)}/{BATCH_SIZE} | "
                  f"trace_id={trace['trace_id'][:8]}...")

            # Size-based flush condition
            if len(batch) >= BATCH_SIZE:
                write_batch_to_s3(batch)
                total_written += len(batch)
                batch = []
                last_flush_time = datetime.now(timezone.utc)

    except KeyboardInterrupt:
        print(f"\nStopping consumer...")
        # Flush any leftover partial batch before exiting
        # Without this, the last few messages would be lost on shutdown
        if batch:
            write_batch_to_s3(batch)
            total_written += len(batch)
        print(f"Total records written to S3: {total_written}")
    finally:
        consumer.close()
        print("Consumer closed cleanly.")


if __name__ == "__main__":
    main()