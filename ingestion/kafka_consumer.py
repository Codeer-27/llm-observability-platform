"""
Kafka Consumer — reads trace events from Kafka and prints them.

Concept: A consumer is anything that READS messages from Kafka.
In our final project, this consumer will write to S3.
For now it just prints so we can verify data is flowing.

Key concept — Consumer Groups:
Multiple consumers can share work by being in the same group_id.
Kafka assigns each partition to exactly one consumer in the group.
If we run 3 consumers with the same group_id and have 3 partitions,
each consumer handles 1 partition in parallel — horizontal scaling.
"""

import json
import os
from datetime import datetime
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_RAW", "llm-traces-raw")


# ─── Consumer setup ──────────────────────────────────────────
consumer = Consumer({
    "bootstrap.servers":  BOOTSTRAP_SERVERS,
    "group.id":           "trace-reader-group",
    # auto.offset.reset controls where to start reading
    # "earliest" = read all messages from the beginning (good for dev)
    # "latest"   = only read new messages (good for production)
    "auto.offset.reset":  "earliest",
    # auto.commit = Kafka automatically saves your position (offset)
    # every 5 seconds. This means if you crash, you restart from
    # at most 5 seconds back — not from the very beginning.
    "enable.auto.commit": True,
})

# Subscribe to our topic
# You can subscribe to multiple topics: consumer.subscribe(["topic1", "topic2"])
consumer.subscribe([TOPIC])

print(f"Consumer listening on topic: {TOPIC}")
print(f"Group ID: trace-reader-group")
print("Waiting for messages... (Ctrl+C to stop)\n")

stats = {"total": 0, "anomalies": 0, "errors": 0}

try:
    while True:
        # poll() waits up to 1 second for a new message
        # Returns None if no message arrived in that time
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            # No message yet — just keep waiting
            continue

        if msg.error():
            # Handle Kafka errors
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # We've read all messages currently in the partition
                # This is not an error — just means we're caught up
                print(f"  Reached end of partition {msg.partition()}")
            else:
                print(f"  Kafka error: {msg.error()}")
            continue

        # ── Decode the message ────────────────────────────────
        # msg.value() returns raw bytes — decode to string, then parse JSON
        raw_value = msg.value().decode("utf-8")
        trace = json.loads(raw_value)

        stats["total"] += 1
        
        # ── Detect if this is an anomaly (simple rule for now) ──
        is_anomaly = trace["latency_ms"] > 5000 or trace["cost_usd"] > 0.1
        if is_anomaly:
            stats["anomalies"] += 1
        if trace["is_error"]:
            stats["errors"] += 1

        # ── Print a readable summary ──────────────────────────
        flag = "ANOMALY" if is_anomaly else "normal "
        print(
            f"[{flag}] "
            f"#{stats['total']:04d} | "
            f"model={trace['model']:<16} | "
            f"latency={trace['latency_ms']:>6}ms | "
            f"cost=${trace['cost_usd']:.6f} | "
            f"app={trace['app_name']}"
        )

        # Every 10 messages, print a summary
        if stats["total"] % 10 == 0:
            print(f"\n  --- Stats so far: "
                  f"total={stats['total']} | "
                  f"anomalies={stats['anomalies']} | "
                  f"errors={stats['errors']} ---\n")

except KeyboardInterrupt:
    print(f"\nStopping consumer.")
    print(f"Final stats: {stats}")
finally:
    # Always close the consumer cleanly
    # This tells Kafka "this consumer is leaving the group"
    # so Kafka can reassign its partitions to other consumers immediately
    consumer.close()
    print("Consumer closed cleanly.")