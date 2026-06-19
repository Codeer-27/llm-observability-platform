"""
Kafka Producer — simulates an LLM app generating trace events.

Concept: A producer is anything that SENDS messages to Kafka.
In real life this would be your LLM app. Here we simulate it
by generating realistic fake trace data.
"""

import json
import time
import random
import uuid
from datetime import datetime
from confluent_kafka import Producer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
# This reads KAFKA_BOOTSTRAP_SERVERS etc. into os.environ
load_dotenv()


# ─── Configuration ───────────────────────────────────────────
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_RAW", "llm-traces-raw")


# ─── Producer setup ──────────────────────────────────────────
# A Producer is a Kafka client that sends messages.
# bootstrap.servers tells it where Kafka is running.
producer = Producer({
    "bootstrap.servers": BOOTSTRAP_SERVERS,
    "client.id": "llm-app-producer",
})


# ─── Simulate realistic LLM trace data ───────────────────────
# These are the same fields a real LangChain callback would capture.
MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"]
APPS = ["support-bot", "code-assistant", "summarizer", "search-agent"]
USERS = [f"user-{i}" for i in range(1, 21)]  # 20 simulated users


def generate_trace(inject_anomaly: bool = False) -> dict:
    """
    Generate one fake LLM trace event.
    
    inject_anomaly: if True, create an obviously bad trace
    (very high latency + cost) to test our anomaly detector later.
    """
    model = random.choice(MODELS)
    
    # Normal latency ranges per model (milliseconds)
    latency_ranges = {
        "gpt-4o":        (800,  2000),
        "gpt-4o-mini":   (300,  800),
        "gpt-3.5-turbo": (200,  600),
        "claude-3-haiku":(250,  700),
    }
    
    # Normal cost ranges per model (USD per call)
    cost_ranges = {
        "gpt-4o":        (0.001, 0.008),
        "gpt-4o-mini":   (0.0001, 0.0005),
        "gpt-3.5-turbo": (0.0001, 0.0003),
        "claude-3-haiku":(0.0001, 0.0004),
    }

    if inject_anomaly:
        # Simulate the exact scenario from our use case walkthrough:
        # a latency spike 10-20x normal, cost 100x normal
        latency_ms = random.randint(10000, 20000)
        cost_usd = round(random.uniform(0.30, 0.55), 4)
        prompt_tokens = random.randint(800, 2000)  # unusually long prompt
    else:
        lo, hi = latency_ranges[model]
        latency_ms = random.randint(lo, hi)
        c_lo, c_hi = cost_ranges[model]
        cost_usd = round(random.uniform(c_lo, c_hi), 6)
        prompt_tokens = random.randint(50, 400)

    completion_tokens = random.randint(30, 200)

    trace = {
        "trace_id":          str(uuid.uuid4()),
        "timestamp":         datetime.utcnow().isoformat(),
        "model":             model,
        "app_name":          random.choice(APPS),
        "session_id":        random.choice(USERS),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      prompt_tokens + completion_tokens,
        "latency_ms":        latency_ms,
        "cost_usd":          cost_usd,
        "is_error":          random.random() < 0.03,   # 3% error rate
        "error_type":        None,
    }

    # Add error detail occasionally
    if trace["is_error"]:
        trace["error_type"] = random.choice([
            "RateLimitError", "Timeout", "ContextLengthExceeded"
        ])

    return trace


def delivery_callback(err, msg):
    """
    Kafka calls this function after each message is delivered (or fails).
    
    Concept: Kafka delivery is asynchronous — you call produce() and
    Kafka tries to deliver it. This callback tells you if it succeeded.
    err=None means success. err=something means it failed.
    """
    if err:
        print(f"  FAILED to deliver message: {err}")
    else:
        print(f"  Delivered to topic={msg.topic()} "
              f"partition={msg.partition()} "
              f"offset={msg.offset()}")


def send_trace(trace: dict):
    """Send one trace event to Kafka."""
    
    # Convert dict to JSON string, then to bytes
    # Kafka messages are raw bytes — JSON is the most common format
    message_bytes = json.dumps(trace).encode("utf-8")
    
    # produce() is non-blocking — it queues the message and returns immediately.
    # The actual network send happens in the background.
    # key=trace_id means messages with the same trace_id go to the same partition.
    producer.produce(
        topic=TOPIC,
        key=trace["trace_id"].encode("utf-8"),
        value=message_bytes,
        callback=delivery_callback,
    )
    
    # poll() triggers the delivery callbacks for completed sends
    # Without this, callbacks never fire. Call it regularly.
    producer.poll(0)


# ─── Main loop ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting producer → Kafka at {BOOTSTRAP_SERVERS}")
    print(f"Topic: {TOPIC}")
    print("Sending traces... (Ctrl+C to stop)\n")

    sent_count = 0

    try:
        while True:
            # Every 20th message, inject an anomaly so we can see it in Kafka UI
            inject = (sent_count % 20 == 0) and (sent_count > 0)
            
            trace = generate_trace(inject_anomaly=inject)
            
            if inject:
                print(f"[ANOMALY] trace_id={trace['trace_id'][:8]}... "
                      f"latency={trace['latency_ms']}ms "
                      f"cost=${trace['cost_usd']}")
            else:
                print(f"[NORMAL]  trace_id={trace['trace_id'][:8]}... "
                      f"model={trace['model']} "
                      f"latency={trace['latency_ms']}ms")
            
            send_trace(trace)
            sent_count += 1
            
            # Send one event every second — realistic rate for a small app
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\nStopping. Sent {sent_count} messages total.")
        # flush() waits for all queued messages to be delivered before exiting
        # Without this, the last few messages might not reach Kafka
        producer.flush()
        print("All messages flushed. Done.")