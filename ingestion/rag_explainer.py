"""
RAG Anomaly Explainer — using Groq (free) instead of OpenAI

What this file does in simple words:
1. Reads all anomalies from your Gold ML table in S3
2. Converts each anomaly into a vector (384 numbers representing meaning)
3. Stores those vectors in Pinecone (vector database)
4. When asked to explain an anomaly:
   - Converts the anomaly to a query vector
   - Finds the 3 most similar past incidents in Pinecone
   - Gives those incidents + the anomaly to Groq LLM as context
   - Gets back a plain-English explanation with fix suggestions
This is RAG: Retrieval (Pinecone search) + Augmented + Generation (Groq LLM)
"""

import os
import boto3
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from deltalake import DeltaTable

load_dotenv()

# ── Configuration ─────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION       = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET        = os.getenv("S3_BUCKET_NAME")
PINECONE_INDEX   = "llm-observability"

# ── Initialize clients ────────────────────────────────────────
print("Initializing clients...")

# Pinecone — vector database
# pc.Index() opens our existing index by name
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Embedding model — runs locally on your laptop, completely free
# Downloads ~80MB on first run, cached forever after that
# all-MiniLM-L6-v2 converts any text to a 384-dimensional vector
print("Loading embedding model (first run ~80MB download)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready")

# Groq client — free LLM API
# We use llama-3.1-8b-instant: fast, free, good quality
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL  = "llama-3.1-8b-instant"

# S3 storage options for Delta Lake
storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

print("All clients initialized\n")


# ════════════════════════════════════════════════════════════
# FUNCTION 1: Build knowledge base
# Embeds all past anomalies and stores them in Pinecone
# Think of this as "loading your memory" before answering questions
# ════════════════════════════════════════════════════════════

def build_knowledge_base():
    """
    Reads Gold ML table → converts anomalies to text →
    embeds text to vectors → uploads vectors to Pinecone.

    Run this once. After this, Pinecone remembers all your
    past anomalies and can find similar ones instantly.
    """
    print("=" * 55)
    print("Step 1: Building knowledge base in Pinecone")
    print("=" * 55)

    # Load the Gold ML anomaly scores table from S3
    gold_ml_path = f"s3://{S3_BUCKET}/gold/ml_anomaly_scores"
    print(f"Reading: {gold_ml_path}")

    dt = DeltaTable(gold_ml_path, storage_options=storage_options)
    df = dt.to_pandas()
    print(f"Loaded {len(df)} total traces")

    # Only embed anomalies — no point storing normal calls
    # because we only search for similar ANOMALIES
    df_anomalies = df[df["is_anomaly_ml"] == True].copy()
    print(f"Found {len(df_anomalies)} anomalies to embed")

    if len(df_anomalies) == 0:
        print("No anomalies found. Run Phase 4 first.")
        return None

    # ── Convert each anomaly to a text description ────────────
    # Why text? Because our embedding model understands text.
    # We describe each anomaly in words so the meaning gets
    # captured mathematically in the vector.
    # Example output:
    # "Model gpt-4o-mini from openai family in app support-bot
    #  had a HIGH severity anomaly. Latency was 19067ms..."
    def anomaly_to_text(row):
        return (
            f"Model {row['model']} from {row['model_family']} "
            f"family in app {row['app_name']} had a "
            f"{row['ml_severity']} severity anomaly. "
            f"Latency was {row['latency_ms']}ms "
            f"(bucket: {row['latency_bucket']}), "
            f"cost was ${row['cost_usd']:.4f}, "
            f"total tokens: {row['total_tokens']} "
            f"(prompt: {row['prompt_tokens']}, "
            f"ratio: {row['token_ratio']:.2f}). "
            f"Anomaly score: {row['anomaly_score']:.4f}. "
            f"Error: {row['is_error']}."
        )

    texts = [anomaly_to_text(row) for _, row in df_anomalies.iterrows()]

    print(f"\nSample anomaly description:")
    print(f"  {texts[0]}")

    # ── Embed all anomaly texts into vectors ──────────────────
    # encode() runs the embedding model on all texts at once
    # Input:  list of 11 text strings
    # Output: numpy array of shape (11, 384)
    #         11 vectors, each with 384 numbers
    print(f"\nEmbedding {len(texts)} anomaly descriptions...")
    vectors = embedding_model.encode(texts)
    print(f"Created {len(vectors)} vectors, "
          f"each with {len(vectors[0])} dimensions")

    # ── Upload vectors to Pinecone ────────────────────────────
    # Each vector needs 3 things:
    # 1. id      — unique identifier string
    # 2. values  — the actual vector (list of 384 floats)
    # 3. metadata — original data stored alongside the vector
    #               returned when Pinecone finds a match
    pinecone_vectors = []
    for i, (_, row) in enumerate(df_anomalies.iterrows()):
        pinecone_vectors.append({
            "id":     f"anomaly-{row['trace_id'][:8]}",
            "values": vectors[i].tolist(),
            "metadata": {
                "trace_id":   str(row["trace_id"]),
                "model":      str(row["model"]),
                "app_name":   str(row["app_name"]),
                "latency_ms": int(row["latency_ms"]),
                "cost_usd":   float(row["cost_usd"]),
                "severity":   str(row["ml_severity"]),
                "text":       texts[i],
            }
        })

    # Upload in batches of 10
    # Pinecone recommends batching for performance
    batch_size = 10
    for i in range(0, len(pinecone_vectors), batch_size):
        batch = pinecone_vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Uploaded {len(batch)} vectors to Pinecone")

    # Verify — check how many vectors are now in the index
    stats = index.describe_index_stats()
    print(f"\nPinecone now contains: "
          f"{stats['total_vector_count']} vectors")
    print("Knowledge base built successfully!")

    return df_anomalies


# ════════════════════════════════════════════════════════════
# FUNCTION 2: Find similar past incidents
# Given a new anomaly, search Pinecone for the most similar
# past incidents using vector similarity
# ════════════════════════════════════════════════════════════

def find_similar_incidents(anomaly: dict, top_k: int = 3) -> list:
    """
    Converts the anomaly to a query vector and searches
    Pinecone for the top_k most similar past incidents.

    Concept: cosine similarity measures the angle between
    two vectors. Score of 1.0 = identical meaning.
    Score of 0.0 = completely unrelated.
    We return the 3 incidents with highest similarity scores.
    """

    # Build a text description of the new anomaly
    query_text = (
        f"Model {anomaly['model']} had anomaly with "
        f"latency {anomaly['latency_ms']}ms and "
        f"cost ${anomaly['cost_usd']:.4f}, "
        f"severity {anomaly['severity']}, "
        f"tokens {anomaly['total_tokens']}"
    )

    # Convert to vector
    # This is the same embedding model that encoded our knowledge base
    # So similar text → similar vector → high similarity score in Pinecone
    query_vector = embedding_model.encode(query_text).tolist()

    # Search Pinecone
    # include_metadata=True means return the original text + data
    # not just the vector IDs
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]
    print(f"\nFound {len(matches)} similar past incidents:")
    for i, match in enumerate(matches):
        print(f"  {i+1}. Similarity: {match['score']:.3f} | "
              f"Model: {match['metadata']['model']} | "
              f"Latency: {match['metadata']['latency_ms']}ms")

    return matches


# ════════════════════════════════════════════════════════════
# FUNCTION 3: Generate explanation using Groq LLM
# Takes the anomaly + similar incidents and asks the LLM
# to explain what happened and suggest fixes
# ════════════════════════════════════════════════════════════

def generate_explanation(anomaly: dict, similar_incidents: list) -> str:
    """
    Builds a RAG prompt combining:
    - The current anomaly details
    - Similar past incidents as context
    Then calls Groq LLM to generate an explanation.

    This is the "Augmented Generation" part of RAG:
    Augmented = enriched with retrieved context
    Generation = LLM writes the explanation
    """

    # Build context string from similar incidents
    # We inject this into the prompt so LLM can reference it
    if similar_incidents:
        context_parts = []
        for i, match in enumerate(similar_incidents):
            context_parts.append(
                f"Past incident {i+1} "
                f"(similarity score: {match['score']:.2f}):\n"
                f"{match['metadata']['text']}"
            )
        similar_context = "\n\n".join(context_parts)
    else:
        similar_context = "No similar past incidents found."

    # System prompt — tells the LLM what role to play
    # and how to format its response
    system_prompt = """You are an expert LLM operations engineer 
who monitors AI API performance in production systems. 
You analyse anomalies in LLM API calls — unusual latency spikes, 
cost explosions, token overflows.

When given an anomaly and similar past incidents, you:
1. Identify the most likely root cause in 2-3 sentences
2. Suggest exactly 3 specific actionable fixes
3. Keep language simple and practical

Always be specific — mention the actual model, latency numbers, 
and cost values in your explanation."""

    # User prompt — the actual question with all context injected
    user_prompt = f"""
A new anomaly has been detected in our LLM observability platform.

CURRENT ANOMALY:
- Model: {anomaly['model']}
- App: {anomaly['app_name']}  
- Latency: {anomaly['latency_ms']}ms (normal range: 300-2000ms)
- Cost: ${anomaly['cost_usd']:.4f} (normal range: $0.0001-$0.005)
- Total tokens: {anomaly['total_tokens']}
- Severity: {anomaly['severity']}

SIMILAR PAST INCIDENTS FROM OUR SYSTEM:
{similar_context}

Please explain:
1. What most likely caused this anomaly
2. Three specific fixes the engineering team should apply immediately
"""

    # Call Groq API
    # messages format: list of role+content dicts
    # "system" sets the LLM's behaviour
    # "user" is the actual question
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,   # 0=deterministic, 1=creative
                           # 0.3 = focused but not robotic
        max_tokens=500,    # limit response length
    )

    return response.choices[0].message.content


# ════════════════════════════════════════════════════════════
# FUNCTION 4: Full pipeline — explain any anomaly
# Combines all three functions into one call
# ════════════════════════════════════════════════════════════

def explain_anomaly(anomaly: dict) -> str:
    """
    Main function — given an anomaly dict, returns
    a complete AI-generated explanation with fix suggestions.

    This is what Phase 6 dashboard will call when showing
    an alert to an engineer.
    """
    similar = find_similar_incidents(anomaly)
    explanation = generate_explanation(anomaly, similar)
    return explanation


# ════════════════════════════════════════════════════════════
# MAIN — run the full pipeline end to end
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Step 1: Build knowledge base
    df_anomalies = build_knowledge_base()

    if df_anomalies is not None and len(df_anomalies) > 0:

        print("\n" + "=" * 55)
        print("Step 2: Testing RAG explanation")
        print("=" * 55)

        # Test on the most severe anomaly
        # nsmallest(1, "anomaly_score") = row with lowest
        # (most negative) anomaly score = most anomalous
        worst = df_anomalies.nsmallest(1, "anomaly_score").iloc[0]

        test_anomaly = {
            "model":        worst["model"],
            "app_name":     worst["app_name"],
            "latency_ms":   int(worst["latency_ms"]),
            "cost_usd":     float(worst["cost_usd"]),
            "total_tokens": int(worst["total_tokens"]),
            "severity":     worst["ml_severity"],
        }

        print(f"\nExplaining this anomaly:")
        for key, val in test_anomaly.items():
            print(f"  {key}: {val}")

        # Get the AI explanation
        print("\nCalling Groq LLM...")
        explanation = explain_anomaly(test_anomaly)

        print("\n" + "=" * 55)
        print("AI-GENERATED EXPLANATION:")
        print("=" * 55)
        print(explanation)

        # Test a second anomaly to show the pipeline works generally
        print("\n" + "=" * 55)
        print("Testing on a second anomaly...")
        print("=" * 55)

        if len(df_anomalies) > 1:
            second_worst = df_anomalies.nsmallest(2, "anomaly_score").iloc[1]
            second_anomaly = {
                "model":        second_worst["model"],
                "app_name":     second_worst["app_name"],
                "latency_ms":   int(second_worst["latency_ms"]),
                "cost_usd":     float(second_worst["cost_usd"]),
                "total_tokens": int(second_worst["total_tokens"]),
                "severity":     second_worst["ml_severity"],
            }

            print(f"Anomaly: {second_anomaly['model']} | "
                  f"{second_anomaly['latency_ms']}ms | "
                  f"${second_anomaly['cost_usd']:.4f}")

            explanation2 = explain_anomaly(second_anomaly)
            print("\nAI-GENERATED EXPLANATION:")
            print(explanation2)

        print("\n" + "=" * 55)
        print("PHASE 5 COMPLETE")
        print("=" * 55)
        print("RAG pipeline working end to end:")
        print("  Anomaly detected → Similar incidents found")
        print("  → Groq LLM explains → Fix suggestions generated")
        print(f"\nKnowledge base: Pinecone index '{PINECONE_INDEX}'")
        print(f"LLM used: {GROQ_MODEL} via Groq API")