# Databricks notebook source
# Cell 1: Install boto3 and configure AWS access
# In Databricks Serverless we use boto3 directly to read from S3
# then create Spark DataFrames from the data

import subprocess
subprocess.run(["pip", "install", "boto3", "pyarrow", "pandas"], 
               capture_output=True)

import boto3
import pandas as pd
import pyarrow.parquet as pq
import io
import os

# Your AWS credentials — paste your actual values here
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

# Create S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print("AWS connection configured")
print(f"Bucket: {S3_BUCKET}")

# Test connection — list what's in your raw folder
response = s3_client.list_objects_v2(
    Bucket=S3_BUCKET,
    Prefix="raw/"
)

files = response.get("Contents", [])
print(f"\nFound {len(files)} files in S3 raw folder:")
for f in files:
    print(f"  {f['Key']}  ({f['Size']} bytes)")

# COMMAND ----------

# Cell 2: Read all Parquet files from S3 and create Bronze Delta table
#
# Concept: We read ALL files under raw/ in one command.
# boto3 lists them, we download each one, convert to pandas,
# then combine into one big Spark DataFrame.
# Bronze = raw data exactly as it arrived, just unified and formatted.

import pandas as pd
import pyarrow.parquet as pq
import io
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, FloatType, BooleanType
)

# ── Step 1: Download all Parquet files from S3 into memory ──
print("Reading Parquet files from S3...")
all_dataframes = []

for file_obj in files:
    key = file_obj["Key"]
    
    # Download file bytes from S3 into memory (not to disk)
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    file_bytes = response["Body"].read()
    
    # Read Parquet bytes into a pandas DataFrame
    pandas_df = pd.read_parquet(io.BytesIO(file_bytes))
    all_dataframes.append(pandas_df)
    print(f"  Read {len(pandas_df)} rows from {key.split('/')[-1]}")

# ── Step 2: Combine all pandas DataFrames into one ──
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"\nTotal rows combined: {len(combined_df)}")
print(f"Columns: {list(combined_df.columns)}")

# ── Step 3: Convert pandas DataFrame to Spark DataFrame ──
# Why? Spark DataFrames can be written as Delta tables.
# Pandas DataFrames cannot. We use pandas as a bridge.
spark_df = spark.createDataFrame(combined_df)

print(f"\nSpark DataFrame created successfully")
print(f"Total records: {spark_df.count()}")
print("\nSchema:")
spark_df.printSchema()

print("\nSample data (first 5 rows):")
spark_df.show(5, truncate=True)


# COMMAND ----------

# Cell 3: Write Bronze Delta table — fully self-contained
# Re-declares everything it needs so it works independently

import subprocess
subprocess.run(["pip", "install", "deltalake"], capture_output=True)

import boto3
import pandas as pd
import io
from deltalake.writer import write_deltalake
from deltalake import DeltaTable

# ── Credentials (re-declared) ────────────────────────────────
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

# ── S3 client ────────────────────────────────────────────────
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# ── Re-download all Parquet files from S3 ───────────────────
print("Re-reading Parquet files from S3...")
response = s3_client.list_objects_v2(
    Bucket=S3_BUCKET,
    Prefix="raw/"
)
files = response.get("Contents", [])

all_dfs = []
for file_obj in files:
    key = file_obj["Key"]
    if not key.endswith(".parquet"):
        continue
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    file_bytes = obj["Body"].read()
    df = pd.read_parquet(io.BytesIO(file_bytes))
    all_dfs.append(df)
    print(f"  Read {len(df)} rows from {key.split('/')[-1]}")

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows: {len(combined_df)}")
print(f"Columns: {list(combined_df.columns)}")

# ── Storage options for Delta Lake ──────────────────────────
storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

bronze_path = f"s3://{S3_BUCKET}/bronze/llm_traces"

# ── Write Bronze Delta table ─────────────────────────────────
print(f"\nWriting Bronze Delta table to: {bronze_path}")
print("Writing... (30-60 seconds)")

write_deltalake(
    bronze_path,
    combined_df,
    mode="overwrite",
    storage_options=storage_options,
)

print("\nBronze Delta table written successfully!")

# ── Verify by reading back ───────────────────────────────────
dt = DeltaTable(bronze_path, storage_options=storage_options)
df_verify = dt.to_pandas()

print(f"Verification: {len(df_verify)} rows in Bronze Delta table")
print(f"Columns: {list(df_verify.columns)}")
print("\nSample rows:")
print(df_verify.head(5).to_string())

# COMMAND ----------

# Cell 4: Silver layer — clean and enrich Bronze data
#
# Concept: Silver NEVER modifies Bronze data.
# It only ADDS new derived columns on top.
# If we find a bug in Silver, we reprocess from Bronze safely.
#
# What we're adding:
# - Proper timestamp type (not just a string)
# - Date and hour columns (for time-based analysis)
# - Cost per token (efficiency metric)
# - Token ratio (prompt vs completion balance)
# - Model family (openai vs anthropic grouping)
# - Latency bucket (human-readable performance category)
# - Anomaly flag (rule-based, before ML in Phase 4)

import boto3
import pandas as pd
import io
import numpy as np
from deltalake.writer import write_deltalake
from deltalake import DeltaTable

# ── Credentials ──────────────────────────────────────────────
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# ── Read Bronze Delta table ──────────────────────────────────
bronze_path = f"s3://{S3_BUCKET}/bronze/llm_traces"
print("Reading Bronze Delta table...")

dt_bronze = DeltaTable(bronze_path, storage_options=storage_options)
df_bronze = dt_bronze.to_pandas()
print(f"Bronze rows loaded: {len(df_bronze)}")

# ── Apply Silver transformations ─────────────────────────────
print("\nApplying Silver transformations...")
df_silver = df_bronze.copy()

# 1. Convert timestamp string to proper datetime type
#    "2026-06-30T13:20:03.864618" → a real datetime object
#    This enables time-based operations like groupby hour, date filtering
df_silver["event_timestamp"] = pd.to_datetime(df_silver["timestamp"])

# 2. Extract date part — for daily aggregations
#    "2026-06-30T13:20:03" → date(2026, 6, 30)
df_silver["event_date"] = df_silver["event_timestamp"].dt.date.astype(str)

# 3. Extract hour — for hourly trend charts
#    "2026-06-30T13:20:03" → 13
df_silver["event_hour"] = df_silver["event_timestamp"].dt.hour

# 4. Cost per token — how efficiently is each model spending money?
#    Lower = more efficient. GPT-4o-mini should be much lower than GPT-4o.
#    We use np.where to avoid division by zero if total_tokens is 0
df_silver["cost_per_token"] = np.where(
    df_silver["total_tokens"] > 0,
    df_silver["cost_usd"] / df_silver["total_tokens"],
    0.0
)

# 5. Token ratio — what fraction of tokens are prompt vs completion?
#    High ratio (>0.8) = very long prompts, might explain high cost
#    Low ratio (<0.2) = very short prompts, mostly completion tokens
df_silver["token_ratio"] = np.where(
    df_silver["total_tokens"] > 0,
    df_silver["prompt_tokens"] / df_silver["total_tokens"],
    0.0
)

# 6. Model family — group models by provider
#    Useful for "OpenAI vs Anthropic" comparisons in dashboard
df_silver["model_family"] = df_silver["model"].apply(
    lambda m: "openai" if str(m).startswith("gpt")
    else "anthropic" if str(m).startswith("claude")
    else "other"
)

# 7. Latency bucket — human-readable performance category
#    This powers the "performance distribution" chart in dashboard
def latency_bucket(ms):
    if ms < 500:   return "fast (<500ms)"
    if ms < 1000:  return "normal (500ms-1s)"
    if ms < 3000:  return "slow (1s-3s)"
    if ms < 5000:  return "very slow (3s-5s)"
    return "anomaly (>5s)"

df_silver["latency_bucket"] = df_silver["latency_ms"].apply(latency_bucket)

# 8. Rule-based anomaly flag
#    True if latency > 5 seconds OR cost > $0.10 per call
#    This is our simple rule. Phase 4 replaces this with ML.
df_silver["is_anomaly_rule"] = (
    (df_silver["latency_ms"] > 5000) |
    (df_silver["cost_usd"] > 0.10)
)

# Drop the original string timestamp — replaced by event_timestamp
df_silver = df_silver.drop(columns=["timestamp"])

print(f"Silver rows: {len(df_silver)}")
print(f"\nNew columns added:")
new_cols = ["event_timestamp","event_date","event_hour",
            "cost_per_token","token_ratio","model_family",
            "latency_bucket","is_anomaly_rule"]
print(df_silver[new_cols].head(5).to_string())

# ── Show some interesting stats ──────────────────────────────
print("\n--- Latency distribution ---")
print(df_silver["latency_bucket"].value_counts().to_string())

print("\n--- Model family breakdown ---")
print(df_silver["model_family"].value_counts().to_string())

print("\n--- Anomalies detected (rule-based) ---")
print(df_silver["is_anomaly_rule"].value_counts().to_string())

print(f"\nAverage latency: {df_silver['latency_ms'].mean():.0f}ms")
print(f"Average cost per call: ${df_silver['cost_usd'].mean():.6f}")
print(f"Total cost across all calls: ${df_silver['cost_usd'].sum():.4f}")

# ── Write Silver Delta table ──────────────────────────────────
silver_path = f"s3://{S3_BUCKET}/silver/llm_traces"
print(f"\nWriting Silver Delta table to: {silver_path}")

write_deltalake(
    silver_path,
    df_silver,
    mode="overwrite",
    storage_options=storage_options,
)

print("Silver Delta table written successfully!")

# ── Verify ───────────────────────────────────────────────────
dt_silver = DeltaTable(silver_path, storage_options=storage_options)
df_check = dt_silver.to_pandas()
print(f"Verification: {len(df_check)} rows in Silver Delta table")
print(f"Total columns: {len(df_check.columns)} (was 12 in Bronze, now {len(df_check.columns)} in Silver)")

# COMMAND ----------

# Cell 5: Gold layer — business-ready aggregations
#
# Concept: Gold tables are what dashboards, ML models, and
# business analysts actually consume. They are pre-aggregated
# so queries are instant — no heavy computation at query time.
#
# We create 3 Gold tables:
# 1. hourly_metrics    — performance trends over time per model
# 2. model_comparison  — which model is best overall?
# 3. anomaly_summary   — how many anomalies, what type, when?

import boto3
import pandas as pd
import numpy as np
import io
from deltalake.writer import write_deltalake
from deltalake import DeltaTable

# ── Credentials ──────────────────────────────────────────────
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# ── Read Silver Delta table ──────────────────────────────────
silver_path = f"s3://{S3_BUCKET}/silver/llm_traces"
print("Reading Silver Delta table...")
dt_silver = DeltaTable(silver_path, storage_options=storage_options)
df_silver = dt_silver.to_pandas()
print(f"Silver rows loaded: {len(df_silver)}")

# ════════════════════════════════════════════════════════════
# GOLD TABLE 1: Hourly metrics per model
# One row per model per hour — powers the time-series charts
# ════════════════════════════════════════════════════════════
print("\n--- Building Gold Table 1: Hourly metrics ---")

df_gold_hourly = df_silver.groupby(
    ["event_date", "event_hour", "model", "model_family"]
).agg(
    total_calls        = ("trace_id",       "count"),
    avg_latency_ms     = ("latency_ms",     "mean"),
    min_latency_ms     = ("latency_ms",     "min"),
    max_latency_ms     = ("latency_ms",     "max"),
    p95_latency_ms     = ("latency_ms",     lambda x: x.quantile(0.95)),
    total_cost_usd     = ("cost_usd",       "sum"),
    avg_cost_usd       = ("cost_usd",       "mean"),
    total_tokens       = ("total_tokens",   "sum"),
    avg_tokens         = ("total_tokens",   "mean"),
    error_count        = ("is_error",       "sum"),
    anomaly_count      = ("is_anomaly_rule","sum"),
).reset_index()

# Cost per 1000 calls — standard business metric
# "How much does it cost us per 1000 AI calls?"
df_gold_hourly["cost_per_1k_calls"] = (
    df_gold_hourly["total_cost_usd"] /
    df_gold_hourly["total_calls"] * 1000
)

# Error rate as percentage
df_gold_hourly["error_rate_pct"] = (
    df_gold_hourly["error_count"] /
    df_gold_hourly["total_calls"] * 100
).round(2)

# Anomaly rate as percentage
df_gold_hourly["anomaly_rate_pct"] = (
    df_gold_hourly["anomaly_count"] /
    df_gold_hourly["total_calls"] * 100
).round(2)

print(f"Hourly metrics rows: {len(df_gold_hourly)}")
print(df_gold_hourly[["model","event_hour","total_calls",
                        "avg_latency_ms","total_cost_usd",
                        "anomaly_count"]].to_string())

# Write Gold Table 1
gold_hourly_path = f"s3://{S3_BUCKET}/gold/hourly_metrics"
write_deltalake(
    gold_hourly_path,
    df_gold_hourly,
    mode="overwrite",
    storage_options=storage_options,
)
print(f"\nGold hourly_metrics written to S3")


# ════════════════════════════════════════════════════════════
# GOLD TABLE 2: Model comparison
# One row per model — answers "which model is best?"
# ════════════════════════════════════════════════════════════
print("\n--- Building Gold Table 2: Model comparison ---")

df_gold_models = df_silver.groupby(
    ["model", "model_family"]
).agg(
    total_calls        = ("trace_id",        "count"),
    avg_latency_ms     = ("latency_ms",      "mean"),
    p50_latency_ms     = ("latency_ms",      lambda x: x.quantile(0.50)),
    p95_latency_ms     = ("latency_ms",      lambda x: x.quantile(0.95)),
    avg_cost_per_call  = ("cost_usd",        "mean"),
    total_cost_usd     = ("cost_usd",        "sum"),
    avg_cost_per_token = ("cost_per_token",  "mean"),
    avg_token_ratio    = ("token_ratio",     "mean"),
    error_count        = ("is_error",        "sum"),
    anomaly_count      = ("is_anomaly_rule", "sum"),
).reset_index()

df_gold_models["error_rate_pct"] = (
    df_gold_models["error_count"] /
    df_gold_models["total_calls"] * 100
).round(2)

df_gold_models["anomaly_rate_pct"] = (
    df_gold_models["anomaly_count"] /
    df_gold_models["total_calls"] * 100
).round(2)

# Round for readability
df_gold_models["avg_latency_ms"]    = df_gold_models["avg_latency_ms"].round(0)
df_gold_models["avg_cost_per_call"] = df_gold_models["avg_cost_per_call"].round(6)
df_gold_models["total_cost_usd"]    = df_gold_models["total_cost_usd"].round(4)

print(f"Model comparison rows: {len(df_gold_models)}")
print("\nModel performance leaderboard:")
print(df_gold_models[[
    "model","total_calls","avg_latency_ms",
    "p95_latency_ms","avg_cost_per_call",
    "anomaly_rate_pct"
]].sort_values("avg_latency_ms").to_string())

# Write Gold Table 2
gold_models_path = f"s3://{S3_BUCKET}/gold/model_comparison"
write_deltalake(
    gold_models_path,
    df_gold_models,
    mode="overwrite",
    storage_options=storage_options,
)
print(f"\nGold model_comparison written to S3")


# ════════════════════════════════════════════════════════════
# GOLD TABLE 3: Anomaly summary
# Details of every flagged anomaly — feeds the alert dashboard
# ════════════════════════════════════════════════════════════
print("\n--- Building Gold Table 3: Anomaly summary ---")

df_anomalies = df_silver[df_silver["is_anomaly_rule"] == True].copy()

df_anomalies["anomaly_type"] = df_anomalies.apply(
    lambda row:
    "latency_and_cost_spike" if (row["latency_ms"] > 5000 and row["cost_usd"] > 0.10)
    else "latency_spike"     if row["latency_ms"] > 5000
    else "cost_spike",
    axis=1
)

df_anomalies["severity"] = df_anomalies.apply(
    lambda row:
    "CRITICAL" if (row["latency_ms"] > 15000 or row["cost_usd"] > 0.40)
    else "HIGH" if (row["latency_ms"] > 10000 or row["cost_usd"] > 0.20)
    else "MEDIUM",
    axis=1
)

df_gold_anomalies = df_anomalies[[
    "trace_id","event_timestamp","model","app_name",
    "session_id","latency_ms","cost_usd","total_tokens",
    "is_error","anomaly_type","severity"
]].copy()

print(f"Total anomalies: {len(df_gold_anomalies)}")
print("\nAnomaly breakdown by type:")
print(df_gold_anomalies["anomaly_type"].value_counts().to_string())
print("\nAnomaly breakdown by severity:")
print(df_gold_anomalies["severity"].value_counts().to_string())
print("\nAnomaly details:")
print(df_gold_anomalies[[
    "model","latency_ms","cost_usd","anomaly_type","severity"
]].to_string())

# Write Gold Table 3
gold_anomalies_path = f"s3://{S3_BUCKET}/gold/anomaly_summary"
write_deltalake(
    gold_anomalies_path,
    df_gold_anomalies,
    mode="overwrite",
    storage_options=storage_options,
)
print(f"\nGold anomaly_summary written to S3")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MEDALLION ARCHITECTURE COMPLETE")
print("="*55)
print(f"Bronze: s3://{S3_BUCKET}/bronze/llm_traces")
print(f"Silver: s3://{S3_BUCKET}/silver/llm_traces")
print(f"Gold 1: s3://{S3_BUCKET}/gold/hourly_metrics")
print(f"Gold 2: s3://{S3_BUCKET}/gold/model_comparison")
print(f"Gold 3: s3://{S3_BUCKET}/gold/anomaly_summary")
print("="*55)
print(f"\nTotal pipeline: {len(df_silver)} raw traces")
print(f"→ {len(df_gold_hourly)} hourly metric rows")
print(f"→ {len(df_gold_models)} model comparison rows")
print(f"→ {len(df_gold_anomalies)} anomalies identified")

# COMMAND ----------

# Cell 6: Phase 4 — Anomaly Detection with Isolation Forest
#
# Concept: We read our Silver Delta table (which has cleaned,
# enriched trace data) and train a machine learning model to
# learn what "normal" LLM call behaviour looks like.
# Any call that doesn't fit the normal pattern gets flagged.

import subprocess
subprocess.run(["pip", "install", "scikit-learn", "mlflow", "matplotlib"],
               capture_output=True)

import boto3
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timezone
from deltalake import DeltaTable

# ── Credentials ──────────────────────────────────────────────
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# ── Load Silver table ─────────────────────────────────────────
silver_path = f"s3://{S3_BUCKET}/silver/llm_traces"
print("Loading Silver Delta table...")

dt_silver = DeltaTable(silver_path, storage_options=storage_options)
df = dt_silver.to_pandas()

print(f"Loaded {len(df)} rows")
print(f"Columns available: {list(df.columns)}")
print(f"\nSample of key columns:")
print(df[["model","latency_ms","cost_usd","total_tokens",
          "is_anomaly_rule"]].head(5).to_string())


# Cell 7: Feature engineering + train Isolation Forest
#
# Concept — Feature engineering:
# We choose WHICH columns to give the model.
# Good features are ones that actually describe LLM call behaviour.
# Bad features are random IDs, text strings, or columns that leak
# the answer (like is_anomaly_rule which we already computed).
#
# Concept — Isolation Forest:
# contamination=0.05 means "I expect roughly 5% of my data to be
# anomalous." Our producer injects 1 anomaly per 20 messages (5%)
# so this is accurate. The model uses this to calibrate its threshold.

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Step 1: Select features ───────────────────────────────────
# These 6 columns describe the "shape" of an LLM call.
# The model learns the normal shape and flags deviations.
FEATURES = [
    "latency_ms",      # how long the call took
    "cost_usd",        # how much it cost
    "total_tokens",    # total tokens consumed
    "prompt_tokens",   # how long the prompt was
    "token_ratio",     # prompt/total ratio (long prompts = high cost)
    "cost_per_token",  # efficiency metric
]

X = df[FEATURES].copy()

# Handle any null values — fill with column median
# Null values crash sklearn models, always handle them first
X = X.fillna(X.median())

print(f"Feature matrix shape: {X.shape}")
print(f"\nFeature statistics:")
print(X.describe().round(4).to_string())

# ── Step 2: Standardize features ─────────────────────────────
# Concept: StandardScaler converts each feature to have
# mean=0 and standard deviation=1.
# Why? latency_ms ranges from 200-20000. cost_usd ranges from
# 0.0001 to 0.5. Without scaling, the model pays too much
# attention to latency_ms just because its numbers are bigger.
# Scaling puts all features on equal footing.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter scaling (first 3 rows):")
print(pd.DataFrame(X_scaled, columns=FEATURES).head(3).round(4).to_string())

# ── Step 3: Train Isolation Forest ───────────────────────────
print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,    # number of trees in the forest
                         # more trees = more stable results
                         # 100 is the standard starting point

    contamination=0.05,  # expected fraction of anomalies
                         # we inject 1 per 20 = 5% so this is accurate
                         # the model uses this to set its internal threshold

    random_state=42,     # makes results reproducible
                         # same number = same model every time you train
                         # important for comparing experiments

    max_samples="auto",  # how many samples to use per tree
                         # auto = min(256, n_samples)
)

model.fit(X_scaled)
print("Model trained successfully!")

# ── Step 4: Generate predictions ─────────────────────────────
# predict() returns: +1 = normal, -1 = anomaly
# decision_function() returns the raw anomaly score
# More negative score = more anomalous

predictions = model.predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

# Convert to our convention: True = anomaly, False = normal
df["is_anomaly_ml"] = predictions == -1
df["anomaly_score"] = anomaly_scores.round(4)

# Add severity based on score
def get_severity(score):
    if score < -0.15:   return "CRITICAL"
    elif score < -0.10: return "HIGH"
    elif score < -0.05: return "MEDIUM"
    else:               return "NORMAL"

df["ml_severity"] = df["anomaly_score"].apply(get_severity)

# ── Step 5: Compare rule-based vs ML detection ───────────────
print("\n=== RESULTS ===")
print(f"\nRule-based anomalies (latency>5s OR cost>$0.10): "
      f"{df['is_anomaly_rule'].sum()}")
print(f"ML-based anomalies (Isolation Forest):           "
      f"{df['is_anomaly_ml'].sum()}")

print("\nML Severity breakdown:")
print(df["ml_severity"].value_counts().to_string())

# Show cases where ML and rules AGREE
both = df[df["is_anomaly_rule"] & df["is_anomaly_ml"]]
print(f"\nCases where BOTH methods agree: {len(both)}")

# Show cases where ML catches something rules MISSED
ml_only = df[df["is_anomaly_ml"] & ~df["is_anomaly_rule"]]
print(f"Cases ML caught that rules MISSED: {len(ml_only)}")
if len(ml_only) > 0:
    print("These are subtle anomalies rules can't detect:")
    print(ml_only[["model","latency_ms","cost_usd",
                   "anomaly_score","ml_severity"]].to_string())

# Show cases where rules fired but ML said NORMAL
rules_only = df[df["is_anomaly_rule"] & ~df["is_anomaly_ml"]]
print(f"Cases rules caught that ML said were normal: {len(rules_only)}")


# Cell 8: Log experiment to MLflow — fully self-contained
# Re-declares everything it needs so it works independently

import subprocess
subprocess.run(["pip", "install", "scikit-learn", "mlflow"], capture_output=True)

import mlflow
import mlflow.sklearn
import boto3
import pandas as pd
import numpy as np
import io
from deltalake import DeltaTable
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Credentials ──────────────────────────────────────────────
# ── Credentials ──────────────────────────────────────────────
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

# ── Re-declare features ───────────────────────────────────────
FEATURES = [
    "latency_ms",
    "cost_usd",
    "total_tokens",
    "prompt_tokens",
    "token_ratio",
    "cost_per_token",
]

# ── Re-load Silver data ───────────────────────────────────────
silver_path = f"s3://{S3_BUCKET}/silver/llm_traces"
print("Loading Silver Delta table...")
dt_silver = DeltaTable(silver_path, storage_options=storage_options)
df = dt_silver.to_pandas()
print(f"Loaded {len(df)} rows")

# ── Re-run feature engineering + training ────────────────────
X = df[FEATURES].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
model.fit(X_scaled)

predictions    = model.predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

df["is_anomaly_ml"] = predictions == -1
df["anomaly_score"]  = anomaly_scores.round(4)

def get_severity(score):
    if score < -0.15:   return "CRITICAL"
    elif score < -0.10: return "HIGH"
    elif score < -0.05: return "MEDIUM"
    else:               return "NORMAL"

df["ml_severity"] = df["anomaly_score"].apply(get_severity)

# ── MLflow logging ────────────────────────────────────────────
EXPERIMENT_NAME = "/Users/ameysagare190@gmail.com/llm-anomaly-detection"
mlflow.set_experiment(EXPERIMENT_NAME)

total_anomalies = int(df["is_anomaly_ml"].sum())
anomaly_rate    = round(total_anomalies / len(df) * 100, 2)
critical_count  = int((df["ml_severity"] == "CRITICAL").sum())
high_count      = int((df["ml_severity"] == "HIGH").sum())

with mlflow.start_run(run_name="isolation-forest-v1"):

    mlflow.log_param("model_type",    "IsolationForest")
    mlflow.log_param("n_estimators",  100)
    mlflow.log_param("contamination", 0.05)
    mlflow.log_param("random_state",  42)
    mlflow.log_param("n_features",    len(FEATURES))
    mlflow.log_param("features",      str(FEATURES))
    mlflow.log_param("training_rows", len(df))

    mlflow.log_metric("total_anomalies",  total_anomalies)
    mlflow.log_metric("anomaly_rate_pct", anomaly_rate)
    mlflow.log_metric("critical_count",   critical_count)
    mlflow.log_metric("high_count",       high_count)
    mlflow.log_metric("training_rows",    len(df))

    # input_example shows MLflow what the input data looks like
    # MLflow uses this to auto-generate the signature
    # signature = the contract: "this model expects these columns
    # with these data types and returns these outputs"
    input_example = pd.DataFrame(X_scaled[:5], columns=FEATURES)

    mlflow.sklearn.log_model(
        model,
        "isolation_forest",
        registered_model_name="LLMAnomalyDetector",
        input_example=input_example,
    )
    
    
    mlflow.sklearn.log_model(scaler, "feature_scaler")
    mlflow.log_dict({"features": FEATURES}, "features.json")

    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow run logged successfully")
    print(f"Run ID: {run_id}")
    print(f"\nParameters logged:")
    print(f"  n_estimators:  100")
    print(f"  contamination: 0.05")
    print(f"  features:      {FEATURES}")
    print(f"\nMetrics logged:")
    print(f"  total_anomalies:  {total_anomalies}")
    print(f"  anomaly_rate_pct: {anomaly_rate}%")
    print(f"  critical_count:   {critical_count}")
    print(f"  high_count:       {high_count}")
    print(f"\nModel registered as: LLMAnomalyDetector")
    print(f"View: left sidebar → Experiments → llm-anomaly-detection")


# Cell 9: Save ML predictions to Gold Delta table
# Fully self-contained — re-runs everything needed

import subprocess
subprocess.run(["pip", "install", "scikit-learn", "deltalake"], capture_output=True)

import boto3
import pandas as pd
import numpy as np
import io
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Credentials ──────────────────────────────────────────────
# ── Credentials ──────────────────────────────────────────────
AWS_ACCESS_KEY = "paste_your_access_key_id_here"
AWS_SECRET_KEY = "paste_your_secret_access_key_here"
AWS_REGION     = "ap-south-1"
S3_BUCKET      = "paste_your_actual_bucket_name_here"

storage_options = {
    "AWS_ACCESS_KEY_ID":          AWS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      AWS_SECRET_KEY,
    "AWS_REGION":                 AWS_REGION,
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

FEATURES = [
    "latency_ms", "cost_usd", "total_tokens",
    "prompt_tokens", "token_ratio", "cost_per_token",
]

# ── Re-load Silver ────────────────────────────────────────────
silver_path = f"s3://{S3_BUCKET}/silver/llm_traces"
print("Loading Silver Delta table...")
dt_silver = DeltaTable(silver_path, storage_options=storage_options)
df = dt_silver.to_pandas()
print(f"Loaded {len(df)} rows")

# ── Re-run training ───────────────────────────────────────────
X = df[FEATURES].copy().fillna(df[FEATURES].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
model.fit(X_scaled)

predictions    = model.predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

df["is_anomaly_ml"] = predictions == -1
df["anomaly_score"]  = anomaly_scores.round(4)

def get_severity(score):
    if score < -0.15:   return "CRITICAL"
    elif score < -0.10: return "HIGH"
    elif score < -0.05: return "MEDIUM"
    else:               return "NORMAL"

df["ml_severity"] = df["anomaly_score"].apply(get_severity)

# ── Build Gold ML table ───────────────────────────────────────
print("\nBuilding Gold ML anomaly scores table...")

df_gold_ml = df[[
    "trace_id",
    "model",
    "app_name",
    "session_id",
    "latency_ms",
    "cost_usd",
    "total_tokens",
    "prompt_tokens",
    "token_ratio",
    "cost_per_token",
    "is_error",
    "model_family",
    "latency_bucket",
    "is_anomaly_rule",
    "is_anomaly_ml",
    "anomaly_score",
    "ml_severity",
    "event_date",
    "event_hour",
]].copy()

# Convert any non-string timestamp columns to string for Delta compatibility
df_gold_ml["event_date"] = df_gold_ml["event_date"].astype(str)

# ── Write to S3 as Delta table ────────────────────────────────
gold_ml_path = f"s3://{S3_BUCKET}/gold/ml_anomaly_scores"
print(f"Writing to: {gold_ml_path}")

write_deltalake(
    gold_ml_path,
    df_gold_ml,
    mode="overwrite",
    storage_options=storage_options,
)

print("Gold ML table written successfully!")

# ── Verify ────────────────────────────────────────────────────
dt_check = DeltaTable(gold_ml_path, storage_options=storage_options)
df_check = dt_check.to_pandas()

print(f"\nVerification: {len(df_check)} rows confirmed")
print(f"Columns: {len(df_check.columns)}")

print("\nSeverity breakdown:")
print(df_check["ml_severity"].value_counts().to_string())

print("\nRule-based vs ML comparison:")
print(f"  Rule-based anomalies: {df_check['is_anomaly_rule'].sum()}")
print(f"  ML anomalies:         {df_check['is_anomaly_ml'].sum()}")

print("\nTop 5 most anomalous calls (lowest score = most suspicious):")
top5 = df_check.nsmallest(5, "anomaly_score")
print(top5[[
    "model", "latency_ms", "cost_usd",
    "anomaly_score", "ml_severity"
]].to_string())

print("\n" + "="*50)
print("PHASE 4 COMPLETE")
print("="*50)
print(f"Gold ML table: {gold_ml_path}")
print(f"Total traces scored: {len(df_check)}")
print(f"Anomalies detected: {df_check['is_anomaly_ml'].sum()}")
print(f"Anomaly rate: {df_check['is_anomaly_ml'].mean()*100:.1f}%")