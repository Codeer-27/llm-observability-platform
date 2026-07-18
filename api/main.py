"""
FastAPI Backend — serves data from S3 Gold tables via REST API

Concept: REST API means the dashboard can request data
using standard HTTP calls (GET /anomalies, GET /metrics etc).
FastAPI handles routing, validation, and response formatting.

Endpoints we expose:
GET /health          — is the API running?
GET /metrics/summary — overall platform stats
GET /anomalies       — list of all detected anomalies
GET /anomalies/{id}  — single anomaly with AI explanation
GET /models          — model comparison leaderboard
"""
import os
import sys
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from deltalake import DeltaTable
from typing import Optional

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.append(str(Path(__file__).parent.parent))

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="LLM Observability Platform API",
    description="Real-time monitoring for LLM applications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Helper: get storage options fresh each call ───────────────
# Read inside function so environment variables are always
# available regardless of which process reads them
def get_storage_options() -> dict:
    return {
        "AWS_ACCESS_KEY_ID":          "YOUR_AWS_ACCESS_KEY_HERE",
        "AWS_SECRET_ACCESS_KEY":      "YOUR_AWS_SECRET_KEY_HERE",
        "AWS_REGION":                 "ap-south-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def get_bucket() -> str:
    return "llm-observability-amey01-063884340656-ap-south-1-an"
# def get_bucket() -> str:
#     return os.getenv("S3_BUCKET_NAME", "")


# ── Helper: load Delta table ──────────────────────────────────
def load_delta_table(path_suffix: str) -> pd.DataFrame:
    bucket = get_bucket()
    storage_options = get_storage_options()
    path = f"s3://{bucket}/{path_suffix}"
    try:
        dt = DeltaTable(path, storage_options=storage_options)
        return dt.to_pandas()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load {path}: {str(e)}"
        )





# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """
    Simple health check — returns OK if API is running.
    Used by monitoring systems to verify the service is alive.
    Load balancers and orchestrators ping this endpoint.
    """
    return {
        "status":  "healthy",
        "service": "LLM Observability API",
        "version": "1.0.0",
    }

@app.get("/debug")
def debug_env():
    return {
        "bucket":      os.getenv("S3_BUCKET_NAME"),
        "key_found":   os.getenv("AWS_ACCESS_KEY_ID") is not None,
        "key_prefix":  str(os.getenv("AWS_ACCESS_KEY_ID", ""))[:8],
        "secret_found": os.getenv("AWS_SECRET_ACCESS_KEY") is not None,
        "region":      os.getenv("AWS_REGION"),
        "storage_options": get_storage_options(),
    }


@app.get("/metrics/summary")
def get_summary_metrics():
    """
    Returns overall platform statistics.
    Powers the top metric cards in the dashboard.
    """
    df = load_delta_table("gold/ml_anomaly_scores")

    total_calls     = len(df)
    total_anomalies = int(df["is_anomaly_ml"].sum())
    anomaly_rate    = round(df["is_anomaly_ml"].mean() * 100, 2)
    avg_latency     = round(df["latency_ms"].mean(), 0)
    total_cost      = round(df["cost_usd"].sum(), 4)
    avg_cost        = round(df["cost_usd"].mean(), 6)
    error_count     = int(df["is_error"].sum())

    # Severity breakdown
    severity_counts = df[df["is_anomaly_ml"]]["ml_severity"] \
        .value_counts().to_dict()

    return {
        "total_calls":      total_calls,
        "total_anomalies":  total_anomalies,
        "anomaly_rate_pct": anomaly_rate,
        "avg_latency_ms":   avg_latency,
        "total_cost_usd":   total_cost,
        "avg_cost_usd":     avg_cost,
        "error_count":      error_count,
        "severity_breakdown": severity_counts,
    }


@app.get("/anomalies")
def get_anomalies(severity: Optional[str] = None):
    """
    Returns all detected anomalies.
    Optional filter: ?severity=HIGH or ?severity=CRITICAL
    Powers the anomaly feed table in the dashboard.
    """
    df = load_delta_table("gold/ml_anomaly_scores")

    # Filter only anomalies
    df_anomalies = df[df["is_anomaly_ml"] == True].copy()

    # Apply severity filter if provided
    if severity:
        df_anomalies = df_anomalies[
            df_anomalies["ml_severity"].str.upper() == severity.upper()
        ]

    # Sort by anomaly score (most anomalous first)
    df_anomalies = df_anomalies.sort_values("anomaly_score")

    # Convert to list of dicts for JSON response
    # JSON doesn't support numpy types so we convert to Python native
    result = []
    for _, row in df_anomalies.iterrows():
        result.append({
            "trace_id":      str(row["trace_id"]),
            "model":         str(row["model"]),
            "app_name":      str(row["app_name"]),
            "latency_ms":    int(row["latency_ms"]),
            "cost_usd":      float(row["cost_usd"]),
            "total_tokens":  int(row["total_tokens"]),
            "anomaly_score": float(row["anomaly_score"]),
            "severity":      str(row["ml_severity"]),
            "is_error":      bool(row["is_error"]),
            "latency_bucket":str(row["latency_bucket"]),
            "model_family":  str(row["model_family"]),
        })

    return {
        "total":     len(result),
        "anomalies": result,
    }


@app.get("/anomalies/{trace_id}/explain")
def explain_anomaly_endpoint(trace_id: str):
    """
    Returns AI-generated explanation for a specific anomaly.
    This calls the RAG pipeline — Pinecone + Groq.
    Powers the explanation panel when you click an anomaly.
    """
    df = load_delta_table("gold/ml_anomaly_scores")

    # Find the specific anomaly by trace_id
    row = df[df["trace_id"] == trace_id]
    if len(row) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Trace {trace_id} not found"
        )

    row = row.iloc[0]

    # Import and call the RAG explainer
    try:
        from ingestion.rag_explainer import explain_anomaly
        anomaly = {
            "model":        str(row["model"]),
            "app_name":     str(row["app_name"]),
            "latency_ms":   int(row["latency_ms"]),
            "cost_usd":     float(row["cost_usd"]),
            "total_tokens": int(row["total_tokens"]),
            "severity":     str(row["ml_severity"]),
        }
        explanation = explain_anomaly(anomaly)
        return {
            "trace_id":    trace_id,
            "anomaly":     anomaly,
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG explanation failed: {str(e)}"
        )


@app.get("/models")
def get_model_comparison():
    """
    Returns model performance leaderboard.
    Powers the model comparison chart in the dashboard.
    """
    df = load_delta_table("gold/model_comparison")

    result = []
    for _, row in df.iterrows():
        result.append({
            "model":              str(row["model"]),
            "model_family":       str(row["model_family"]),
            "total_calls":        int(row["total_calls"]),
            "avg_latency_ms":     float(row["avg_latency_ms"]),
            "p95_latency_ms":     float(row["p95_latency_ms"]),
            "avg_cost_per_call":  float(row["avg_cost_per_call"]),
            "total_cost_usd":     float(row["total_cost_usd"]),
            "anomaly_rate_pct":   float(row["anomaly_rate_pct"]),
        })

    return {
        "models": sorted(result, key=lambda x: x["avg_latency_ms"])
    }


@app.get("/latency/distribution")
def get_latency_distribution():
    """
    Returns latency bucket distribution.
    Powers the latency distribution chart.
    """
    df = load_delta_table("gold/ml_anomaly_scores")

    distribution = df["latency_bucket"].value_counts().to_dict()

    return {
        "distribution": distribution,
        "total_calls":  len(df),
    }