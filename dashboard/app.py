"""
Streamlit Dashboard — LLM Observability Platform

Concept: Streamlit turns Python scripts into web UIs.
Every time you interact with the dashboard (click a button,
select a filter), Streamlit reruns the entire script from
top to bottom and updates the display.

Our dashboard has 4 sections:
1. Header + key metrics (total calls, anomalies, cost)
2. Charts (latency distribution, model comparison)
3. Anomaly feed (table of all anomalies)
4. AI explanation panel (click anomaly → see explanation)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page configuration ────────────────────────────────────────
# Must be the first Streamlit command
st.set_page_config(
    page_title="LLM Observability Platform",
    page_icon="🔍",
    layout="wide",        # use full browser width
    initial_sidebar_state="expanded",
)

# ── API base URL ──────────────────────────────────────────────
# Your FastAPI server runs here locally
API_URL = "http://localhost:8000"


# ── Helper: call API ──────────────────────────────────────────
def call_api(endpoint: str) -> dict:
    """
    Calls the FastAPI backend and returns JSON response.
    If API is down, shows a friendly error instead of crashing.
    """
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API. "
            "Make sure FastAPI is running: "
            "uvicorn api.main:app --reload"
        )
        return {}
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return {}


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔍 LLM Observability")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["Overview", "Anomalies", "Model Comparison"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Refresh button
    if st.button("Refresh Data"):
        st.rerun()

    st.markdown("---")
    st.caption("Built with Kafka · Databricks · Pinecone · Groq")


# ════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("LLM Observability Platform")
    st.caption("Real-time monitoring for your AI applications")

    # ── Metric cards ──────────────────────────────────────────
    # Load summary metrics from API
    summary = call_api("/metrics/summary")

    if summary:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="Total Calls",
                value=f"{summary.get('total_calls', 0):,}",
            )
        with col2:
            st.metric(
                label="Anomalies Detected",
                value=summary.get("total_anomalies", 0),
                delta=f"{summary.get('anomaly_rate_pct', 0)}% rate",
                delta_color="inverse",  # red = bad for anomalies
            )
        with col3:
            st.metric(
                label="Avg Latency",
                value=f"{summary.get('avg_latency_ms', 0):.0f}ms",
            )
        with col4:
            st.metric(
                label="Total Cost",
                value=f"${summary.get('total_cost_usd', 0):.2f}",
            )
        with col5:
            st.metric(
                label="Errors",
                value=summary.get("error_count", 0),
                delta_color="inverse",
            )

    st.markdown("---")

    # ── Charts row ────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Latency Distribution")

        latency_data = call_api("/latency/distribution")
        if latency_data and "distribution" in latency_data:
            dist = latency_data["distribution"]

            # Define order for latency buckets
            bucket_order = [
                "fast (<500ms)",
                "normal (500ms-1s)",
                "slow (1s-3s)",
                "very slow (3s-5s)",
                "anomaly (>5s)",
            ]

            # Build DataFrame in the right order
            df_dist = pd.DataFrame([
                {"bucket": b, "count": dist.get(b, 0)}
                for b in bucket_order
                if b in dist
            ])

            # Color map — green for fast, red for anomaly
            color_map = {
                "fast (<500ms)":       "#1D9E75",
                "normal (500ms-1s)":   "#378ADD",
                "slow (1s-3s)":        "#EF9F27",
                "very slow (3s-5s)":   "#D85A30",
                "anomaly (>5s)":       "#E24B4A",
            }

            fig = px.bar(
                df_dist,
                x="bucket",
                y="count",
                color="bucket",
                color_discrete_map=color_map,
                labels={"bucket": "Latency", "count": "Calls"},
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Severity Breakdown")

        if summary and "severity_breakdown" in summary:
            severity = summary["severity_breakdown"]

            if severity:
                color_map_sev = {
                    "CRITICAL": "#E24B4A",
                    "HIGH":     "#D85A30",
                    "MEDIUM":   "#EF9F27",
                    "NORMAL":   "#1D9E75",
                }

                fig_sev = px.pie(
                    names=list(severity.keys()),
                    values=list(severity.values()),
                    color=list(severity.keys()),
                    color_discrete_map=color_map_sev,
                    hole=0.4,
                )
                fig_sev.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig_sev, use_container_width=True)
            else:
                st.info("No anomalies detected yet")


# ════════════════════════════════════════════════════════════
# PAGE 2: ANOMALIES
# ════════════════════════════════════════════════════════════
elif page == "Anomalies":
    st.title("Anomaly Feed")
    st.caption("All detected anomalies, ranked by severity")

    # Severity filter
    severity_filter = st.selectbox(
        "Filter by severity",
        ["All", "HIGH", "MEDIUM", "NORMAL"],
    )

    # Load anomalies
    endpoint = "/anomalies"
    if severity_filter != "All":
        endpoint += f"?severity={severity_filter}"

    data = call_api(endpoint)

    if data and "anomalies" in data:
        anomalies = data["anomalies"]
        st.caption(f"Showing {len(anomalies)} anomalies")

        if anomalies:
            # Convert to DataFrame for display
            df_display = pd.DataFrame(anomalies)

            # Add severity color indicator
            def severity_badge(sev):
                colors = {
                    "CRITICAL": "🔴",
                    "HIGH":     "🟠",
                    "MEDIUM":   "🟡",
                    "NORMAL":   "🟢",
                }
                return colors.get(sev, "⚪") + " " + sev

            df_display["severity_display"] = \
                df_display["severity"].apply(severity_badge)

            # Display table
            st.dataframe(
                df_display[[
                    "severity_display",
                    "model",
                    "app_name",
                    "latency_ms",
                    "cost_usd",
                    "total_tokens",
                    "anomaly_score",
                    "trace_id",
                ]].rename(columns={
                    "severity_display": "Severity",
                    "model":            "Model",
                    "app_name":         "App",
                    "latency_ms":       "Latency (ms)",
                    "cost_usd":         "Cost ($)",
                    "total_tokens":     "Tokens",
                    "anomaly_score":    "Anomaly Score",
                    "trace_id":         "Trace ID",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # ── AI Explanation panel ──────────────────────────
            st.markdown("---")
            st.subheader("AI-Powered Explanation")
            st.caption(
                "Select a trace ID to get an automatic "
                "AI explanation of why it was flagged"
            )

            # Dropdown to pick which anomaly to explain
            trace_ids = [a["trace_id"] for a in anomalies]
            selected_trace = st.selectbox(
                "Select trace to explain",
                trace_ids,
                format_func=lambda x: (
                    f"{x[:8]}... | "
                    f"{next((a['model'] for a in anomalies if a['trace_id']==x), '')} | "
                    f"{next((a['latency_ms'] for a in anomalies if a['trace_id']==x), '')}ms"
                )
            )

            if st.button("Get AI Explanation", type="primary"):
                with st.spinner(
                    "Searching similar incidents and "
                    "generating explanation..."
                ):
                    explanation_data = call_api(
                        f"/anomalies/{selected_trace}/explain"
                    )

                if explanation_data and "explanation" in explanation_data:
                    anomaly_info = explanation_data.get("anomaly", {})

                    # Show anomaly details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Model",
                            anomaly_info.get("model", "")
                        )
                    with col2:
                        st.metric(
                            "Latency",
                            f"{anomaly_info.get('latency_ms',0)}ms"
                        )
                    with col3:
                        st.metric(
                            "Cost",
                            f"${anomaly_info.get('cost_usd',0):.4f}"
                        )

                    # Show AI explanation
                    st.markdown("#### AI Explanation")
                    st.info(explanation_data["explanation"])
        else:
            st.info("No anomalies found for selected filter")


# ════════════════════════════════════════════════════════════
# PAGE 3: MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Performance Leaderboard")
    st.caption("Compare latency, cost, and reliability across models")

    data = call_api("/models")

    if data and "models" in data:
        models = data["models"]

        if models:
            df_models = pd.DataFrame(models)

            # ── Latency comparison bar chart ──────────────────
            st.subheader("Average Latency by Model")
            fig_lat = px.bar(
                df_models,
                x="model",
                y="avg_latency_ms",
                color="model_family",
                color_discrete_map={
                    "openai":    "#378ADD",
                    "anthropic": "#D85A30",
                },
                labels={
                    "model":          "Model",
                    "avg_latency_ms": "Avg Latency (ms)",
                    "model_family":   "Provider",
                },
                text="avg_latency_ms",
            )
            fig_lat.update_traces(
                texttemplate="%{text:.0f}ms",
                textposition="outside"
            )
            fig_lat.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

            # ── Cost comparison ───────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Cost per Call by Model")
                fig_cost = px.bar(
                    df_models,
                    x="model",
                    y="avg_cost_per_call",
                    color="model_family",
                    color_discrete_map={
                        "openai":    "#378ADD",
                        "anthropic": "#D85A30",
                    },
                    labels={
                        "avg_cost_per_call": "Avg Cost ($)",
                        "model":             "Model",
                    },
                )
                fig_cost.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig_cost, use_container_width=True)

            with col2:
                st.subheader("Anomaly Rate by Model")
                fig_anom = px.bar(
                    df_models,
                    x="model",
                    y="anomaly_rate_pct",
                    color="model_family",
                    color_discrete_map={
                        "openai":    "#378ADD",
                        "anthropic": "#D85A30",
                    },
                    labels={
                        "anomaly_rate_pct": "Anomaly Rate (%)",
                        "model":            "Model",
                    },
                )
                fig_anom.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig_anom, use_container_width=True)

            # ── Full comparison table ─────────────────────────
            st.subheader("Full Comparison Table")
            st.dataframe(
                df_models[[
                    "model",
                    "total_calls",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "avg_cost_per_call",
                    "total_cost_usd",
                    "anomaly_rate_pct",
                ]].rename(columns={
                    "model":             "Model",
                    "total_calls":       "Total Calls",
                    "avg_latency_ms":    "Avg Latency (ms)",
                    "p95_latency_ms":    "P95 Latency (ms)",
                    "avg_cost_per_call": "Avg Cost ($)",
                    "total_cost_usd":    "Total Cost ($)",
                    "anomaly_rate_pct":  "Anomaly Rate (%)",
                }),
                use_container_width=True,
                hide_index=True,
            )