
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import traffic_generator
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Autoscaling",
    page_icon="☁️",
    layout="wide"
)

PROMETHEUS_URL = "http://localhost:9090"

# --------------------------------------------------
# PROMETHEUS FUNCTIONS
# --------------------------------------------------

def get_metric(query):
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=3
        )
        response.raise_for_status()
        results = response.json()["data"]["result"]

        if results:
            return float(results[0]["value"][1])

    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None

    return None


def format_value(value, suffix="", multiplier=1):
    if value is None:
        return "No data"
    return f"{value * multiplier:,.2f}{suffix}"


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(" AI-Based Predictive Autoscaling")
st.caption("Cloud-Native Application Resource Optimization")

st.info(
    "Project prototype: live application monitoring, "
    "traffic generation, and baseline scaling recommendations."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Control Center")

st.sidebar.subheader("Traffic Generator")

traffic_level = st.sidebar.selectbox(
    "Traffic intensity",
    ["Low", "Medium", "High"]
)

start_col, stop_col = st.sidebar.columns(2)

with start_col:
    if st.button("▶ Start", use_container_width=True):
        traffic_generator.start_traffic(traffic_level)

with stop_col:
    if st.button("■ Stop", use_container_width=True):
        traffic_generator.stop_traffic()

if traffic_generator.running:
    st.sidebar.success("Traffic generator is running")
else:
    st.sidebar.warning("Traffic generator is stopped")

st.sidebar.divider()

st.sidebar.subheader("Scaling Configuration")

min_replicas = st.sidebar.number_input(
    "Minimum replicas",
    min_value=1,
    max_value=10,
    value=1
)

max_replicas = st.sidebar.number_input(
    "Maximum replicas",
    min_value=2,
    max_value=20,
    value=8
)

requests_per_replica = st.sidebar.number_input(
    "Requests per replica",
    min_value=10,
    max_value=500,
    value=100
)

if min_replicas >= max_replicas:
    st.sidebar.error("Maximum replicas must exceed minimum replicas.")

st.sidebar.divider()

if st.sidebar.button("Refresh metrics", use_container_width=True):
    st.rerun()

st.sidebar.caption(
    f"Last dashboard refresh: {datetime.now().strftime('%H:%M:%S')}"
)

# --------------------------------------------------
# FETCH LIVE METRICS
# --------------------------------------------------

request_rate = get_metric(
    'sum(rate(flask_http_request_duration_seconds_count[1m]))'
)

cpu_usage = get_metric(
    'flask_process_cpu_percent'
)

memory_usage = get_metric(
    'flask_process_memory_bytes'
)

latency = get_metric(
    '''
    sum(rate(flask_http_request_duration_seconds_sum{
        path="/work",status="200"
    }[5m]))
    /
    sum(rate(flask_http_request_duration_seconds_count{
        path="/work",status="200"
    }[5m]))
    '''
)

# --------------------------------------------------
# APPLICATION STATUS
# --------------------------------------------------

st.subheader(" Application Overview")

if request_rate is None and cpu_usage is None:
    st.error(
        "Unable to retrieve application metrics. "
        "Check that Flask and Prometheus are running."
    )
else:
    st.success("Connected to Prometheus")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Request Rate",
    format_value(request_rate, " req/s")
)

metric2.metric(
    "Application CPU",
    format_value(cpu_usage, "%")
)

metric3.metric(
    "Application Memory",
    format_value(memory_usage, " MiB", 1 / (1024 * 1024))
)

metric4.metric(
    "Average /work Latency",
    format_value(latency, " s")
)

# --------------------------------------------------
# TRAFFIC AND MONITORING
# --------------------------------------------------

st.divider()
st.subheader("Live Monitoring")

if "history" not in st.session_state:
    st.session_state.history = []

if request_rate is not None:
    st.session_state.history.append({
        "Time": datetime.now(),
        "Request Rate": request_rate,
        "CPU Usage": cpu_usage
    })

    st.session_state.history = st.session_state.history[-30:]

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_requests = go.Figure()
        fig_requests.add_trace(
            go.Scatter(
                x=history_df["Time"],
                y=history_df["Request Rate"],
                mode="lines+markers",
                name="Requests/sec"
            )
        )
        fig_requests.update_layout(
            title="HTTP Request Rate",
            xaxis_title="Time",
            yaxis_title="Requests per second",
            template="plotly_dark"
        )
        st.plotly_chart(fig_requests, use_container_width=True)

    with chart_col2:
        fig_cpu = go.Figure()
        fig_cpu.add_trace(
            go.Scatter(
                x=history_df["Time"],
                y=history_df["CPU Usage"],
                mode="lines+markers",
                name="CPU %"
            )
        )
        fig_cpu.update_layout(
            title="Application CPU Usage",
            xaxis_title="Time",
            yaxis_title="CPU (%)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
else:
    st.warning(
        "No monitoring history yet. Generate traffic and refresh metrics."
    )

# --------------------------------------------------
# BASELINE SCALING RECOMMENDATION
# --------------------------------------------------

st.divider()
st.subheader("🧠 Baseline Scaling Recommendation")

if request_rate is not None and min_replicas < max_replicas:
    estimated_demand = request_rate * 60

    recommended = max(
        min_replicas,
        min(
            int((estimated_demand / requests_per_replica) + 0.999),
            max_replicas
        )
    )

    result1, result2, result3 = st.columns(3)

    result1.metric(
        "Observed requests/min",
        f"{estimated_demand:.2f}"
    )

    result2.metric(
        "Estimated replicas",
        str(recommended)
    )

    result3.metric(
        "Configured replica range",
        f"{min_replicas}–{max_replicas}"
    )

    st.success(
        f"Baseline recommendation: {recommended} application replicas"
    )

    st.caption(
        "This is a simple estimate based on observed request rate, "
        "not a GRU prediction or an actual Kubernetes scaling action."
    )
else:
    st.warning(
        "Scaling recommendation unavailable. Check metrics and replica limits."
    )

# --------------------------------------------------
# PROJECT STATUS
# --------------------------------------------------

st.divider()
st.subheader("🚀 Project Implementation Status")

status_col1, status_col2, status_col3 = st.columns(3)

status_col1.success("Flask application and metrics")
status_col2.success("Prometheus and Grafana monitoring")
status_col3.success("Traffic generator module")

st.info(
    "Next implementation stages: GRU workload forecasting, "
    "Kubernetes autoscaling integration, and HPA comparison."
)
