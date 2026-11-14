"""
Pull workload metrics from Prometheus into a CSV for forecasting.

Example:
    python export_metrics.py --minutes 60 --step 15 --out workload.csv

The CSV has one row per time step: timestamp, request_rate, cpu_percent,
memory_bytes. Feed it to compare_models.py.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

QUERIES = {
    "request_rate": "sum(rate(flask_http_request_duration_seconds_count[1m]))",
    "cpu_percent": "flask_process_cpu_percent",
    "memory_bytes": "flask_process_memory_bytes",
}

# Prometheus refuses more than 11,000 points per series in one query,
# so long ranges are fetched in chunks.
MAX_POINTS = 10000


def query_range(prom_url, query, start, end, step):
    """Fetch one query over [start, end] (unix seconds) in chunks."""
    chunk = MAX_POINTS * step
    rows = []
    t = start
    while t <= end:
        stop = min(t + chunk, end)
        response = requests.get(
            f"{prom_url}/api/v1/query_range",
            params={"query": query, "start": t, "end": stop, "step": step},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {body}")
        result = body["data"]["result"]
        if result:
            rows += [(float(ts), float(v)) for ts, v in result[0]["values"]]
        t = stop + step
    series = pd.Series(
        [v for _, v in rows],
        index=pd.to_datetime([ts for ts, _ in rows], unit="s", utc=True),
    )
    return series[~series.index.duplicated()]


def export(prom_url, start, end, step, queries=None):
    queries = queries or QUERIES
    table = {name: query_range(prom_url, q, start, end, step) for name, q in queries.items()}
    df = pd.DataFrame(table)
    df.index.name = "timestamp"
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prom", default="http://localhost:9090")
    p.add_argument("--minutes", type=float, default=60, help="how far back to export (default 60)")
    p.add_argument("--step", type=int, default=15, help="seconds between points (default 15)")
    p.add_argument("--out", default="workload.csv")
    args = p.parse_args(argv)

    end = time.time()
    start = end - args.minutes * 60
    try:
        df = export(args.prom, start, end, args.step)
    except requests.RequestException as error:
        sys.exit(f"Could not reach Prometheus at {args.prom}: {error}")

    df.to_csv(args.out)
    first = datetime.fromtimestamp(start, timezone.utc).strftime("%H:%M:%S")
    print(f"Saved {len(df)} rows to {args.out} (from {first} UTC, step {args.step}s)")
    print(df.describe().loc[["count", "mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
