"""
Phase-1 tests. Run with:   pytest -v

T2 (boundary) and T3 (negative) are the two tests that were still pending
in the Phase-1 report. The rest check the new forecasting code.
"""
import csv
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_metrics  # noqa: E402
import forecast  # noqa: E402
import traffic_generator as tg  # noqa: E402


# ------------------------------------------------------------------
# T2: boundary testing - replicas stay inside the configured limits
# ------------------------------------------------------------------

@pytest.mark.parametrize("rate", [0, 0.01, 1, 6.76, 50, 1e6])
def test_t2_replicas_stay_between_1_and_8(rate):
    assert 1 <= forecast.recommend_replicas(rate, 1, 8, 100) <= 8


def test_t2_zero_traffic_gives_minimum():
    assert forecast.recommend_replicas(0, 2, 8, 100) == 2


def test_t2_huge_traffic_gives_maximum():
    assert forecast.recommend_replicas(10_000, 1, 8, 100) == 8


def test_t2_exact_capacity_edges():
    # 100 requests/min is exactly one replica; one request more needs two
    assert forecast.recommend_replicas(100 / 60, 1, 8, 100) == 1
    assert forecast.recommend_replicas(101 / 60, 1, 8, 100) == 2


def test_t2_negative_rate_treated_as_zero():
    assert forecast.recommend_replicas(-5, 1, 8, 100) == 1


def test_t2_invalid_limits_rejected():
    with pytest.raises(ValueError):
        forecast.recommend_replicas(5, 8, 1, 100)
    with pytest.raises(ValueError):
        forecast.recommend_replicas(5, 1, 8, 0)


# ------------------------------------------------------------------
# T3: negative testing - Flask is down, the generator must not crash
# ------------------------------------------------------------------

def test_t3_generator_survives_server_down(tmp_path, monkeypatch):
    log = tmp_path / "scenario_log.csv"
    monkeypatch.setattr(tg, "TARGET_URL", "http://127.0.0.1:9/work")  # nothing listens here
    monkeypatch.setattr(tg, "LOG_FILE", str(log))

    assert tg.start_traffic("Low") is True
    time.sleep(3.3)
    assert tg.running is True            # still running, did not crash
    tg.stop_traffic()
    tg.worker.join(timeout=5)

    assert not tg.worker.is_alive()
    assert tg.running is False
    rows = list(csv.DictReader(open(log)))
    assert len(rows) >= 2
    assert sum(int(r["failed"]) for r in rows) > 0   # failures were counted
    assert sum(int(r["ok"]) for r in rows) == 0


def test_generator_rejects_unknown_scenario_and_double_start(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "TARGET_URL", "http://127.0.0.1:9/work")
    monkeypatch.setattr(tg, "LOG_FILE", str(tmp_path / "log.csv"))
    assert tg.start_traffic("Nope") is False
    assert tg.start_traffic("Low") is True
    assert tg.start_traffic("Low") is False   # already running
    tg.stop_traffic()
    tg.worker.join(timeout=5)


# ------------------------------------------------------------------
# Scenario shapes
# ------------------------------------------------------------------

def test_scenario_shapes():
    wave = tg.wave(base=8, amplitude=6, period=300)
    values = [wave(t) for t in range(0, 300)]
    assert min(values) == pytest.approx(2, abs=0.01) and max(values) == pytest.approx(14, abs=0.01)

    spike = tg.spike(base=4, peak=25, every=120, length=30)
    assert spike(10) == 4 and spike(100) == 25 and spike(130) == 4

    ramp = tg.ramp(low=2, high=25, up=150, down=150)
    assert ramp(0) == 2 and ramp(150) == 25 and ramp(300) == pytest.approx(2)


def test_nasa_replay_scaled_to_peak():
    rate = tg.nasa_replay(peak=25, seconds_per_bucket=1)
    values = [rate(t) for t in range(288)]
    assert max(values) == pytest.approx(25) and min(values) > 0


# ------------------------------------------------------------------
# Forecasting
# ------------------------------------------------------------------

def test_windows_alignment():
    X, y = forecast.make_windows(np.arange(10), lookback=3, horizon=2)
    assert X[0].tolist() == [0, 1, 2] and y[0] == 4
    assert len(X) == len(y) == 6


def test_linear_beats_naive_on_a_wave():
    rng = np.random.default_rng(0)
    t = np.arange(400)
    series = 10 + 5 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.3, 400)
    idx, predicted, actual = forecast.rolling_forecast(series, 12, 4, min_train=100, last=100)
    naive = series[idx - 4]
    assert forecast.rmse(actual, predicted) < 0.5 * forecast.rmse(actual, naive)


def test_forecast_never_uses_the_future():
    rng = np.random.default_rng(1)
    series = 10 + rng.normal(0, 1, 300)
    changed = series.copy()
    changed[200:] += 1000                       # alter the future
    idx1, pred1, _ = forecast.rolling_forecast(series, 12, 4, min_train=100)
    idx2, pred2, _ = forecast.rolling_forecast(changed, 12, 4, min_train=100)
    keep = idx1 <= 203                          # forecasts made before the change
    assert np.allclose(pred1[keep], pred2[keep])


def test_too_little_data_is_rejected():
    with pytest.raises(ValueError):
        forecast.LinearForecaster(12, 4).fit(np.arange(20.0))


# ------------------------------------------------------------------
# Prometheus export (against a fake Prometheus)
# ------------------------------------------------------------------

class FakePrometheus(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        start, end, step = float(q["start"][0]), float(q["end"][0]), float(q["step"][0])
        points, ts = [], start
        while ts <= end:
            points.append([ts, str(round(math.sin(ts / 100.0), 6))])
            ts += step
        body = {"status": "success", "data": {"resultType": "matrix",
                "result": [{"metric": {}, "values": points}]}}
        import json
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_export_in_chunks_has_no_gaps_or_duplicates(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakePrometheus)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(export_metrics, "MAX_POINTS", 50)   # force several chunks
    start, step, n = 1_700_000_000, 15, 300
    df = export_metrics.export(f"http://127.0.0.1:{server.server_port}",
                               start, start + n * step, step, {"request_rate": "x"})
    server.shutdown()
    assert len(df) == n + 1
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert (df.index[1:] - df.index[:-1]).unique().tolist() == [export_metrics.pd.Timedelta(seconds=step)]


# ------------------------------------------------------------------
# Dashboard smoke test (fake Prometheus answers, no real server needed)
# ------------------------------------------------------------------

def test_dashboard_renders_forecast_panel():
    pytest.importorskip("streamlit")
    from unittest import mock
    from streamlit.testing.v1 import AppTest

    class Reply:
        def __init__(self, body):
            self.body = body
        def raise_for_status(self):
            pass
        def json(self):
            return self.body

    def fake_get(url, params=None, timeout=None, **kwargs):
        if url.endswith("/api/v1/query_range"):
            end, step = float(params["end"]), float(params["step"])
            pts = [[end - (119 - i) * step, str(8 + 6 * math.sin(2 * math.pi * i / 20))]
                   for i in range(120)]
            return Reply({"data": {"result": [{"values": pts}]}})
        return Reply({"data": {"result": [{"value": [0, "5.0"]}]}})

    with mock.patch("requests.get", fake_get):
        at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=60).run()

    assert not at.exception
    assert any("Predictive Recommendation" in s.value for s in at.subheader)
    labels = [m.label for m in at.metric]
    assert "Forecast in 1 min" in labels and "Replicas for forecast" in labels


def test_quick_restart_keeps_new_run_alive(tmp_path, monkeypatch):
    """Stop then Start straight away: the old run must not switch off the new one."""
    monkeypatch.setattr(tg, "TARGET_URL", "http://127.0.0.1:9/work")
    monkeypatch.setattr(tg, "LOG_FILE", str(tmp_path / "log.csv"))
    assert tg.start_traffic("Low")
    first = tg.worker
    tg.stop_traffic()
    assert tg.start_traffic("Wave")      # before the first thread has finished
    time.sleep(0.8)                      # let the first thread finish
    assert tg.running is True
    assert tg.worker.is_alive()
    assert tg.current_scenario == "Wave"
    first.join(timeout=5)
    assert not first.is_alive()
    tg.stop_traffic()
    tg.worker.join(timeout=5)