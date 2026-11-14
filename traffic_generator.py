"""
Traffic generator with varying load scenarios.

Differences from the first version:
  * Open-loop: requests are sent at a target rate (requests per second)
    no matter how slowly the server answers. The old version sent one
    request at a time, so its real rate was limited by the response time.
  * Scenarios change the target rate over time (wave, spike, ramp, NASA
    replay) so a forecaster has a pattern to learn.
  * Every second the target rate and the number of requests sent are
    written to scenario_log.csv, so each run has known ground truth.

app.py keeps working unchanged: it uses start_traffic(name),
stop_traffic() and the `running` flag.
"""
import csv
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

TARGET_URL = "http://localhost:5001/work"
LOG_FILE = "scenario_log.csv"
NASA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasa_replay.csv")

TICK = 0.1          # seconds between dispatch rounds
MAX_WORKERS = 64    # most requests in flight at once

# state read by app.py
running = False
worker = None
current_scenario = None
current_target_rps = 0.0
_stop = threading.Event()   # stop signal of the current run; every run has its own

# ------------------------------------------------------------------
# Scenario shapes: each function takes elapsed seconds, returns req/s
# ------------------------------------------------------------------

def steady(rate):
    return lambda t: rate


def wave(base=8.0, amplitude=6.0, period=300.0):
    """Smooth up and down, like a compressed daily cycle."""
    return lambda t: base + amplitude * math.sin(2 * math.pi * t / period)


def spike(base=4.0, peak=25.0, every=120.0, length=30.0):
    """Quiet load with a sudden burst at the end of every cycle."""
    return lambda t: peak if (t % every) >= (every - length) else base


def ramp(low=2.0, high=25.0, up=150.0, down=150.0):
    """Climb steadily to the peak, then fall back."""
    def rate(t):
        p = t % (up + down)
        if p < up:
            return low + (high - low) * p / up
        return high - (high - low) * (p - up) / down
    return rate


def nasa_replay(peak=25.0, seconds_per_bucket=2.0):
    """Replay one real NASA-HTTP day (5-minute buckets) scaled to `peak` req/s."""
    with open(NASA_FILE) as f:
        counts = [float(row["requests"]) for row in csv.DictReader(f)]
    scale = peak / max(counts)

    def rate(t):
        return counts[int(t / seconds_per_bucket) % len(counts)] * scale
    return rate


# Old three levels are kept, as constant rates.
SCENARIOS = {
    "Low": lambda: steady(2.0),
    "Medium": lambda: steady(8.0),
    "High": lambda: steady(20.0),
    "Wave": wave,
    "Spike": spike,
    "Ramp": ramp,
    "NASA replay": nasa_replay,
}

# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

def _send(counters, lock):
    try:
        requests.get(TARGET_URL, timeout=5)
        with lock:
            counters["ok"] += 1
    except requests.RequestException:
        with lock:
            counters["failed"] += 1


def generate_traffic(level="Medium", stop=None):
    global running, current_scenario, current_target_rps

    stop = stop or _stop

    rate_fn = SCENARIOS[level]()
    current_scenario = level
    counters = {"ok": 0, "failed": 0}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    new_file = not os.path.exists(LOG_FILE)
    log = open(LOG_FILE, "a", newline="")
    writer = csv.writer(log)
    if new_file:
        writer.writerow(["timestamp", "scenario", "target_rps", "ok", "failed"])

    start = time.monotonic()
    next_tick = start
    next_log = start + 1.0
    credit = 0.0
    sent_at_last_log = (0, 0)

    try:
        while not stop.is_set():
            now = time.monotonic()
            t = now - start
            current_target_rps = max(0.0, rate_fn(t))

            credit += current_target_rps * TICK
            n = int(credit)
            credit -= n
            for _ in range(n):
                try:
                    pool.submit(_send, counters, lock)
                except RuntimeError:     # pool closed, the program is exiting
                    stop.set()
                    break

            if now >= next_log:
                with lock:
                    ok, failed = counters["ok"], counters["failed"]
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"), level,
                    round(current_target_rps, 2),
                    ok - sent_at_last_log[0], failed - sent_at_last_log[1],
                ])
                log.flush()
                sent_at_last_log = (ok, failed)
                next_log += 1.0

            next_tick += TICK
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        pool.shutdown(wait=False)
        log.close()
        if stop is _stop:        # only the current run may reset the shared state
            running = False
            current_target_rps = 0.0


def start_traffic(level="Medium"):
    global running, worker, _stop

    if running:
        return False
    if level not in SCENARIOS:
        print(f"Unknown scenario: {level}")
        return False
    try:
        SCENARIOS[level]()          # fail early, e.g. missing NASA file
    except (OSError, KeyError, ValueError) as error:
        print(f"Cannot start {level}: {error}")
        return False

    _stop = threading.Event()
    running = True
    worker = threading.Thread(target=generate_traffic, args=(level, _stop), daemon=True)
    worker.start()
    print(f"{level} traffic started")
    return True


def stop_traffic():
    global running
    _stop.set()
    running = False
    print("Traffic stopped")