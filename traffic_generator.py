
import requests
import time
import threading
import random

TARGET_URL = "http://localhost:5001/work"

running = False
worker = None

TRAFFIC_LEVELS = {
    "Low": (1, 2),
    "Medium": (5, 10),
    "High": (15, 25)
}

def generate_traffic(level="Medium"):
    global running

    min_delay, max_delay = TRAFFIC_LEVELS[level]

    while running:
        try:
            requests.get(TARGET_URL, timeout=5)
        except requests.RequestException as error:
            print("Request failed:", error)

        time.sleep(random.uniform(
            1 / max_delay,
            1 / min_delay
        ))

def start_traffic(level="Medium"):
    global running, worker

    if running:
        return

    running = True

    worker = threading.Thread(
        target=generate_traffic,
        args=(level,),
        daemon=True
    )

    worker.start()
    print(f"{level} traffic started")

def stop_traffic():
    global running

    running = False
    print("Traffic stopped")
