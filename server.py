#sample cloud-native application. It will receive HTTP requests and expose metrics for Prometheus.
from flask import Flask, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge
import time
import random
import psutil

app = Flask(__name__)

# Expose application metrics for Prometheus
metrics = PrometheusMetrics(app)
process_cpu = Gauge(
    "flask_process_cpu_percent",
    "CPU usage of the Flask application process"
)

process_memory = Gauge(
    "flask_process_memory_bytes",
    "Memory usage of the Flask application process"
)
process = psutil.Process()
def update_process_metrics():
    process_cpu.set(process.cpu_percent())
    process_memory.set(process.memory_info().rss)
@app.route("/")
def home():
    return jsonify({
        "application": "Predictive Autoscaling Demo",
        "status": "running"
    })

@app.route("/work")
def work():
    update_process_metrics()

    # Simulate a small amount of application processing
    time.sleep(random.uniform(0.02, 0.15))

    return jsonify({
        "status": "success",
        "message": "Workload processed"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
