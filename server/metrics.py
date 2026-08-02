"""v0.7: Prometheus metrics for the paged/routed serving path."""
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "inference_requests_total", "Total generate requests", ["endpoint", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", "End-to-end request latency", ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
    registry=REGISTRY,
)
TTFT = Histogram(
    "inference_ttft_seconds", "Time to first token", ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=REGISTRY,
)
ACTIVE_REQUESTS = Gauge(
    "inference_active_requests", "Currently in-flight requests", ["endpoint"],
    registry=REGISTRY,
)
POOL_FREE_BLOCKS = Gauge(
    "inference_kv_pool_free_blocks", "Free KV blocks remaining", ["engine"],
    registry=REGISTRY,
)


def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)
