"""Real concurrent load test against a running server: v0.2 continuous
batching vs. v0.1 single-request path.

Requires the server already running: `uvicorn server.main:app --port 8001`.
Hits one endpoint at a time (not both together -- they share one model on
one MPS device, running both concurrently would corrupt each other's state).

Usage:
    python benchmarks/loadtest_v0.2.py --endpoint generate_batched --concurrency 50
    python benchmarks/loadtest_v0.2.py --endpoint generate --concurrency 50
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone

import httpx

PROMPT = "Explain what a hash table is in two sentences."
MAX_NEW_TOKENS = 32


async def one_request(client: httpx.AsyncClient, url: str):
    t0 = time.perf_counter()
    r = await client.post(url, json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS}, timeout=180)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed, r.json()


async def run(endpoint: str, concurrency: int, base_url: str):
    url = f"{base_url}/{endpoint}"
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_request(client, url) for _ in range(concurrency)])
        wall_s = time.perf_counter() - t0

    latencies = sorted(e for e, _ in results)
    n = len(latencies)
    total_tokens = sum(r["new_tokens"] for _, r in results)

    def pct(p):
        idx = min(n - 1, int(p * n))
        return latencies[idx]

    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "wall_s": wall_s,
        "throughput_req_per_s": n / wall_s,
        "throughput_tokens_per_s": total_tokens / wall_s,
        "p50_s": pct(0.50),
        "p95_s": pct(0.95),
        "p99_s": pct(0.99),
        "min_s": latencies[0],
        "max_s": latencies[-1],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, choices=["generate", "generate_batched", "generate_paged"])
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = await run(args.endpoint, args.concurrency, args.base_url)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["max_new_tokens"] = MAX_NEW_TOKENS
    print(json.dumps(result, indent=2))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"v0.2_loadtest_{args.endpoint}_c{args.concurrency}.json",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
