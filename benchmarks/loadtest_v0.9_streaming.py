"""v0.9: TTFT and throughput, streaming (/generate_paged_stream) vs
non-streaming (/generate_paged), at the same concurrency levels used
throughout this project. Requires the server already running.

TTFT for streaming = time from request start to the first SSE `data:`
event. TTFT for non-streaming = time from request start to the whole
response (there's no earlier signal possible -- that's the point being
measured: streaming's actual value is TTFT, not total completion time,
which shouldn't change and isn't claimed to).
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


async def one_streaming_request(client: httpx.AsyncClient, url: str):
    t0 = time.perf_counter()
    ttft = None
    new_tokens = 0
    async with client.stream("POST", url, json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS}, timeout=180) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            new_tokens += 1
    elapsed = time.perf_counter() - t0
    return elapsed, ttft, new_tokens


async def one_nonstreaming_request(client: httpx.AsyncClient, url: str):
    t0 = time.perf_counter()
    r = await client.post(url, json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS}, timeout=180)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    # non-streaming has no earlier signal than the full response -- TTFT
    # here is reported as the engine's own measured ttft_s for reference,
    # not something the HTTP layer can observe any sooner than "elapsed"
    return elapsed, body.get("ttft_s"), body["new_tokens"]


async def run(mode: str, concurrency: int, base_url: str):
    if mode == "stream":
        url = f"{base_url}/generate_paged_stream"
        fn = one_streaming_request
    else:
        url = f"{base_url}/generate_paged"
        fn = one_nonstreaming_request

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[fn(client, url) for _ in range(concurrency)])
        wall_s = time.perf_counter() - t0

    elapsed_list = sorted(e for e, _, _ in results)
    ttft_list = sorted(t for _, t, _ in results if t is not None)
    total_tokens = sum(n for _, _, n in results)
    n = len(elapsed_list)

    def pct(lst, p):
        if not lst:
            return None
        idx = min(len(lst) - 1, int(p * len(lst)))
        return lst[idx]

    return {
        "mode": mode,
        "concurrency": concurrency,
        "wall_s": wall_s,
        "throughput_tokens_per_s": total_tokens / wall_s,
        "ttft_p50_s": pct(ttft_list, 0.50),
        "ttft_p95_s": pct(ttft_list, 0.95),
        "ttft_p99_s": pct(ttft_list, 0.99),
        "elapsed_p50_s": pct(elapsed_list, 0.50),
        "elapsed_p95_s": pct(elapsed_list, 0.95),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["stream", "nonstream"])
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--base-url", default="http://localhost:8010")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = await run(args.mode, args.concurrency, args.base_url)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["max_new_tokens"] = MAX_NEW_TOKENS
    print(json.dumps(result, indent=2))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"v0.9_{args.mode}_c{args.concurrency}.json",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
