"""Real measured baseline benchmark for v0.1 (no batching).

Runs single-request generation repeatedly, records latency, tokens/sec,
and process memory (RSS). Writes raw JSON to benchmarks/results/ -- this is
the only source of truth for numbers quoted in the README.
"""
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.model import Model

PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Explain the difference between a process and a thread in operating systems, covering memory isolation and scheduling.",
    "long": "Write a detailed step-by-step explanation of how a B-tree index in a relational database speeds up range queries, including how it differs from a hash index and why databases choose one over the other for different workloads.",
}

RUNS_PER_PROMPT = 5
MAX_NEW_TOKENS = 128


def rss_mb():
    # ru_maxrss is bytes on macOS, KB on Linux
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val / (1024 * 1024) if sys.platform == "darwin" else val / 1024


def main():
    model = Model()
    results = {
        "milestone": "v0.1-baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model.model_name,
        "device": model.device,
        "load_time_s": model.load_time_s,
        "max_new_tokens": MAX_NEW_TOKENS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "prompts": {},
    }

    for label, prompt in PROMPTS.items():
        runs = []
        for i in range(RUNS_PER_PROMPT):
            r = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
            runs.append(
                {
                    "elapsed_s": r["elapsed_s"],
                    "new_tokens": r["new_tokens"],
                    "tokens_per_sec": r["tokens_per_sec"],
                }
            )
            print(f"[{label}] run {i+1}/{RUNS_PER_PROMPT}: "
                  f"{r['tokens_per_sec']:.2f} tok/s, {r['elapsed_s']:.3f}s")

        latencies = sorted(r["elapsed_s"] for r in runs)
        tps = [r["tokens_per_sec"] for r in runs]
        results["prompts"][label] = {
            "prompt_len_chars": len(prompt),
            "runs": runs,
            "mean_tokens_per_sec": sum(tps) / len(tps),
            "min_latency_s": latencies[0],
            "max_latency_s": latencies[-1],
            "median_latency_s": latencies[len(latencies) // 2],
        }

    results["peak_rss_mb"] = rss_mb()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v0.1_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Peak RSS: {results['peak_rss_mb']:.1f} MB")


if __name__ == "__main__":
    main()
