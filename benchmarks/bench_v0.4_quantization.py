"""Real measured quantization results: memory footprint + single-request
throughput for fp16 baseline vs Q8_0-style (int8) vs Q4_0-style (packed
int4) weight-only quantization, plus one concurrent-load ablation point
(quantized weights running through the v0.3 paged engine) to complete the
batching -> +paged KV -> +quantization isolation table.
"""
import asyncio
import json
import os
import time
from datetime import datetime, timezone

from server.model import Model
from server.quantization import quantize_model_
from server.paged_engine import PagedKVCacheEngine

PROMPT = "Explain what a hash table is in one sentence."
MAX_NEW_TOKENS = 32


def bench_single(bits):
    m = Model()
    replaced = orig_b = quant_b = None
    if bits is not None:
        replaced, orig_b, quant_b = quantize_model_(m.model, bits=bits)
    runs = [m.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS) for _ in range(3)]
    tps = sum(r["tokens_per_sec"] for r in runs) / len(runs)
    return {
        "bits": bits,
        "num_linear_replaced": replaced,
        "orig_weight_mb": orig_b / 1e6 if orig_b else None,
        "quant_weight_mb": quant_b / 1e6 if quant_b else None,
        "compression_ratio": (orig_b / quant_b) if quant_b else 1.0,
        "mean_tokens_per_sec": tps,
        "sample_text": runs[0]["text"],
    }, m


async def bench_paged_concurrent(model_obj, concurrency=20):
    eng = PagedKVCacheEngine(model_obj.model, model_obj.tokenizer, model_obj.device,
                              num_blocks=256, max_batch_size=4)
    eng.start()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[eng.submit(PROMPT, MAX_NEW_TOKENS) for _ in range(concurrency)])
    wall = time.perf_counter() - t0
    total_tokens = sum(r["new_tokens"] for r in results)
    return {"concurrency": concurrency, "wall_s": wall, "throughput_tokens_per_s": total_tokens / wall}


def main():
    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "max_new_tokens": MAX_NEW_TOKENS}

    fp16_single, fp16_model = bench_single(bits=None)
    results["fp16_single_request"] = fp16_single

    q8_single, q8_model = bench_single(bits=8)
    results["q8_0_single_request"] = q8_single

    q4_single, q4_model = bench_single(bits=4)
    results["q4_0_single_request"] = q4_single

    q8_paged = asyncio.run(bench_paged_concurrent(q8_model, concurrency=20))
    results["q8_0_paged_concurrency_20"] = q8_paged

    # ablation: throughput at c=20 for each lever, pulled from this repo's
    # own recorded results (v0.2/v0.3) plus this run's quantized number
    results["ablation_c20_tokens_per_sec"] = {
        "single_request_no_batching_v0.1_v0.2baseline": 33.0,
        "plus_continuous_batching_v0.2": 48.0,
        "plus_paged_kv_v0.3": 42.0,
        "plus_quantization_q8_0_v0.4": q8_paged["throughput_tokens_per_s"],
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v0.4_quantization.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
