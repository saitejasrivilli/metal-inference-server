# Changelog

## v0.1-baseline
Single-request FastAPI server on torch MPS, Qwen2.5-0.5B-Instruct, no
batching. Establishes the control-group benchmark (~36-39 tok/s single
request, 1344MB peak RSS) that every later milestone is measured against.
See `benchmarks/results/v0.1_baseline.json`.

## v0.2-batching
Continuous batching engine (`server/engine.py`) — dynamic queue, per-step
admit/evict, no KV-cache reuse yet (recomputes full sequence each step).
Measured ~1.5-1.6x throughput over v0.1 baseline at concurrency 20-50, and
holds up cleanly through concurrency 200 at `max_batch_size=4`.
Honest negative result: `max_batch_size=16` reliably wedged the MPS backend
under load (uninterruptible process, needed `kill -9`); reduced to 4.
See `benchmarks/results/v0.2_*.json`.

## v0.3-paged-kv
Real paged KV cache (`server/paged_engine.py`) — fixed block pool, actual
cross-step reuse, verified token-for-token correct against a hand-rolled
greedy reference (32/32 match). Honest results: throughput at c=20/50 is
*slower* than v0.2's naive recompute batching (42.0/41.4 vs 48.0/45.9 tok/s)
due to Python-loop pool writes and no fused MPS paged-attention kernel;
memory capacity is the real win (~12 concurrent short replies from a budget
naive prealloc would cap at 4 — see `benchmarks/results/v0.3_memory_analysis.json`).
Also fixed a real bug: mid-decode pool exhaustion silently hung the entire
engine; now rejects the specific request(s) with a clean error instead.
See `benchmarks/results/v0.3_*.json` and `docs/DESIGN_DECISIONS.md`.

## v0.4-quantization
Weight-only Q8_0 (int8) / Q4_0 (packed int4) quantization
(`server/quantization.py`) applied directly to the loaded PyTorch model —
scoped away from GGUF/llama.cpp since that runtime doesn't expose the
`DynamicCache` API v0.2/v0.3's engines depend on (see DESIGN_DECISIONS.md).
Real 2.0x/4.0x memory compression, but 5.2x/9.1x *slower* single-request
throughput -- no fused low-bit matmul kernel on MPS, so every forward pass
dequantizes to fp16 first. Full ablation (batching -> +paged KV ->
+quantization) at c=20: 33.0 -> 48.0 -> 42.0 -> 17.1 tok/s. Biggest honest
negative result in the project so far. See `benchmarks/results/v0.4_quantization.json`.
