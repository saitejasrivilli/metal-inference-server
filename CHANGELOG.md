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

## v0.9-streaming
SSE streaming (`POST /generate_paged_stream`) -- tokens yielded as SSE
`data:` events the moment they're decoded, verified not buffered (real
~100-150ms gaps between events). `/generate_paged` kept unchanged as the
comparison baseline. Cancellation took three attempts before it actually
worked: racing a fresh `is_disconnected()` per token failed (undetected
10+s), a persistent watcher task (v0.5's exact working pattern) also
failed (confirmed via a minimal no-engine repro that this Starlette/uvicorn
version's `is_disconnected()` never fires against an active streaming
response as a separately-raced task) -- what worked was awaiting it
inline in the per-token loop, verified freeing blocks within ~0.85s across
repeated real curl kills. Verified composition with v0.2 (streaming +
non-streaming batch together, not serialized) and v0.5 (cancel-mid-stream
frees blocks) -- 11/11 tests passing. Benchmarks: TTFT improvement over
non-streaming shrinks from 41.2% (c=10) to 9.5% (c=50) as admission
queueing dominates; throughput roughly a wash, streaming trailing slightly
at c=20/50 (37.3-37.5 vs 41.0-41.5 tok/s), plausibly from per-token SSE
framing overhead. See `benchmarks/results/v0.9_*.json`.

## v0.8-vllm-comparison
vLLM comparison, as mandated -- but vLLM could not be run on this machine
(`pip install vllm` fails to build here, pins torch==2.6.0; vLLM also has
no Apple Silicon/MPS support and no CUDA GPU was accessible in this
environment). Rather than skip it or invent a number, cites a real public
third-party benchmark (Qwen2.5-3B-Instruct on an A5000, 45.9 vs 2714.88
tok/s at c=50) with every confound stated explicitly (6x model size,
dedicated vs. integrated GPU, fused vs. non-fused kernels, output length)
so the ~59x gap isn't presented as apples-to-apples. Full breakdown in
`benchmarks/results/v0.8_vllm_comparison.json`. Final README pass: TL;DR
block, updated architecture diagram, expanded limitations section. This
closes out the v0.1-v0.8 build.

## v0.7-observability
Prometheus metrics (`GET /metrics`: request counts, latency/TTFT
histograms, active-request and KV-pool gauges), OpenTelemetry tracing
(console exporter, one span per `/generate_paged` call), and a burn-rate
SLO monitor (`server/slo_monitor.py`, `GET /alerts`). Alert verified to
actually fire in tests (3/3 passing,
`tests/test_v0_7_slo_monitor.py`): sustained breach fires, healthy
latencies never fire, and a single slow request among healthy ones does
NOT fire alone. Also fired for real against the live server
(`SLO_LATENCY_THRESHOLD_S=0.5`, two real ~1.8s requests -> alert with
`p99_latency_s: 1.824` within 3s).

## v0.6-router
Multi-model router (`server/router.py`) -- two genuinely different
architectures (Qwen2.5-0.5B, 24L/2KV-heads; SmolLM2-360M, 32L/5KV-heads)
behind one `/generate_multi` endpoint, isolated queues/pools/decode loops
per model. Verified concurrent isolation by firing simultaneous requests at
both and confirming each completed in ~its own solo time (1.72s / 0.39s),
not serialized. Combined weight footprint measured directly: 1711.7 MB for
both loaded at once. Honest finding: SmolLM2 returns near-empty output on
raw (non-chat-templated) prompts Qwen tolerates fine -- a model-sensitivity
property, reproduced outside the router, not a routing bug.
See `benchmarks/results/v0.6_router.json`.

## v0.5-cancellation
Client disconnects now actually free their in-flight KV blocks. Engine-side
plumbing (`cancelled`/`freed` flags on `PagedSlot`) was necessary but not
sufficient -- found by manually killing a real curl request that
FastAPI/Starlette doesn't cancel non-streaming handlers on disconnect at
all; the generation kept running and holding blocks with nothing detecting
it. Fixed by racing generation against a `request.is_disconnected()`
poller in `/generate_paged`. 5/5 tests passing in
`tests/test_v0_5_cancellation.py` against the real engine (no mocks):
normal completion, cancel-while-pending, cancel-mid-decode, pool-exhaustion
rejection, and mixed cancel+completion all fully reclaim the pool with no
leak or double-free.
