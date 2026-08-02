# metal-inference-server

Continuous-batching LLM inference server built from scratch for Apple Silicon
(MPS/Metal) — because no existing engine (vLLM, SGLang) supports Apple GPUs
natively. Every number below comes from an actual run on this machine
(M-series Mac, macOS 15.5). No projected or estimated numbers.

**Status: v0.7 observability + SLO** — Prometheus metrics, OpenTelemetry
traces, and a burn-rate alert verified to actually fire (and not
false-positive) in real tests.

## Architecture (v0.1)

```
client --HTTP--> FastAPI (server/main.py) --> Model (server/model.py) --> torch MPS
```

One request in flight at a time. No queue, no batching, no KV cache reuse
across requests.

## How it works

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, loaded via HuggingFace `transformers`,
  weights in fp16, running on the `mps` device.
- `/generate` takes a prompt, runs greedy `model.generate()` synchronously,
  returns the completion plus timing.
- No batching, no scheduling — this milestone exists to establish the
  baseline every optimization after it is measured against.

## Benchmarks — v0.1 baseline

Run: `benchmarks/bench_v0.1_baseline.py`, raw output:
`benchmarks/results/v0.1_baseline.json` (5 runs per prompt, greedy decoding,
max_new_tokens=128, single request at a time, no batching).

| Prompt | Mean tok/s | Median latency (s) | Min / Max latency (s) |
|---|---|---|---|
| short ("What is the capital of France?") | 35.5 | 2.567 | 2.557 / 4.389 |
| medium (process vs thread explanation) | 37.2 | 3.334 | 3.258 / 3.824 |
| long (B-tree index explanation) | 36.5 | 3.499 | 3.363 / 3.725 |

- Model load time: ~48s (cold weight download/tokenizer init on first call
  in this run; not part of steady-state serving cost).
- Peak process RSS during benchmark: **1344 MB**.
- First run of each prompt is consistently slower (MPS kernel warmup /
  graph compile) — visible in the "short" prompt's first run at 22.8 tok/s
  vs. ~39 tok/s on subsequent runs. This warmup cost is real and will matter
  for cold-start latency later; not hidden.

## What's new in v0.2 — continuous batching

`server/engine.py`: a request queue + dynamic batch assembly + decode loop,
written from scratch (no off-the-shelf batching runtime targets MPS).
Every decode step, active requests are left-padded into one tensor and
advanced one token together; finished requests are evicted and new ones
admitted from the queue on the next step — batch composition changes every
step, which is what makes it "continuous" rather than fixed-batch.

**Honest limitation (by design, not a bug):** v0.2 has no KV-cache reuse
across steps yet. Each step recomputes the forward pass over every active
sequence's full token history so far. That's the real ceiling on this
milestone's throughput, not batch size — paged KV cache (v0.3) is what
actually fixes it.

**Honest negative result:** the first version of this milestone used
`max_batch_size=16`. At concurrency 50+ it reliably wedged the PyTorch MPS
backend — the process entered an uninterruptible-sleep state and stopped
responding to any request, including `/health`, and required `kill -9` to
recover. Reproduced twice. Root-caused to batch size, not concurrency:
dropping to `max_batch_size=4` (`MAX_BATCH_SIZE` env var) made every run at
concurrency 10/20/50/100/200 complete cleanly. This is left as-is rather
than silently tuned away — it's a real constraint of running large batched
tensor ops through the MPS backend under this workload, worth knowing before
trusting Metal for serving at scale.

### Benchmarks — batching (bs=4) vs. v0.1 single-request, same load

Same prompt/settings as v0.1 (greedy decode, 32 max new tokens), varying
concurrent requests. Raw JSON in `benchmarks/results/v0.2_*.json`.

| Concurrency | Single-request throughput (tok/s) | Batched (bs=4) throughput (tok/s) | Batched speedup |
|---|---|---|---|
| 10 | 27.5 | 5.2* | 0.19x* |
| 20 | 33.0 | 48.0 | 1.45x |
| 50 | 28.9 | 45.9 | 1.59x |
| 100 | n/a (not run) | 47.9 | — |
| 200 | n/a (not run) | 43.9 | — |

\* c=10 batched run measured immediately after server start and includes
MPS kernel warmup (matches the same warmup effect seen in v0.1); not
representative of steady state — see c=20/50 for the real comparison.

At concurrency 20-50, continuous batching gives a real ~1.5x throughput
improvement over one-request-at-a-time, using the same model, same
hardware, same decode settings. It also holds up cleanly to c=200, which
the single-request path was never going to do gracefully (unbounded queued
connections, not tested at that scale for that reason).

## What's new in v0.3 — paged KV cache

`server/paged_engine.py`: a fixed pool of fixed-size KV blocks
(block_size=16 tokens), shared across all sequences per layer. Each
sequence holds a logical block table instead of one growing tensor, and —
unlike v0.2 — the model only processes the *new* token each decode step;
past attention is served from the pool. Verified correct by comparing
token-for-token against a hand-rolled greedy reference loop: exact match,
32/32 tokens.

**Apple unified memory tradeoff, stated explicitly:** paged allocation
exists in vLLM/similar because on discrete-GPU hardware, VRAM is small and
separate from host RAM, so reserving `max_seq_len` per request wastes a
scarce, expensive tier. On Apple Silicon, CPU and GPU share one physical
memory pool — there's no separate scarce VRAM tier to protect. The benefit
paging still provides here is capacity efficiency under one shared budget:
at the same memory budget as a naive scheme that reserves
`max_batch_size=4 × max_seq_len=128` (6.29MB) up front, a paged pool with
the same footprint (32 blocks) can serve roughly **12 concurrent
short/medium replies** (~40 tokens each) instead of a hard cap of 4 — real
computed numbers in `benchmarks/results/v0.3_memory_analysis.json`, not
simulated.

**Honest negative result #1 — paged is slower, not faster, on this
workload:** at concurrency 20/50, paged KV throughput measured 42.0 / 41.4
tok/s vs. v0.2's naive recompute-batching at 48.0 / 45.9 tok/s. Root cause:
the per-token pool write (`_write_kv`) is a pure-Python loop over token
positions, and the gather-from-blocks step reconstructing a contiguous
batch tensor each decode step adds real overhead that a fused
paged-attention kernel (which MPS doesn't have) would eliminate. So the
architecturally "smarter" cache strategy loses to the "dumber" recompute
strategy at this scale, on this hardware, with this implementation. This is
the "friction" this milestone was expected to surface — not smoothed over.

**Honest negative result #2 — mid-decode pool exhaustion silently hung the
server:** the first version only checked pool capacity at admission time.
A sequence needs a *new* block at any later decode step too, and that path
had no capacity check. Reproduced reliably with `num_blocks=6` and 8
concurrent requests: the unhandled exception killed the background engine
task with no error surfaced, and every pending request hung forever (not
even `/health` responded). Fixed by wrapping the per-step allocation in
try/except and rejecting only the specific request(s) that fail with a
real 503 error, rather than crashing the shared engine loop. Verified:
same 6-block/8-request scenario now completes cleanly, 4 succeed and 4 are
rejected with a clear error instead of hanging. There's no retry/preemption
queue yet for rejected requests — noted as follow-up, not implemented.

### Benchmarks — paged KV vs. v0.2 batching, same load

| Concurrency | v0.2 batched (bs=4) tok/s | v0.3 paged (bs=4) tok/s |
|---|---|---|
| 10 | 5.2* | 36.6* |
| 20 | 48.0 | 42.0 |
| 50 | 45.9 | 41.4 |

\* c=10 numbers for both engines are the first run after server start and
include MPS warmup — not apples-to-apples with c=20/50 (see v0.2 note).

Memory: `benchmarks/bench_v0.3_memory.py` computes exact byte sizes for
this model's actual config (24 layers, 2 KV heads, head_dim 64, fp16) —
raw output in `benchmarks/results/v0.3_memory_analysis.json`.

## What's new in v0.4 — quantization (Q8_0/Q4_0-style)

**Scope decision, stated up front:** the plan called for GGUF conversion
via llama.cpp. llama.cpp is a separate C++/GGML runtime with its own
forward pass and KV-cache representation — it doesn't expose the
`past_key_values`/`DynamicCache` hooks `server/paged_engine.py` and
`server/engine.py` are built on, so pointing them at a `.gguf` file would
mean rewriting both engines against a different API, not adding a format.
Instead, `server/quantization.py` implements the same *technique* GGUF's
Q8_0 (int8) and Q4_0 (packed int4) use — per-output-channel scale + integer
weights — directly on the loaded PyTorch model, so it drops into the exact
same engines everything else in this repo runs on, and its effect is
measured in the same harness as every other milestone.

**Honest negative result (the biggest one in this project so far):**
weight-only quantization here is a pure memory optimization, not a speed
one — because MPS has no int8/int4 matmul kernel, every forward pass
dequantizes the weights back to fp16 first. Measured on this machine,
single-request, greedy, 32 tokens:

| Config | Weight memory | Compression | tok/s | vs fp16 |
|---|---|---|---|---|
| fp16 baseline | 715.7 MB | 1.0x | 35.9 | — |
| Q8_0 (int8) | 358.4 MB | 2.00x | 6.9 | **5.2x slower** |
| Q4_0 (packed int4) | 179.5 MB | 3.99x | 3.9 | **9.1x slower** |

Quality holds up reasonably at Q8_0 (coherent, on-topic), degrades
noticeably at Q4_0 (grammatically off, trails into a stray newline) — see
`sample_text` in `benchmarks/results/v0.4_quantization.json`. The memory
compression numbers are exactly what Q8_0/Q4_0 promise (2x / ~4x). The
throughput numbers are not what a real GGUF/llama.cpp deployment would
show — llama.cpp's Q4_0/Q8_0 kernels do integer matmul natively; this
from-scratch PyTorch implementation dequantizes-then-matmuls, and that
dequant step is the entire cost. This is reported as a finding about *this
implementation*, not a verdict on quantization as a technique.

### Ablation table — isolating each optimization's contribution

Same prompt, same hardware, concurrency=20, tokens/sec:

| Configuration | tok/s | Delta |
|---|---|---|
| Single-request, no batching (v0.1/v0.2 baseline) | 33.0 | — |
| + continuous batching (v0.2) | 48.0 | +45% |
| + paged KV cache (v0.3) | 42.0 | -13% vs batching alone |
| + quantization, Q8_0 (v0.4) | 17.1 | -59% vs paged KV |

Batching is the only lever that was an unambiguous win in this project.
Paged KV traded throughput for memory capacity (see v0.3). Quantization, as
implemented here without a fused low-bit kernel, traded both throughput and
some quality for memory — the isolation makes clear it's the memory lever,
not a free win, and shouldn't be read as "quantization is bad," just "naive
weight-only quantization without a matching compute kernel is a memory-only
lever on this hardware."

## What's new in v0.5 — cancellation & resource lifecycle

Client disconnects mid-generation now free their in-flight KV blocks
instead of running to completion unseen. Two layers had to change:

**Engine layer (`server/paged_engine.py`):** `PagedSlot` gained a
`cancelled` flag. `submit()` catches `asyncio.CancelledError` around
`await fut` and sets the flag; `_admit()` drops still-pending cancelled
requests before they ever allocate a block, and `_decode_step()` checks
every active slot's flag at the top of each step and frees its blocks
immediately rather than continuing to generate for it. `_free_slot()`
gained a `freed` guard so a slot can never be double-freed if it's
cancelled and finishes in the same window.

**Honest bug found while testing this milestone, not before shipping it:**
the first version only implemented the engine-side plumbing above and
called it done. Manually killing a real `curl` request mid-stream against
the running server proved that did *nothing* — FastAPI/Starlette does not
cancel a non-streaming request handler just because the client went away;
the handler runs to completion regardless, holding its KV blocks the whole
time (`paged_active: 1`, blocks not returned, confirmed via `/health`
before the fix). `/generate_paged` now explicitly races the generation
task against a polling `request.is_disconnected()` watcher and cancels the
generation the moment a disconnect is detected. Re-tested the same
kill-mid-request scenario after the fix: pool returns to
`paged_pool_free_blocks: 256/256` within one poll interval. This is the
kind of gap that's invisible in a unit test that only calls
`task.cancel()` directly — it only shows up by actually disconnecting a
real HTTP client.

### Tests

`tests/test_v0_5_cancellation.py` — 5/5 passing, against the real engine
and real model (no mocks): normal completion frees all blocks; cancelling
a still-queued (pre-prefill) request leaks nothing; cancelling mid-decode
frees blocks within one step; pool-exhaustion rejection frees cleanly with
no double-free; and a mixed cancel+completion batch fully reclaims the
pool.

## What's new in v0.6 — multi-model router

`server/router.py`: serves two genuinely different models behind one
`/generate_multi` endpoint (pick with `{"model": "qwen-0.5b" | "smollm2-360m", ...}`)
— `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, 2 KV heads) and
`HuggingFaceTB/SmolLM2-360M-Instruct` (32 layers, 5 KV heads), a different
architecture entirely, not just a different size of the same family. Each
model gets its own `Model` instance, own `PagedKVCacheEngine`, own paged KV
pool, own background decode loop — fully isolated, not a shared pool split
at request time (see DESIGN_DECISIONS.md for the tradeoff). Opt-in via
`ENABLE_ROUTER=1` since it doubles load time and memory.

**Verified, not assumed, that isolation is real:** fired one request at
each model simultaneously and measured wall-clock — Qwen's request
completed in ~1.72s and SmolLM2's in ~0.39s, each close to its own
solo-request time, confirming independent per-model queues rather than one
shared queue serializing them. Combined weight footprint measured directly:
988.1 MB + 723.6 MB = 1711.7 MB for both loaded at once — real numbers in
`benchmarks/results/v0.6_router.json`.

**Honest finding, not a router bug:** SmolLM2-360M-Instruct returns an
almost-empty completion (`new_tokens: 1`, immediate EOS) on the same raw,
non-chat-templated prompt format Qwen tolerates fine. Reproduced calling
SmolLM2 directly, outside the router, so it's a property of that model's
sensitivity to prompt formatting, not something the routing layer broke.
This repo doesn't apply per-model chat templates yet — noted as a real gap
rather than papered over with a cherry-picked prompt.

## What's new in v0.7 — observability + SLO

- `GET /metrics` — Prometheus text format: request counts by status,
  request-latency and TTFT histograms, active-request and KV-pool-free
  gauges (`server/metrics.py`).
- OpenTelemetry tracing (`server/tracing.py`) — a span per `/generate_paged`
  call with `prompt_len`/`max_new_tokens`/`new_tokens`/`elapsed_s`
  attributes. **Scope note:** exported to the console, not a real
  trace-storage backend (Jaeger/Tempo/an OTLP collector) — standing up a
  trace-storage service wasn't the point of this milestone; verifying
  spans exist with the right structure around the real request path was.
  Swapping in `OTLPSpanExporter` for a real collector is a one-line change.
- `server/slo_monitor.py` — a burn-rate SLO monitor: tracks a rolling
  window of real request latencies, and fires an alert only when p99 stays
  above the threshold for several consecutive checks (not on one slow
  request). `GET /alerts` exposes fired alerts.

**Verified the alert actually fires, not just exists as config:**
`tests/test_v0_7_slo_monitor.py` (3/3 passing) runs the real background
monitor loop with a short check interval and asserts: (1) sustained slow
latencies fire a real alert with a p99 above threshold, (2) healthy
latencies never fire one, (3) a single slow request among otherwise
healthy ones does *not* fire alone — sustained means sustained. Also fired
for real against the live server: with `SLO_LATENCY_THRESHOLD_S=0.5` and
two real ~1.8s requests, `/alerts` showed a fired alert with
`p99_latency_s: 1.824` within 3 seconds of sending them.

## Design decisions

See `docs/DESIGN_DECISIONS.md` (grows per milestone).

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn server.main:app --reload
```

```bash
curl -X POST localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a hash table?", "max_new_tokens": 64}'
```

Run the benchmark yourself:
```bash
python benchmarks/bench_v0.1_baseline.py
```

## What this doesn't do yet (v0.1 limitations)

- No concurrent request handling — every request blocks the next.
- No batching, no paged KV cache, no quantization.
- No cancellation handling, no multi-model routing, no observability.
- These land in v0.2 through v0.8 (see roadmap below), each with its own
  measured benchmark before being called done.

## Roadmap

- [x] v0.1 — baseline single-request server
- [x] v0.2 — continuous batching scheduler
- [x] v0.3 — paged KV cache
- [x] v0.4 — quantization (Q4_0/Q8_0-style, PyTorch weight-only; see scope note)
- [x] v0.5 — cancellation & resource lifecycle
- [x] v0.6 — multi-model router
- [x] v0.7 — observability + SLO
- [ ] v0.8 — comparison against vLLM
