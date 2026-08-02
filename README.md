# metal-inference-server

Continuous-batching LLM inference server built from scratch for Apple Silicon
(MPS/Metal) — because no existing engine (vLLM, SGLang) supports Apple GPUs
natively. Every number below comes from an actual run on this machine
(M-series Mac, macOS 15.5). No projected or estimated numbers.

**Status: v0.3 paged KV cache** — real block-based KV cache with actual
cross-step reuse, benchmarked against v0.2. This is the hardest milestone so
far and the results are mixed — see below, reported honestly.

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
- [ ] v0.4 — quantization (GGUF Q4_0/Q8_0)
- [ ] v0.5 — cancellation & resource lifecycle
- [ ] v0.6 — multi-model router
- [ ] v0.7 — observability + SLO
- [ ] v0.8 — comparison against vLLM
