# metal-inference-server

Continuous-batching LLM inference server built from scratch for Apple Silicon
(MPS/Metal) — because no existing engine (vLLM, SGLang) supports Apple GPUs
natively. Every number below comes from an actual run on this machine
(M-series Mac, macOS 15.5). No projected or estimated numbers.

**Status: v0.1 baseline** — single request at a time, no batching. This is
the control group every later milestone benchmarks against.

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
- [ ] v0.2 — continuous batching scheduler
- [ ] v0.3 — paged KV cache
- [ ] v0.4 — quantization (GGUF Q4_0/Q8_0)
- [ ] v0.5 — cancellation & resource lifecycle
- [ ] v0.6 — multi-model router
- [ ] v0.7 — observability + SLO
- [ ] v0.8 — comparison against vLLM
