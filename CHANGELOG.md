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
