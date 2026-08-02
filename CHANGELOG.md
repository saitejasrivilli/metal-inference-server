# Changelog

## v0.1-baseline
Single-request FastAPI server on torch MPS, Qwen2.5-0.5B-Instruct, no
batching. Establishes the control-group benchmark (~36-39 tok/s single
request, 1344MB peak RSS) that every later milestone is measured against.
See `benchmarks/results/v0.1_baseline.json`.
