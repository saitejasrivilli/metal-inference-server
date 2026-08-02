# Design Decisions

## v0.1 — Baseline model/runtime choice

Chose `transformers` + torch MPS over llama.cpp's Metal backend for the
first milestone because it gets a working end-to-end server up fastest with
a stock HF model — no GGUF conversion step required yet. Tradeoff: HF
`generate()` is not built for serving (no batching primitives, no paged KV),
so it's a ceiling we will hit and replace piece by piece starting in v0.2
(custom batching loop) and v0.4 (quantized GGUF weights, likely via
llama.cpp bindings at that point). Using it now, not copying its code.

## v0.2 — Recompute-per-step batching instead of KV-cache reuse

Chose to implement continuous batching first without carrying a KV cache
across decode steps, even though that means each step recomputes the full
forward pass over every active sequence's growing token history. Tradeoff:
this is asymptotically worse (O(seq_len) work per step instead of O(1)),
but it's simple, correct, and easy to reason about with dynamic
insert/evict every step. A real KV cache that survives insertion/eviction
of sequences mid-batch needs either careful left-pad-and-splice of cache
tensors or a paged allocator — that's the actual subject of v0.3, so v0.2
deliberately doesn't attempt it yet. Building the naive version first
surfaced the batch-size/MPS-stability issue documented in the README before
adding cache complexity on top of it.

## v0.2 — max_batch_size=16 wedges the MPS backend; use 4

Discovered empirically, not assumed: `max_batch_size=16` reliably hung the
PyTorch MPS backend under concurrent load (uninterruptible process state,
required `kill -9`), reproduced twice. `max_batch_size=4` ran cleanly at
concurrency up to 200. Root cause is unconfirmed (likely an MPS driver
limit on batched tensor op size under this access pattern, not something
`transformers`/`torch` expose a clean error for) — flagged as an open
question rather than papered over with a made-up explanation.

More decisions will be appended here as later milestones (paged KV under
unified memory, quorum-style tradeoffs — n/a here — heartbeat vs. gossip —
n/a here) get built. This file grows per milestone, not rewritten.
