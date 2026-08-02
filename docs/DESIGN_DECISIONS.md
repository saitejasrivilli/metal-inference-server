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

## v0.3 — Paged KV cache under Apple's unified memory model

This was expected to be the hardest milestone and it was, for a different
reason than expected. The textbook case for paging (vLLM's original
motivation) is protecting scarce, separate GPU VRAM on discrete-GPU
hardware from being wasted by naive per-request pre-allocation. Apple
Silicon shares one physical memory pool between CPU and GPU, so that
scarcity argument doesn't really apply here — there's no separate fast tier
being protected. What paging still buys on this hardware is capacity
efficiency at a fixed *shared* memory budget: real numbers in
`benchmarks/results/v0.3_memory_analysis.json` show ~12 concurrent
short replies servable from a paged pool sized to what a naive scheme would
reserve for only 4 fixed slots.

The actual friction was implementation-level, not conceptual: (1) building
a genuinely correct paged cache without HF's internal Cache-object
assumptions took several failed diffs before token-for-token correctness
was verified against a hand-rolled reference loop; (2) a naive Python
per-token write loop into the block pool, plus a from-scratch gather every
decode step, turned out slower than v0.2's simpler recompute-everything
approach at the concurrencies tested (20-50) — no fused paged-attention
kernel exists for MPS, so this reference-path overhead is currently
unavoidable without writing custom Metal kernels, which is out of scope
here; (3) mid-decode pool exhaustion was not handled in the first version
and silently killed the whole engine's background task, hanging every
in-flight and future request with zero error surfaced -- fixed with a
try/except around the per-step allocation that rejects only the failing
request(s).

Net: v0.3 is real, correct, and demonstrates a genuine memory-capacity
advantage, but is not a throughput win in this implementation on this
hardware at these scales. That's reported as-is rather than reframed to
look like an unambiguous improvement.

More decisions will be appended here as later milestones (quantization,
cancellation lifecycle, multi-model routing) get built. This file grows
per milestone, not rewritten.
