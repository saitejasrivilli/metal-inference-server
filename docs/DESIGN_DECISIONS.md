# Design Decisions

## v0.1 — Baseline model/runtime choice

Chose `transformers` + torch MPS over llama.cpp's Metal backend for the
first milestone because it gets a working end-to-end server up fastest with
a stock HF model — no GGUF conversion step required yet. Tradeoff: HF
`generate()` is not built for serving (no batching primitives, no paged KV),
so it's a ceiling we will hit and replace piece by piece starting in v0.2
(custom batching loop) and v0.4 (quantized GGUF weights, likely via
llama.cpp bindings at that point). Using it now, not copying its code.

More decisions will be appended here as later milestones (paged KV under
unified memory, quorum-style tradeoffs — n/a here — heartbeat vs. gossip —
n/a here) get built. This file grows per milestone, not rewritten.
