"""Real memory-footprint comparison: paged KV pool vs. naive per-slot
pre-allocation, for the actual model config loaded on this machine.

No simulated numbers -- these are exact tensor byte sizes computed from the
real model config (layers, kv heads, head dim) and dtype, the same values
the running server actually allocates.
"""
import json
import os
from datetime import datetime, timezone

from server.model import Model
from server.paged_engine import PagedKVCacheEngine


def main():
    m = Model()
    eng = PagedKVCacheEngine(m.model, m.tokenizer, m.device, num_blocks=256, max_batch_size=4)

    bytes_per_token_per_layer = eng.num_kv_heads * eng.head_dim * 2  # fp16, one of K or V
    bytes_per_block = eng.block_size * bytes_per_token_per_layer * eng.num_layers * 2  # K+V, all layers

    num_blocks = eng.allocator.num_blocks
    paged_pool_bytes = bytes_per_block * num_blocks

    naive_configs = [128, 256, 512]
    naive = {}
    for max_seq_len in naive_configs:
        naive_bytes = eng.num_layers * 2 * eng.max_batch_size * max_seq_len * bytes_per_token_per_layer
        naive[f"max_seq_len_{max_seq_len}"] = {
            "reserved_mb": naive_bytes / 1e6,
            "max_batch_size": eng.max_batch_size,
        }

    matching_budget = naive[f"max_seq_len_128"]["reserved_mb"] * 1e6
    matching_blocks = int(matching_budget // bytes_per_block)
    typical_reply_tokens = 40
    paged_concurrent_capacity_at_same_budget = (matching_blocks * eng.block_size) // typical_reply_tokens

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": m.model_name,
        "num_layers": eng.num_layers,
        "num_kv_heads": eng.num_kv_heads,
        "head_dim": eng.head_dim,
        "block_size": eng.block_size,
        "paged_pool": {
            "num_blocks": num_blocks,
            "reserved_mb": paged_pool_bytes / 1e6,
            "token_slot_capacity": num_blocks * eng.block_size,
        },
        "naive_prealloc": naive,
        "same_budget_comparison": {
            "budget_mb": matching_budget / 1e6,
            "naive_fixed_capacity_requests": eng.max_batch_size,
            "paged_blocks_at_same_budget": matching_blocks,
            "paged_token_capacity": matching_blocks * eng.block_size,
            "assumed_typical_reply_tokens": typical_reply_tokens,
            "paged_concurrent_capacity_estimate": paged_concurrent_capacity_at_same_budget,
        },
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v0.3_memory_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
