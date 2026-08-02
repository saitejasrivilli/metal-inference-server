"""v0.3: paged KV cache.

Physical KV storage is a fixed pool of fixed-size blocks, shared across all
sequences (per layer). Each sequence holds a logical block table (a list of
block ids) instead of one monolithic growing tensor. Real KV-cache reuse:
unlike v0.2, decode steps do NOT recompute the forward pass over full
sequence history -- only the new token is fed through the model each step,
with past attention supplied from the pool.

Why paging matters on Apple's unified memory model specifically: on a
discrete-GPU box, host RAM and GPU VRAM are separate, so a naive
per-sequence pre-allocation (reserve `max_seq_len` slots per request up
front) wastes *scarce, expensive* VRAM whenever a request finishes early.
On Apple Silicon, GPU and CPU share one physical memory pool, so the
scarcity argument is weaker -- the OS will page/evict as needed and the
"waste" is DRAM, not a dedicated fast tier. The benefit paging still gives
here is capacity: fixed-size blocks let many more short/medium sequences
share a bounded pool without over-reserving for a worst-case max_seq_len
that most requests never reach.

Each decode step still gathers each active sequence's blocks into a
contiguous batched tensor to hand to the model (no fused paged-attention
kernel -- MPS has no such kernel available, this is the reference/gather
path, same tradeoff vLLM had before writing custom CUDA kernels). That
gather is real, measurable overhead -- documented honestly in the README
rather than presented as free.
"""
import asyncio
import time
import uuid

import torch
from transformers.cache_utils import DynamicCache

BLOCK_SIZE = 16


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks))

    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("paged KV pool exhausted: no free blocks")
        return self.free.pop()

    def free_block(self, block_id: int):
        self.free.append(block_id)

    @property
    def num_free(self):
        return len(self.free)


class PagedSlot:
    __slots__ = (
        "request_id", "token_ids", "prompt_len", "max_new_tokens",
        "block_table", "future", "submit_time", "first_token_time",
    )

    def __init__(self, request_id, token_ids, prompt_len, max_new_tokens, future):
        self.request_id = request_id
        self.token_ids = token_ids
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.block_table: list[int] = []
        self.future = future
        self.submit_time = time.perf_counter()
        self.first_token_time = None

    @property
    def seq_len(self):
        return len(self.token_ids)


class PagedKVCacheEngine:
    def __init__(self, model, tokenizer, device, num_blocks=256,
                 max_batch_size=4, block_size=BLOCK_SIZE):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.num_layers = model.config.num_hidden_layers
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        dtype = next(model.parameters()).dtype

        self.allocator = BlockAllocator(num_blocks)
        self.k_pool = [
            torch.zeros(num_blocks, block_size, self.num_kv_heads, self.head_dim,
                        dtype=dtype, device=device)
            for _ in range(self.num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, block_size, self.num_kv_heads, self.head_dim,
                        dtype=dtype, device=device)
            for _ in range(self.num_layers)
        ]

        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pending: asyncio.Queue = asyncio.Queue()
        self.active: list[PagedSlot] = []
        self._task = None

    def start(self):
        self._task = asyncio.get_event_loop().create_task(self._run_forever())

    async def submit(self, prompt: str, max_new_tokens: int = 128):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        req_id = uuid.uuid4().hex[:8]
        await self.pending.put(
            PagedSlot(req_id, list(prompt_ids), len(prompt_ids), max_new_tokens, fut)
        )
        return await fut

    # -- block table helpers --------------------------------------------

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _ensure_capacity(self, slot: PagedSlot, num_tokens: int):
        needed = self._blocks_needed(num_tokens)
        while len(slot.block_table) < needed:
            slot.block_table.append(self.allocator.alloc())

    def _write_kv(self, slot: PagedSlot, layer_idx: int, keys, values, start_pos: int):
        """Write keys/values (shape [T, H, D]) into the pool starting at
        token position start_pos in this sequence's logical stream."""
        T = keys.shape[0]
        for t in range(T):
            pos = start_pos + t
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = slot.block_table[block_idx]
            self.k_pool[layer_idx][block_id, offset] = keys[t]
            self.v_pool[layer_idx][block_id, offset] = values[t]

    def _gather_kv(self, slot: PagedSlot, layer_idx: int, up_to: int):
        """Return (keys, values) each shape [up_to, H, D] gathered from pool."""
        n_blocks = self._blocks_needed(up_to)
        block_ids = slot.block_table[:n_blocks]
        k = self.k_pool[layer_idx][block_ids].reshape(-1, self.num_kv_heads, self.head_dim)
        v = self.v_pool[layer_idx][block_ids].reshape(-1, self.num_kv_heads, self.head_dim)
        return k[:up_to], v[:up_to]

    def _free_slot(self, slot: PagedSlot):
        for b in slot.block_table:
            self.allocator.free_block(b)
        slot.block_table = []

    # -- prefill / decode --------------------------------------------------

    @torch.inference_mode()
    def _prefill(self, slot: PagedSlot):
        input_ids = torch.tensor([slot.token_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values

        self._ensure_capacity(slot, slot.prompt_len)
        for layer_idx in range(self.num_layers):
            layer = cache.layers[layer_idx]
            keys = layer.keys[0].transpose(0, 1)    # [P, H, D]
            values = layer.values[0].transpose(0, 1)  # [P, H, D]
            self._write_kv(slot, layer_idx, keys, values, start_pos=0)

        next_token = torch.argmax(out.logits[0, -1, :]).item()
        slot.token_ids.append(next_token)
        slot.first_token_time = time.perf_counter()
        self._check_finished(slot)

    def _admit(self):
        while len(self.active) < self.max_batch_size and not self.pending.empty():
            slot = self.pending.get_nowait()
            needed = self._blocks_needed(slot.prompt_len)
            if self.allocator.num_free < needed:
                # not enough room for this prompt right now -- try again next round
                self.pending._queue.appendleft(slot)  # noqa: SLF001
                break
            self._prefill(slot)  # calls _check_finished, which manages self.active

    def _reject(self, slot: PagedSlot, reason: str):
        self._free_slot(slot)
        if not slot.future.done():
            slot.future.set_exception(RuntimeError(reason))

    @torch.inference_mode()
    def _decode_step(self):
        # Growing a sequence's KV cache can need a fresh block at any step
        # (not just at admission time). Left unhandled, an exhausted pool
        # raises inside the background task and silently kills the whole
        # engine loop -- every future submitted after that point hangs
        # forever with no error surfaced. Reproduced with num_blocks=6 and
        # 8 concurrent requests. A pool-exhaustion pre-check here is racy
        # (admission and decode both consume the same free-block counter
        # between check and use), so the actual allocation attempt below is
        # wrapped in try/except instead and rejects only the slot(s) that
        # really fail, rather than trying to predict failure in advance.
        B = len(self.active)
        lengths = [s.seq_len - 1 for s in self.active]  # tokens already in cache (excludes not-yet-fed last token)
        Lmax = max(lengths)

        batched_keys = [
            torch.zeros(B, Lmax, self.num_kv_heads, self.head_dim,
                        dtype=self.k_pool[0].dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        batched_values = [torch.zeros_like(k) for k in batched_keys]
        attn_mask = torch.zeros(B, Lmax + 1, dtype=torch.long, device=self.device)
        input_ids = torch.zeros(B, 1, dtype=torch.long, device=self.device)

        for i, s in enumerate(self.active):
            L = lengths[i]
            for layer_idx in range(self.num_layers):
                k, v = self._gather_kv(s, layer_idx, L)
                batched_keys[layer_idx][i, Lmax - L:] = k
                batched_values[layer_idx][i, Lmax - L:] = v
            attn_mask[i, Lmax - L:] = 1
            attn_mask[i, -1] = 1
            input_ids[i, 0] = s.token_ids[-1]

        cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            cache.update(
                batched_keys[layer_idx].transpose(1, 2),
                batched_values[layer_idx].transpose(1, 2),
                layer_idx,
            )

        out = self.model(input_ids=input_ids, attention_mask=attn_mask,
                          past_key_values=cache, use_cache=True)
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)
        new_cache = out.past_key_values

        finished = []
        rejected = []
        for i, s in enumerate(self.active):
            L = lengths[i]
            try:
                self._ensure_capacity(s, L + 1)
            except RuntimeError:
                rejected.append(i)
                continue
            for layer_idx in range(self.num_layers):
                new_k = new_cache.layers[layer_idx].keys[i, :, -1, :].unsqueeze(0)   # [1,H,D]
                new_v = new_cache.layers[layer_idx].values[i, :, -1, :].unsqueeze(0)
                self._write_kv(s, layer_idx, new_k, new_v, start_pos=L)

            tok = next_tokens[i].item()
            s.token_ids.append(tok)
            if s.first_token_time is None:
                s.first_token_time = time.perf_counter()
            generated = len(s.token_ids) - s.prompt_len
            if tok == self.eos_id or generated >= s.max_new_tokens:
                finished.append(i)

        for i in sorted(set(finished + rejected), reverse=True):
            s = self.active.pop(i)
            if i in rejected:
                self._reject(s, "paged KV pool exhausted mid-decode; request dropped")
            else:
                self._check_finished(s)

    def _check_finished(self, slot: PagedSlot):
        generated = len(slot.token_ids) - slot.prompt_len
        if slot.token_ids[-1] == self.eos_id or generated >= slot.max_new_tokens:
            self._finalize(slot)
        else:
            if slot not in self.active:
                self.active.append(slot)

    def _finalize(self, slot: PagedSlot):
        gen_ids = slot.token_ids[slot.prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        now = time.perf_counter()
        result = {
            "text": text,
            "new_tokens": len(gen_ids),
            "elapsed_s": now - slot.submit_time,
            "ttft_s": (slot.first_token_time - slot.submit_time) if slot.first_token_time else None,
            "tokens_per_sec": len(gen_ids) / (now - slot.submit_time) if now > slot.submit_time else 0.0,
            "blocks_used": len(slot.block_table),
        }
        self._free_slot(slot)
        if not slot.future.done():
            slot.future.set_result(result)

    async def _run_forever(self):
        while True:
            self._admit()
            if not self.active:
                await asyncio.sleep(0.005)
                continue
            self._decode_step()
            await asyncio.sleep(0)
