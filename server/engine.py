"""Continuous batching engine — own code, no off-the-shelf batching runtime
exists for MPS.

v0.2 design: dynamic batch assembly (slots fill/drain every decode step) but
NO cross-step KV cache reuse yet — each step recomputes the forward pass
over each active sequence's full token history so far, batched together via
left-padding. Left-padding means the next-token logits are always at
position -1 for every slot regardless of that slot's real length, which
keeps slot bookkeeping simple.

Honest limitation: recompute-per-step is O(seq_len) work per step instead of
O(1) with a real KV cache, so this milestone's throughput ceiling is capped
by that, not by batch size. Paged KV cache (v0.3) is what fixes this.
"""
import asyncio
import time
import uuid

import torch


class Slot:
    __slots__ = (
        "request_id", "token_ids", "prompt_len", "max_new_tokens",
        "future", "submit_time", "first_token_time",
    )

    def __init__(self, request_id, token_ids, prompt_len, max_new_tokens, future):
        self.request_id = request_id
        self.token_ids = token_ids
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.future = future
        self.submit_time = time.perf_counter()
        self.first_token_time = None


class ContinuousBatchingEngine:
    def __init__(self, model, tokenizer, device, max_batch_size=16, step_sleep_s=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.step_sleep_s = step_sleep_s
        self.pending: asyncio.Queue = asyncio.Queue()
        self.active: list[Slot] = []
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id
        self._task = None

    def start(self):
        self._task = asyncio.get_event_loop().create_task(self._run_forever())

    async def submit(self, prompt: str, max_new_tokens: int = 128):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        req_id = uuid.uuid4().hex[:8]
        await self.pending.put(
            Slot(req_id, list(prompt_ids), len(prompt_ids), max_new_tokens, fut)
        )
        return await fut

    def _admit(self):
        while len(self.active) < self.max_batch_size and not self.pending.empty():
            self.active.append(self.pending.get_nowait())

    def _build_batch(self):
        max_len = max(len(s.token_ids) for s in self.active)
        input_ids = torch.full(
            (len(self.active), max_len), self.pad_id, dtype=torch.long
        )
        attn_mask = torch.zeros((len(self.active), max_len), dtype=torch.long)
        for i, s in enumerate(self.active):
            L = len(s.token_ids)
            input_ids[i, max_len - L:] = torch.tensor(s.token_ids, dtype=torch.long)
            attn_mask[i, max_len - L:] = 1
        return input_ids.to(self.device), attn_mask.to(self.device)

    @torch.inference_mode()
    def _step(self):
        input_ids, attn_mask = self._build_batch()
        out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

        finished = []
        for i, s in enumerate(self.active):
            tok = next_tokens[i].item()
            if s.first_token_time is None:
                s.first_token_time = time.perf_counter()
            s.token_ids.append(tok)
            generated = len(s.token_ids) - s.prompt_len
            if tok == self.eos_id or generated >= s.max_new_tokens:
                finished.append(i)

        for i in sorted(finished, reverse=True):
            s = self.active.pop(i)
            self._finalize(s)

    def _finalize(self, s: Slot):
        gen_ids = s.token_ids[s.prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        now = time.perf_counter()
        result = {
            "text": text,
            "new_tokens": len(gen_ids),
            "elapsed_s": now - s.submit_time,
            "ttft_s": (s.first_token_time - s.submit_time) if s.first_token_time else None,
            "tokens_per_sec": len(gen_ids) / (now - s.submit_time) if now > s.submit_time else 0.0,
        }
        if not s.future.done():
            s.future.set_result(result)

    async def _run_forever(self):
        while True:
            self._admit()
            if not self.active:
                await asyncio.sleep(0.005)
                continue
            self._step()
            if self.step_sleep_s:
                await asyncio.sleep(self.step_sleep_s)
            else:
                await asyncio.sleep(0)
