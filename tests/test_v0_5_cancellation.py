"""v0.5: cancellation + resource lifecycle tests.

Verifies the paged KV pool never leaks blocks and never double-frees them,
across every cancellation/rejection pattern the engine can hit: cancel
while still queued, cancel mid-decode, pool-exhaustion rejection, and mixed
concurrent cancel + normal completion. Runs against the real model and the
real engine -- no mocking of the allocator or the model.
"""
import asyncio

import pytest

from server.model import Model
from server.paged_engine import PagedKVCacheEngine

PROMPT = "Tell me a short fact."


@pytest.fixture(scope="module")
def model():
    return Model()


def make_engine(model, num_blocks=64, max_batch_size=4):
    return PagedKVCacheEngine(model.model, model.tokenizer, model.device,
                               num_blocks=num_blocks, max_batch_size=max_batch_size)


@pytest.mark.asyncio
async def test_normal_completion_frees_all_blocks(model):
    eng = make_engine(model)
    eng.start()
    start_free = eng.allocator.num_free
    await eng.submit(PROMPT, max_new_tokens=16)
    assert eng.allocator.num_free == start_free, "blocks not returned after normal completion"


@pytest.mark.asyncio
async def test_cancel_while_still_pending_leaks_nothing(model):
    eng = make_engine(model, max_batch_size=1)
    start_free = eng.allocator.num_free
    # fill the only active slot with a long-running request first, so the
    # second request is stuck in the pending queue (never prefilled) when
    # we cancel it.
    blocker = asyncio.ensure_future(eng.submit(PROMPT, max_new_tokens=40))
    await asyncio.sleep(0.05)
    eng.start()
    await asyncio.sleep(0.1)  # let the blocker get admitted first

    victim = asyncio.ensure_future(eng.submit(PROMPT, max_new_tokens=40))
    await asyncio.sleep(0.05)
    assert victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim

    await blocker
    assert eng.allocator.num_free == start_free, "pending-cancel leaked or held blocks"


@pytest.mark.asyncio
async def test_cancel_mid_decode_frees_blocks(model):
    eng = make_engine(model, max_batch_size=4)
    eng.start()
    start_free = eng.allocator.num_free

    task = asyncio.ensure_future(eng.submit(PROMPT, max_new_tokens=64))
    # let it get admitted and generate a few tokens before cancelling
    await asyncio.sleep(0.3)
    assert task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # engine needs one more loop iteration to notice and free
    await asyncio.sleep(0.2)
    assert eng.allocator.num_free == start_free, "mid-decode cancel leaked blocks"


@pytest.mark.asyncio
async def test_pool_exhaustion_rejects_without_leak_or_double_free(model):
    eng = make_engine(model, num_blocks=6, max_batch_size=4)
    eng.start()
    start_free = eng.allocator.num_free

    results = await asyncio.gather(
        *[eng.submit(PROMPT, max_new_tokens=24) for _ in range(8)],
        return_exceptions=True,
    )
    succeeded = [r for r in results if not isinstance(r, Exception)]
    rejected = [r for r in results if isinstance(r, Exception)]
    assert len(succeeded) + len(rejected) == 8
    assert len(rejected) > 0, "test is only meaningful if the pool actually exhausts"
    assert eng.allocator.num_free == start_free, "pool exhaustion left blocks leaked or not returned"


@pytest.mark.asyncio
async def test_mixed_cancel_and_completion_reclaims_full_pool(model):
    eng = make_engine(model, num_blocks=64, max_batch_size=4)
    eng.start()
    start_free = eng.allocator.num_free

    tasks = [asyncio.ensure_future(eng.submit(PROMPT, max_new_tokens=48)) for _ in range(6)]
    await asyncio.sleep(0.25)
    # cancel half, let the rest run to completion
    for t in tasks[::2]:
        t.cancel()

    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass

    await asyncio.sleep(0.2)
    assert eng.allocator.num_free == start_free, "mixed cancel/completion did not fully reclaim the pool"
