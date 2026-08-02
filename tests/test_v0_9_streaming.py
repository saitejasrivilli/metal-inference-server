"""v0.9: SSE streaming tests.

Same real-engine, no-mocks standard as tests/test_v0_5_cancellation.py.
Covers: tokens actually arrive incrementally (not buffered), cancellation
composes with v0.5's block-freeing guarantee, and streaming composes with
v0.2's continuous batching (a streaming request doesn't block other
requests from being batched alongside it).
"""
import asyncio
import time

import pytest

from server.model import Model
from server.paged_engine import STREAM_DONE, PagedKVCacheEngine

PROMPT = "Tell me a short fact about the ocean."


@pytest.fixture(scope="module")
def model():
    return Model()


def make_engine(model, num_blocks=64, max_batch_size=4):
    return PagedKVCacheEngine(model.model, model.tokenizer, model.device,
                               num_blocks=num_blocks, max_batch_size=max_batch_size)


@pytest.mark.asyncio
async def test_tokens_arrive_incrementally_not_buffered(model):
    eng = make_engine(model)
    eng.start()

    slot = await eng.submit_stream(PROMPT, max_new_tokens=24)
    arrival_times = []
    while True:
        item = await slot.stream_queue.get()
        if item is STREAM_DONE:
            break
        arrival_times.append(time.perf_counter())

    assert len(arrival_times) == 24
    span = arrival_times[-1] - arrival_times[0]
    # if tokens were all buffered and dumped at once, this span would be
    # ~0; real incremental generation takes real wall-clock time
    assert span > 0.2, f"tokens arrived in a {span:.4f}s burst -- looks buffered, not streamed"


@pytest.mark.asyncio
async def test_cancel_mid_stream_frees_blocks(model):
    eng = make_engine(model, max_batch_size=4)
    eng.start()
    start_free = eng.allocator.num_free

    slot = await eng.submit_stream(PROMPT, max_new_tokens=64)
    # consume a few tokens, same as a client reading part of an SSE stream
    for _ in range(3):
        await slot.stream_queue.get()

    slot.cancelled = True  # what the HTTP layer sets on real disconnect
    # engine needs one more decode step to notice and free
    await asyncio.sleep(0.3)

    assert eng.allocator.num_free == start_free, "cancelling mid-stream leaked blocks"

    # queue must not hang a consumer forever after cancellation
    final = await asyncio.wait_for(slot.stream_queue.get(), timeout=1.0)
    assert final is STREAM_DONE


@pytest.mark.asyncio
async def test_streaming_composes_with_batching_not_serialized(model):
    """A streaming request and a non-streaming request submitted around
    the same time must be admitted into the same batch, not one blocking
    the other -- verified by observing the engine's active list, the same
    signal the Prometheus active-request gauge is built on."""
    eng = make_engine(model, max_batch_size=4)
    eng.start()

    stream_slot = await eng.submit_stream(PROMPT, max_new_tokens=32)
    plain_task = asyncio.ensure_future(eng.submit(PROMPT, max_new_tokens=32))

    await asyncio.sleep(0.15)  # let both get admitted
    assert len(eng.active) == 2, (
        f"expected both requests batched together, got {len(eng.active)} active "
        "-- streaming may be serializing against non-streaming requests"
    )

    # drain the stream so the test doesn't leak a background task
    while (await stream_slot.stream_queue.get()) is not STREAM_DONE:
        pass
    await plain_task
