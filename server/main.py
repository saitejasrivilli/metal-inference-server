"""v0.2: adds continuous batching alongside the v0.1 single-request path so
the two can be A/B benchmarked on the same running process.

/generate          -> v0.1 path, one request fully processed at a time (lock)
/generate_batched  -> v0.2 path, continuous batching engine
"""
import asyncio
import os
import time

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from server.engine import ContinuousBatchingEngine
from server.metrics import ACTIVE_REQUESTS, POOL_FREE_BLOCKS, REQUEST_COUNT, REQUEST_LATENCY, TTFT, metrics_payload
from server.model import Model
from server.paged_engine import PagedKVCacheEngine
from server.router import ModelRouter
from server.slo_monitor import SLOMonitor
from server.tracing import tracer

app = FastAPI(title="metal-inference-server")
model: Model | None = None
engine: ContinuousBatchingEngine | None = None
paged_engine: PagedKVCacheEngine | None = None
router: ModelRouter | None = None
slo_monitor: SLOMonitor | None = None
_single_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


class RoutedGenerateRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: int = 128


@app.on_event("startup")
async def load_model():
    global model, engine, paged_engine, router, slo_monitor
    model = Model()
    # 16 reliably wedges the MPS backend under concurrent load (see
    # docs/DESIGN_DECISIONS.md v0.2 entry) -- 4 is the safe measured default.
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "4"))
    engine = ContinuousBatchingEngine(
        model.model, model.tokenizer, model.device, max_batch_size=max_batch_size
    )
    engine.start()

    num_blocks = int(os.environ.get("NUM_KV_BLOCKS", "256"))
    paged_engine = PagedKVCacheEngine(
        model.model, model.tokenizer, model.device,
        num_blocks=num_blocks, max_batch_size=max_batch_size,
    )
    paged_engine.start()

    if os.environ.get("ENABLE_ROUTER", "0") == "1":
        router = ModelRouter(
            num_blocks_per_model=int(os.environ.get("ROUTER_BLOCKS_PER_MODEL", "128")),
            max_batch_size=max_batch_size,
        )
        router.start()

    slo_monitor = SLOMonitor(
        latency_threshold_s=float(os.environ.get("SLO_LATENCY_THRESHOLD_S", "8.0")),
        check_interval_s=float(os.environ.get("SLO_CHECK_INTERVAL_S", "5.0")),
        sustained_breaches=int(os.environ.get("SLO_SUSTAINED_BREACHES", "3")),
        window_size=int(os.environ.get("SLO_WINDOW_SIZE", "20")),
    )
    slo_monitor.start()


@app.get("/metrics")
def metrics():
    return Response(content=metrics_payload(), media_type="text/plain; version=0.0.4")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model.model_name if model else None,
        "device": model.device if model else None,
        "active_batch_size": len(engine.active) if engine else 0,
        "paged_active": len(paged_engine.active) if paged_engine else 0,
        "paged_pool_free_blocks": paged_engine.allocator.num_free if paged_engine else 0,
        "paged_pool_total_blocks": paged_engine.allocator.num_blocks if paged_engine else 0,
        "router": router.health() if router else None,
        "slo_alerts_fired": len(slo_monitor.alerts_fired) if slo_monitor else 0,
    }


@app.get("/alerts")
def alerts():
    return {"alerts": slo_monitor.alerts_fired if slo_monitor else []}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """v0.1 baseline path: strictly one request in flight at a time."""
    async with _single_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, model.generate, req.prompt, req.max_new_tokens
        )


@app.post("/generate_batched")
async def generate_batched(req: GenerateRequest):
    """v0.2 path: submitted into the continuous batching engine."""
    return await engine.submit(req.prompt, req.max_new_tokens)


@app.post("/generate_paged")
async def generate_paged(req: GenerateRequest, request: Request):
    """v0.3/v0.5 path: continuous batching with a real paged KV cache.

    FastAPI/Starlette does NOT cancel a non-streaming handler just because
    the client went away -- confirmed by hand: killing curl mid-request
    left the generation running to completion server-side and holding its
    KV blocks the whole time (found while testing this milestone). Disconnect
    has to be polled explicitly and raced against the generation task, or
    "cancellation" never actually reaches the engine at all.
    """
    async def _watch_disconnect():
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)

    ACTIVE_REQUESTS.labels(endpoint="generate_paged").inc()
    t0 = time.perf_counter()
    with tracer.start_as_current_span("generate_paged") as span:
        span.set_attribute("prompt_len", len(req.prompt))
        span.set_attribute("max_new_tokens", req.max_new_tokens)
        try:
            gen_task = asyncio.ensure_future(paged_engine.submit(req.prompt, req.max_new_tokens))
            disconnect_task = asyncio.ensure_future(_watch_disconnect())
            done, pending = await asyncio.wait(
                {gen_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if gen_task in done:
                disconnect_task.cancel()
                try:
                    result = gen_task.result()
                except RuntimeError as e:
                    REQUEST_COUNT.labels(endpoint="generate_paged", status="error").inc()
                    raise HTTPException(status_code=503, detail=str(e))

                elapsed = time.perf_counter() - t0
                REQUEST_LATENCY.labels(endpoint="generate_paged").observe(elapsed)
                if result.get("ttft_s") is not None:
                    TTFT.labels(endpoint="generate_paged").observe(result["ttft_s"])
                REQUEST_COUNT.labels(endpoint="generate_paged", status="ok").inc()
                slo_monitor.record(elapsed)
                span.set_attribute("new_tokens", result["new_tokens"])
                span.set_attribute("elapsed_s", elapsed)
                return result

            # client disconnected first -- cancel the in-flight generation so
            # the engine frees its KV blocks instead of running to completion unseen
            gen_task.cancel()
            try:
                await gen_task
            except asyncio.CancelledError:
                pass
            REQUEST_COUNT.labels(endpoint="generate_paged", status="cancelled").inc()
            span.set_attribute("cancelled", True)
            return None
        finally:
            ACTIVE_REQUESTS.labels(endpoint="generate_paged").dec()
            POOL_FREE_BLOCKS.labels(engine="paged").set(paged_engine.allocator.num_free)


@app.post("/generate_multi")
async def generate_multi(req: RoutedGenerateRequest):
    """v0.6 path: routes to one of several models, each with its own
    engine, KV pool, and queue -- serving platform, not single-model demo."""
    if router is None:
        raise HTTPException(status_code=503, detail="router disabled (set ENABLE_ROUTER=1)")
    try:
        return await router.submit(req.model, req.prompt, req.max_new_tokens)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
