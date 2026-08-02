"""v0.2: adds continuous batching alongside the v0.1 single-request path so
the two can be A/B benchmarked on the same running process.

/generate          -> v0.1 path, one request fully processed at a time (lock)
/generate_batched  -> v0.2 path, continuous batching engine
"""
import asyncio
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.engine import ContinuousBatchingEngine
from server.model import Model
from server.paged_engine import PagedKVCacheEngine

app = FastAPI(title="metal-inference-server")
model: Model | None = None
engine: ContinuousBatchingEngine | None = None
paged_engine: PagedKVCacheEngine | None = None
_single_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


@app.on_event("startup")
async def load_model():
    global model, engine, paged_engine
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model.model_name if model else None,
        "device": model.device if model else None,
        "active_batch_size": len(engine.active) if engine else 0,
    }


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
async def generate_paged(req: GenerateRequest):
    """v0.3 path: continuous batching with a real paged KV cache."""
    try:
        return await paged_engine.submit(req.prompt, req.max_new_tokens)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
