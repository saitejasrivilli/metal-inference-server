"""v0.1: single-request inference server, no batching.

One request handled fully before the next starts. This is the control-group
baseline every later milestone (batching, paged KV, quantization) benchmarks
against.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from server.model import Model

app = FastAPI(title="metal-inference-server (v0.1 baseline)")
model: Model | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


@app.on_event("startup")
def load_model():
    global model
    model = Model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model.model_name if model else None,
        "device": model.device if model else None,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    result = model.generate(req.prompt, req.max_new_tokens)
    return result
