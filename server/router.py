"""v0.6: multi-model router.

Serves N models behind one process, each with its own model instance, own
paged KV pool, and own background decode loop -- fully isolated, not a
shared pool split at runtime. That's a deliberate simplicity tradeoff (see
DESIGN_DECISIONS.md): isolated budgets can't starve each other and don't
need a cross-model fairness policy, at the cost of each model reserving its
own fixed memory whether the other is busy or idle.
"""
import os

from server.model import Model
from server.paged_engine import PagedKVCacheEngine

MODEL_REGISTRY = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


class ModelRoute:
    def __init__(self, name: str, hf_name: str, num_blocks: int, max_batch_size: int):
        self.name = name
        self.model = Model(model_name=hf_name)
        self.engine = PagedKVCacheEngine(
            self.model.model, self.model.tokenizer, self.model.device,
            num_blocks=num_blocks, max_batch_size=max_batch_size,
        )

    def start(self):
        self.engine.start()


class ModelRouter:
    def __init__(self, num_blocks_per_model=128, max_batch_size=4):
        self.routes: dict[str, ModelRoute] = {}
        for name, hf_name in MODEL_REGISTRY.items():
            self.routes[name] = ModelRoute(name, hf_name, num_blocks_per_model, max_batch_size)

    def start(self):
        for route in self.routes.values():
            route.start()

    async def submit(self, model_name: str, prompt: str, max_new_tokens: int = 128):
        if model_name not in self.routes:
            raise KeyError(f"unknown model '{model_name}', available: {list(self.routes)}")
        route = self.routes[model_name]
        result = await route.engine.submit(prompt, max_new_tokens)
        result["model"] = model_name
        return result

    def health(self):
        return {
            name: {
                "hf_name": route.model.model_name,
                "active": len(route.engine.active),
                "pool_free_blocks": route.engine.allocator.num_free,
                "pool_total_blocks": route.engine.allocator.num_blocks,
            }
            for name, route in self.routes.items()
        }
