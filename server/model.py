"""Model loading + single-request generation on Apple Silicon (MPS)."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class Model:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_name = model_name
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.load_time_s = time.perf_counter() - t0

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - t0

        new_tokens = output.shape[1] - input_len
        text = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0

        return {
            "text": text,
            "new_tokens": new_tokens,
            "elapsed_s": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }
