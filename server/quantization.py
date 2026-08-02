"""v0.4: weight-only quantization (Q8_0/Q4_0-style), applied in-place to an
already-loaded HF model so it drops into the existing paged/batched engines
unchanged.

Scoping note: the master plan calls for GGUF conversion via llama.cpp.
llama.cpp's runtime is a separate C++/GGML backend with its own forward
pass and KV-cache representation -- it doesn't expose the same
`past_key_values`/`DynamicCache` hooks server/paged_engine.py depends on, so
the v0.2/v0.3 engines can't be pointed at a `.gguf` file without rewriting
them against llama.cpp's API instead of transformers'. Rather than fork the
whole serving stack for one milestone, this implements the same *technique*
GGUF's Q8_0 and Q4_0 formats use -- per-channel/per-group scale +
integer weights -- directly on the loaded PyTorch model, so quantization's
actual effect (memory footprint, throughput, quality) can be measured
inside the same engine everything else in this repo already runs on. This
is a deliberate, documented scope decision, not an oversight.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear. Weight stored as int8 (Q8_0-style,
    per-output-channel scale) or packed int4 (Q4_0-style, two weights per
    byte, per-output-channel scale). Dequantized to fp16 on every forward
    call -- this is weight-only quantization: matmul still happens in fp16
    because MPS has no int8/int4 matmul kernel, so this saves memory, not
    compute. That tradeoff is measured, not assumed."""

    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        assert bits in (4, 8)
        self.bits = bits
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None

        weight = linear.weight.data.float()  # [out, in]
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (2 ** (bits - 1) - 1)
        q = torch.round(weight / scale).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1).to(torch.int8)

        self.register_buffer("scale", scale.to(torch.float16))
        if bits == 8:
            self.register_buffer("qweight", q)
        else:
            # pack two int4 values (offset to unsigned 0-15) per byte
            q_unsigned = (q + 8).to(torch.uint8)
            lo = q_unsigned[:, 0::2]
            hi = q_unsigned[:, 1::2]
            packed = (lo | (hi << 4)).to(torch.uint8)
            self.register_buffer("qweight", packed)

        if self.has_bias:
            self.register_buffer("bias_data", linear.bias.data.to(torch.float16))
        else:
            self.bias_data = None

    def _dequantize(self) -> torch.Tensor:
        if self.bits == 8:
            return self.qweight.to(torch.float16) * self.scale
        packed = self.qweight
        lo = (packed & 0x0F).to(torch.int16) - 8
        hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
        out = torch.empty(self.out_features, self.in_features, dtype=torch.int16, device=packed.device)
        out[:, 0::2] = lo
        out[:, 1::2] = hi
        return out.to(torch.float16) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize()
        return F.linear(x, w, self.bias_data)

    def memory_bytes(self) -> int:
        return self.qweight.numel() * self.qweight.element_size() + self.scale.numel() * self.scale.element_size()


def quantize_model_(model: nn.Module, bits: int, skip_names=("lm_head",)):
    """In-place: replace every nn.Linear not matching skip_names with a
    QuantizedLinear. Returns (num_replaced, original_bytes, quantized_bytes)."""
    original_bytes = 0
    quantized_bytes = 0
    num_replaced = 0

    def _recurse(module: nn.Module, prefix: str):
        nonlocal original_bytes, quantized_bytes, num_replaced
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(s in full_name for s in skip_names):
                original_bytes += child.weight.numel() * child.weight.element_size()
                qlinear = QuantizedLinear(child, bits=bits)
                quantized_bytes += qlinear.memory_bytes()
                setattr(module, name, qlinear)
                num_replaced += 1
            else:
                _recurse(child, full_name)

    _recurse(model, "")
    return num_replaced, original_bytes, quantized_bytes
