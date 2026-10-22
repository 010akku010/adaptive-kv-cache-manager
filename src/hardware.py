"""
hardware.py

Physical cost model for the event-driven simulator: how big a KV-cache
block is, how long it takes to move over PCIe, and how long it takes to
recompute. These are the quantities that make the four actions
(KEEP / MIGRATE / COMPRESS / EVICT) genuinely different in cost, so every
constant here is explicit, documented and overridable.

How the numbers are derived
---------------------------
KV bytes per token (derived, not assumed)::

    2 (K and V) x layers x kv_heads x head_dim x bytes_per_value

e.g. Llama-3.2-1B: 2 x 16 x 8 x 64 x 2 B = 32,768 B = 32 KB per token, so
a vLLM-style block of 16 tokens is ~0.52 MB.

Transfer time of a block over PCIe::

    setup_ms + size_MB / bandwidth_GB_per_s        (1 GB/s == 1 MB/ms)

Recompute time of a block = tokens / prefill_throughput.

IMPORTANT -- what is measured vs. assumed
-----------------------------------------
Model architecture numbers (layers, KV heads, head dim) come from the
public model configs and should be double-checked against each model's
``config.json``. Bandwidth and throughput values are **typical/assumed
figures, not measurements**: they should be replaced by numbers profiled
on the project's real GPU during the vLLM-integration phase. All results
produced by the simulator inherit this caveat and say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelProfile:
    """Architecture facts that determine KV-cache size."""

    name: str
    num_layers: int
    num_kv_heads: int  # KV heads (grouped-query attention makes this < attention heads)
    head_dim: int
    dtype_bytes: int = 2  # fp16 / bf16

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def block_size_mb(self, block_tokens: int) -> float:
        """Size of one KV block of ``block_tokens`` tokens, in (decimal) MB."""
        return self.kv_bytes_per_token * block_tokens / 1e6


@dataclass(frozen=True)
class ServingProfile:
    """A model on a specific GPU/host: everything the simulator needs about speed."""

    name: str
    model: ModelProfile
    pcie_gb_per_s: float  # effective one-direction host<->GPU bandwidth
    prefill_tokens_per_s: float  # throughput when (re)computing KV for tokens
    decode_ms_per_token: float  # time per generated token (single stream)
    pcie_setup_ms: float = 0.02  # fixed per-transfer launch overhead
    decompress_ms_per_mb: float = 0.05  # cost to dequantize a compressed block
    compression_savings: float = 0.5  # fraction of a block's size removed by COMPRESS
    request_overhead_ms: float = 2.0  # tokenization / scheduling per request

    def transfer_ms(self, size_mb: float) -> float:
        """Time to move ``size_mb`` across PCIe in one direction."""
        return self.pcie_setup_ms + size_mb / self.pcie_gb_per_s

    def prefill_ms(self, tokens: float) -> float:
        """Time to compute the KV cache for ``tokens`` tokens."""
        return tokens / self.prefill_tokens_per_s * 1000.0

    def block_size_mb(self, block_tokens: int) -> float:
        return self.model.block_size_mb(block_tokens)


# --- Models (architecture numbers: verify against each model's config.json) ---
LLAMA_3_2_1B = ModelProfile("Llama-3.2-1B", num_layers=16, num_kv_heads=8, head_dim=64)
QWEN_2_5_1_5B = ModelProfile("Qwen2.5-1.5B", num_layers=28, num_kv_heads=2, head_dim=128)
LLAMA_3_8B = ModelProfile("Llama-3-8B", num_layers=32, num_kv_heads=8, head_dim=128)

# --- Serving profiles (speeds are ASSUMED typical values; calibrate on real hardware) ---
# The project's target environment: free-tier Kaggle/Colab T4 (PCIe 3.0 x16,
# ~12 GB/s effective) with a ~1B-parameter model (Zeroth Review, Budget slide).
T4_LLAMA_3_2_1B = ServingProfile(
    name="t4-llama3.2-1b",
    model=LLAMA_3_2_1B,
    pcie_gb_per_s=12.0,
    prefill_tokens_per_s=6000.0,
    decode_ms_per_token=10.0,
)
T4_QWEN_2_5_1_5B = ServingProfile(
    name="t4-qwen2.5-1.5b",
    model=QWEN_2_5_1_5B,
    pcie_gb_per_s=12.0,
    prefill_tokens_per_s=4500.0,
    decode_ms_per_token=14.0,
)
# The base paper's testbed class (A10, PCIe 4.0): useful for comparing with its setting.
A10_LLAMA_3_8B = ServingProfile(
    name="a10-llama3-8b",
    model=LLAMA_3_8B,
    pcie_gb_per_s=20.0,
    prefill_tokens_per_s=3000.0,
    decode_ms_per_token=30.0,
)

PROFILES: Dict[str, ServingProfile] = {
    p.name: p for p in (T4_LLAMA_3_2_1B, T4_QWEN_2_5_1_5B, A10_LLAMA_3_8B)
}
DEFAULT_PROFILE = T4_LLAMA_3_2_1B
