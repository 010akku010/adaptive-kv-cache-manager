"""
costmodel.py

The cost-benefit engine re-expressed in ONE unit: milliseconds of delay.

Why this exists
---------------
The first engine (``importance_engine.AdaptiveImportanceEngine``) compared
quantities that had no common unit: an importance score in [0, 1], a
size ratio that could be 2 or more, and hand-picked constants (0.3, 1.2,
0.5, 32 MB). Subtracting such numbers ("Net Utility = Benefit - Cost") is
mathematically shaky, and in practice it never chose COMPRESS and almost
always chose EVICT. This module replaces it with quantities that mean
something physical, so that Benefit - Cost is a difference of like with
like.

The model
---------
For a block that must give up GPU memory, each action has

* a **cost**: the *expected future delay* it causes, in milliseconds::

      expected_reads = reuse_scale x importance
      cost(EVICT)    = expected_reads x recompute_ms
      cost(MIGRATE)  = expected_reads x swap_in_ms        (only offered if the CPU has room)
                     + cpu_price x MB                     (host memory is contended too)
      cost(COMPRESS) = expected_reads x (decompress_ms + quality_penalty_ms)

* a **benefit**: the memory it frees, valued at the *shadow price* of GPU
  memory, in milliseconds per MB::

      price = w_pressure x pressure^k x recompute_ms_per_MB x price_multiplier
      benefit(action) = price x MB_freed

The host tier has its own shadow price, ``cpu_price``, which rises with host
utilisation exactly like the GPU's does. It matters because a block parked on
the host is not free: when the host pool fills, the platform drops the
least-recently-used host blocks and they must be recomputed later. A diagnosis
of the default regime showed the loss this causes: a policy that compresses
before offloading fits twice as many blocks on the host, suffers half the
host evictions and recomputes less than one that offloads full-size blocks.
Without ``cpu_price`` the engine treated MIGRATE as nearly free whenever the host
had *any* room, and so filled it with blocks never read again. ``cpu_price``
defaults to 0 (the earlier behaviour) so it can be ablated.

Net Utility = benefit - cost, in milliseconds, exactly as in the project's
design. KEEP is the zero point (benefit 0, cost 0): when memory is plentiful
the price is ~0, every shedding action has negative utility, and KEEP wins.

How the five factors map onto it (Zeroth Review architecture)
-------------------------------------------------------------
1. Recency            -> importance -> expected_reads
2. Frequency          -> importance -> expected_reads
3. Migration cost     -> swap_in_ms (the block's recorded migration cost)
4. Compression benefit-> MB freed by COMPRESS (and its decompress/quality cost)
5. GPU pressure       -> the shadow price of memory

Honest note on the constants
----------------------------
Hardware-derived terms (swap_in_ms, recompute_ms, decompress_ms) come from
``hardware.py`` and are only as good as those assumed figures. Three knobs
are *exchange rates*, not physical constants, and are exposed in
``MsParams`` so they can be tuned and reported with a sensitivity analysis:

* ``reuse_scale``       -- expected future reads of a maximally important block
* ``price_multiplier``  -- how many times the recompute cost per MB a full GPU is worth
* ``quality_penalty_ms``-- the latency-equivalent charged per read of lossy data
  (the simulator does not model accuracy, so this is the only place the
  quality cost of compression enters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from src.clock import Clock
from src.hardware import ServingProfile
from src.importance_engine import (
    ActionEstimate,
    AdaptiveImportanceEngine,
    ImportanceWeights,
)
from src.metadata_collector import KVBlock

_MIN_MB = 1e-9


@dataclass(frozen=True)
class CostModel:
    """Physical costs (ms) of reading a block back, derived from a serving profile."""

    profile: ServingProfile
    block_tokens: int = 16

    @property
    def block_mb(self) -> float:
        """Size of one standard, uncompressed block."""
        return self.profile.block_size_mb(self.block_tokens)

    @property
    def recompute_ms_per_mb(self) -> float:
        """Cost of rebuilding one MB of KV cache from scratch."""
        return self.profile.prefill_ms(self.block_tokens) / self.block_mb

    def uncompressed_mb(self, block: KVBlock) -> float:
        """Size the block has (or would have) when stored at full precision."""
        if block.compressed:
            return block.size_mb / max(1e-9, 1.0 - self.profile.compression_savings)
        return block.size_mb

    def recompute_ms(self, block: KVBlock) -> float:
        """Time to rebuild this block from scratch (proportional to its full size)."""
        return self.recompute_ms_per_mb * self.uncompressed_mb(block)

    def swap_in_ms(self, block: KVBlock) -> float:
        """Time to bring this block back from CPU memory (recorded migration cost if known)."""
        if block.migration_cost_ms > 0.0:
            return block.migration_cost_ms
        return self.profile.transfer_ms(block.size_mb)

    def decompress_ms(self, block: KVBlock) -> float:
        return self.profile.decompress_ms_per_mb * block.size_mb


@dataclass
class MsParams:
    """Exchange-rate knobs of the millisecond engine (mutable: the tuner adjusts reuse_scale)."""

    reuse_scale: float = 2.0
    price_multiplier: float = 2.0
    pressure_exponent: float = 2.0
    quality_penalty_ms: float = 0.5
    # Value of host memory relative to GPU memory's recompute-based price scale.
    # 0 disables the host shadow price (the engine's behaviour before it was added).
    cpu_price_multiplier: float = 0.0


class MsImportanceEngine(AdaptiveImportanceEngine):
    """Cost-benefit estimates in milliseconds (see module docstring)."""

    def __init__(
        self,
        cost_model: CostModel,
        params: Optional[MsParams] = None,
        weights: Optional[ImportanceWeights] = None,
        recency_tau_s: float = 30.0,
        frequency_scale: float = 5.0,
        clock: Optional[Clock] = None,
        cpu_free_mb: Optional[Callable[[], float]] = None,
        cpu_utilization: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(
            weights=weights,
            recency_half_life_s=recency_tau_s,
            frequency_scale=frequency_scale,
            reference_block_mb=cost_model.block_mb,
            clock=clock,
        )
        self.cost_model = cost_model
        self.params = params if params is not None else MsParams()
        # Optional live view of free CPU memory: MIGRATE is only offered while
        # the host tier can take the block (otherwise it would be an EVICT).
        self._cpu_free_mb = cpu_free_mb
        self._cpu_utilization = cpu_utilization

    # ------------------------------------------------------------------
    # Building blocks (each returns milliseconds or ms/MB)
    # ------------------------------------------------------------------

    def expected_reads(self, block: KVBlock) -> float:
        """Expected number of future reads of this block over the planning horizon."""
        return self.params.reuse_scale * self.compute_importance(block)

    def memory_price_ms_per_mb(self, gpu_pressure: float) -> float:
        """Shadow price of GPU memory: what one freed MB is worth, in ms."""
        p = max(0.0, min(1.0, gpu_pressure))
        return (
            self.weights.gpu_pressure
            * (p ** self.params.pressure_exponent)
            * self.cost_model.recompute_ms_per_mb
            * self.params.price_multiplier
        )

    def cpu_price_ms_per_mb(self) -> float:
        """Shadow price of host memory (ms per MB); 0 when disabled or unobservable."""
        if self._cpu_utilization is None or self.params.cpu_price_multiplier <= 0.0:
            return 0.0
        u = max(0.0, min(1.0, self._cpu_utilization()))
        return (
            self.params.cpu_price_multiplier
            * (u ** self.params.pressure_exponent)
            * self.cost_model.recompute_ms_per_mb
        )

    def _cpu_has_room(self, block: KVBlock) -> bool:
        return self._cpu_free_mb is None or self._cpu_free_mb() >= block.size_mb

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def compute_action_values(self, block: KVBlock, gpu_pressure: float) -> Dict[str, ActionEstimate]:
        cm, w = self.cost_model, self.weights
        reads = self.expected_reads(block)
        price = self.memory_price_ms_per_mb(gpu_pressure)

        evict = ActionEstimate(
            benefit=price * block.size_mb,
            cost=reads * cm.recompute_ms(block),
        )
        values: Dict[str, ActionEstimate] = {"KEEP": ActionEstimate(benefit=0.0, cost=0.0)}
        # MIGRATE is only a candidate while the host tier can actually take the
        # block; offering it when it cannot would just be an EVICT in disguise.
        if self._cpu_has_room(block):
            values["MIGRATE"] = ActionEstimate(
                benefit=price * block.size_mb,
                cost=reads * w.migration_cost * cm.swap_in_ms(block)
                + self.cpu_price_ms_per_mb() * block.size_mb,
            )
        if not block.compressed and block.compression_benefit_mb > 0.0:
            freed = w.compression_benefit * block.compression_benefit_mb
            values["COMPRESS"] = ActionEstimate(
                benefit=price * freed,
                cost=reads * (cm.decompress_ms(block) + self.params.quality_penalty_ms),
            )
        values["EVICT"] = evict
        return values

    def victim_score(self, block: KVBlock) -> float:
        """
        Cheapest way to free one MB of this block, in expected ms of delay per MB.
        Lower = better victim. Independent of pressure and of ``reuse_scale``
        (both scale every block alike), so the ranking depends only on the
        block's importance and its relief costs.
        """
        cm = self.cost_model
        reads = self.expected_reads(block)
        size = max(block.size_mb, _MIN_MB)
        per_mb = [reads * cm.recompute_ms(block) / size]  # EVICT
        if self._cpu_has_room(block):
            per_mb.append(
                reads * self.weights.migration_cost * cm.swap_in_ms(block) / size + self.cpu_price_ms_per_mb()
            )
        if not block.compressed and block.compression_benefit_mb > 0.0:
            freed = max(self.weights.compression_benefit * block.compression_benefit_mb, _MIN_MB)
            per_mb.append(reads * (cm.decompress_ms(block) + self.params.quality_penalty_ms) / freed)
        return min(per_mb)
