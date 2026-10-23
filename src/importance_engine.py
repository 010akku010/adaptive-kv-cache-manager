"""
importance_engine.py

Implements the Adaptive Importance Engine: the module that turns raw
block metadata (from MetadataCollector) plus system state (from
MemoryTracker) into the numbers the Cost-Benefit Decision Engine
actually compares.

Design note (why these formulas):
----------------------------------
The project deliberately stays on the lightweight-heuristic end of the
importance-scoring spectrum (see literature Part 2) rather than using a
trained predictor: every quantity below is cheap to compute per-block,
per-request, with no training step and no extra inference-time model.

Five factors feed the engine, matching the Zeroth Review architecture:

    1. Recency              -> how recently the block was accessed
    2. Frequency             -> how often the block has been accessed
    3. Migration cost         -> how expensive it is to move GPU<->CPU
    4. Compression benefit    -> how much memory compression would free
    5. GPU memory pressure    -> how contested GPU memory currently is

Factors 1-2 combine into a single "importance" score representing how
valuable the block is to keep readily accessible. Factors 3-5 are used
directly to shape the estimated benefit and cost of each of the four
candidate actions (KEEP / MIGRATE / COMPRESS / EVICT), which is what
the Decision Engine (Net Utility = Benefit - Cost) ultimately compares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from src.clock import Clock, SystemClock
from src.metadata_collector import KVBlock


@dataclass
class ImportanceWeights:
    """
    Tunable weights for the five-factor score. Defaults are reasonable
    starting points for demonstration; in the full system these are
    what the (future) Adaptive Weight Tuner feedback loop would adjust
    based on observed Performance Monitor metrics.
    """

    recency: float = 0.5
    frequency: float = 0.5
    migration_cost: float = 1.0
    compression_benefit: float = 1.0
    gpu_pressure: float = 1.0


@dataclass
class ActionEstimate:
    """Estimated benefit and cost of taking a specific action on a block."""

    benefit: float
    cost: float


class AdaptiveImportanceEngine:
    """
    Computes an intrinsic "importance" score for a KV-cache block and,
    from it, the per-action (benefit, cost) estimates consumed by the
    Cost-Benefit Decision Engine.
    """

    def __init__(
        self,
        weights: "ImportanceWeights | None" = None,
        recency_half_life_s: float = 30.0,
        frequency_scale: float = 5.0,
        recompute_penalty_scale: float = 1.0,
        quality_loss_scale: float = 0.5,
        base_compress_cost: float = 0.05,
        reference_block_mb: float = 32.0,
        clock: Optional[Clock] = None,
    ) -> None:
        self.weights = weights if weights is not None else ImportanceWeights()
        # Injected clock: recency is computed against this, not time.time(),
        # so simulation runs are reproducible.
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.recency_half_life_s = recency_half_life_s
        self.frequency_scale = frequency_scale
        self.recompute_penalty_scale = recompute_penalty_scale
        self.quality_loss_scale = quality_loss_scale
        self.base_compress_cost = base_compress_cost
        # Blocks are sized relative to this reference rather than to total
        # GPU capacity: normalizing by capacity makes any single block's
        # freed-memory benefit vanishingly small (a 50MB block out of a
        # 40GB GPU is ~0.1%), which would make KEEP win almost every
        # comparison regardless of pressure. Normalizing by a "typical
        # block size" instead keeps benefit/cost on a comparable scale to
        # the [0, 1] importance score, so pressure can actually tip the
        # decision for large, cold blocks -- which is the behavior we want
        # to demonstrate.
        self.reference_block_mb = reference_block_mb

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def recency_score(self, block: KVBlock) -> float:
        """
        Exponential recency decay in (0, 1]. A block accessed just now
        scores ~1.0; a block idle for several half-lives decays toward 0.
        Chosen over a raw age value because it needs no dataset-wide max
        to stay bounded -- important for an online, per-block score.
        """
        age = block.age_seconds(self.clock.now())
        return math.exp(-age / self.recency_half_life_s)

    def frequency_score(self, block: KVBlock) -> float:
        """
        Saturating access-frequency score in [0, 1). Uses
        1 - exp(-count/scale) so early accesses matter a lot and
        additional accesses give diminishing returns, avoiding runaway
        scores for very hot blocks.
        """
        return 1.0 - math.exp(-block.access_count / self.frequency_scale)

    def compute_importance(self, block: KVBlock) -> float:
        """
        Intrinsic importance in [0, 1]: how valuable it is to keep this
        block instantly accessible on GPU, based on recency + frequency.
        """
        w = self.weights
        total_weight = w.recency + w.frequency
        if total_weight == 0:
            return 0.0
        raw = w.recency * self.recency_score(block) + w.frequency * self.frequency_score(block)
        return raw / total_weight

    def victim_score(self, block: KVBlock) -> float:
        """
        Ranking key used by the pressure-driven controller: how much
        keep-worth each MB of this block holds (importance per MB). The
        block with the LOWEST score is shed first. Subclasses with a
        richer cost model (see ``costmodel.MsImportanceEngine``) override it.
        """
        return self.compute_importance(block) / max(block.size_mb, 1e-6)

    # ------------------------------------------------------------------
    # Per-action benefit/cost estimation
    # ------------------------------------------------------------------

    def compute_action_values(
        self,
        block: KVBlock,
        gpu_pressure: float,
    ) -> Dict[str, ActionEstimate]:
        """
        Estimate (benefit, cost) for each of the four candidate actions.

        Args:
            block: the KV-cache block being evaluated.
            gpu_pressure: current GPU utilization in [0, 1], from
                MemoryTracker.gpu_pressure().

        Returns:
            A dict mapping action name -> ActionEstimate(benefit, cost).
            ``COMPRESS`` is omitted for blocks that are already compressed.
        """
        importance = self.compute_importance(block)
        w = self.weights
        gpu_pressure = max(0.0, min(1.0, gpu_pressure))
        size_factor = block.size_mb / self.reference_block_mb if self.reference_block_mb > 0 else 0.0

        # KEEP: benefit is the value of instant access; cost is the
        # opportunity cost of occupying contested GPU memory. The cost is
        # convex in pressure (pressure**2) so it stays negligible while
        # the GPU has headroom and grows sharply as the GPU fills up --
        # mirroring how real systems only start shedding memory once
        # they're genuinely under pressure, not proactively at low load.
        keep = ActionEstimate(
            benefit=importance,
            cost=w.gpu_pressure * (gpu_pressure ** 2) * size_factor,
        )

        # MIGRATE (GPU -> CPU): benefit is GPU memory freed (matters more
        # under pressure, and more for larger blocks); cost is transfer
        # latency plus a penalty for losing fast access to a block that
        # is still fairly important.
        migrate = ActionEstimate(
            benefit=w.gpu_pressure * gpu_pressure * size_factor,
            cost=(w.migration_cost * self._normalized_migration_cost(block))
            + (0.3 * importance),
        )

        # COMPRESS: benefit is the memory freed by shrinking the block in
        # place; cost is a fixed processing overhead plus a quality-loss
        # penalty that scales with how important (hot) the block is.
        # A block is compressed at most once: repeated lossy compression
        # compounds quality loss, so COMPRESS is simply not a candidate for
        # an already-compressed block.
        compress = None
        if not block.compressed:
            compress = ActionEstimate(
                benefit=w.compression_benefit
                * (block.compression_benefit_mb / self.reference_block_mb if self.reference_block_mb > 0 else 0.0)
                * (0.5 + 0.5 * gpu_pressure),
                cost=self.base_compress_cost + (self.quality_loss_scale * importance),
            )

        # EVICT: benefit is the full memory freed -- slightly more than
        # MIGRATE since there is no ongoing CPU-tier footprint; cost is
        # the recompute penalty, proportional to importance, since
        # evicting a hot block means recomputing it later.
        evict = ActionEstimate(
            benefit=1.2 * w.gpu_pressure * gpu_pressure * size_factor,
            cost=self.recompute_penalty_scale * importance,
        )

        values = {"KEEP": keep, "MIGRATE": migrate, "EVICT": evict}
        if compress is not None:
            values["COMPRESS"] = compress
        return values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_migration_cost(block: KVBlock, reference_ms: float = 50.0) -> float:
        """Squash raw migration latency (ms) into a bounded [0, 1) cost."""
        if reference_ms <= 0:
            return 0.0
        return block.migration_cost_ms / (block.migration_cost_ms + reference_ms)
