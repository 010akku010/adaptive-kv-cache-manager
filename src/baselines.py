"""
baselines.py

Implements the three comparison baselines promised in the Zeroth Review
(Slide 9/12): FIFO, LRU, and a simplified MTDS-style static-threshold
policy. These exist so the project can make an honest, apples-to-apples
claim like "our adaptive engine achieves X% better GPU utilization than
FIFO/LRU/MTDS on the same workload" instead of an unsupported one.

Design note -- why these are "batch" policies and our engine is
"per-block":
-------------------------------------------------------------------
FIFO and LRU are classical *cache replacement* policies: they don't ask
"is this one block worth keeping?" in isolation, they ask "given all
resident blocks, which ones must go to relieve pressure?" That's a
different shape of decision than our engine's per-block Net Utility
comparison, and reproducing that difference honestly is itself part of
the point -- it's precisely the "actions evaluated independently, with
no cost-benefit reasoning" limitation called out in the Research Gap
slide. Modeling the baselines the way real FIFO/LRU work (rather than
forcing them into our per-block interface) keeps the comparison fair.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.metadata_collector import KVBlock, Tier


class BaselinePolicy:
    """Common interface: given resident blocks + pressure, decide actions."""

    name = "baseline"

    def select_actions(
        self,
        blocks: List[KVBlock],
        gpu_pressure: float,
        gpu_capacity_mb: float,
        target_pressure: float = 0.7,
        trigger_pressure: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        ``trigger_pressure`` (optional) is the utilization at which shedding
        begins; below it every block is kept. Setting it equal to the
        controller's high watermark gives all policies the same trigger and
        the same target, so the comparison is apples to apples. When
        ``None`` the policy sheds whenever pressure exceeds the target.
        """
        raise NotImplementedError


class FIFOPolicy(BaselinePolicy):
    """
    Evicts the oldest-created blocks first, regardless of how important
    or frequently used they currently are, until pressure drops back to
    `target_pressure`. Only ever chooses KEEP or EVICT -- classic FIFO
    has no notion of migration or compression.
    """

    name = "FIFO"

    def select_actions(self, blocks, gpu_pressure, gpu_capacity_mb, target_pressure=0.7, trigger_pressure=None):
        return _evict_until_target(
            blocks, gpu_pressure, gpu_capacity_mb, target_pressure,
            sort_key=lambda b: b.created_at,  # oldest-created first
            trigger_pressure=trigger_pressure,
        )


class LRUPolicy(BaselinePolicy):
    """
    Evicts the Least Recently Used blocks first -- an improvement over
    FIFO because it accounts for recency, but (unlike our engine) still
    ignores access frequency, migration cost, and compression benefit,
    and still only chooses between KEEP and EVICT.
    """

    name = "LRU"

    def select_actions(self, blocks, gpu_pressure, gpu_capacity_mb, target_pressure=0.7, trigger_pressure=None):
        return _evict_until_target(
            blocks, gpu_pressure, gpu_capacity_mb, target_pressure,
            sort_key=lambda b: b.last_access,  # least-recently-accessed first
            trigger_pressure=trigger_pressure,
        )


class MTDSStaticThresholdPolicy(BaselinePolicy):
    """
    Simplified stand-in for the base paper's static-threshold approach:
    a single fixed rule is applied to *every* block based only on the
    current GPU pressure -- it never looks at an individual block's
    importance, so a hot, frequently-used block and a cold, stale one
    are treated identically whenever pressure crosses a threshold. This
    is precisely the "static, not adaptive" and "no per-block cost-
    benefit reasoning" gap identified in the literature review.
    """

    name = "MTDS-static"

    def __init__(self, migrate_threshold: float = 0.75, evict_threshold: float = 0.9):
        self.migrate_threshold = migrate_threshold
        self.evict_threshold = evict_threshold

    def select_actions(self, blocks, gpu_pressure, gpu_capacity_mb, target_pressure=0.7, trigger_pressure=None):
        # MTDS's own thresholds define when it acts; trigger/target are ignored.
        if gpu_pressure >= self.evict_threshold:
            action = "EVICT"
        elif gpu_pressure >= self.migrate_threshold:
            action = "MIGRATE"
        else:
            action = "KEEP"
        # Same action for every block -- that uniformity IS the limitation.
        return {b.block_id: action for b in blocks}


def _evict_until_target(
    blocks: List[KVBlock],
    gpu_pressure: float,
    gpu_capacity_mb: float,
    target_pressure: float,
    sort_key,
    trigger_pressure: Optional[float] = None,
) -> Dict[str, str]:
    """
    Shared logic for queue-ordered eviction (FIFO/LRU): repeatedly evict
    the front of the ordered queue until the estimated GPU utilization
    drops to `target_pressure` (or there is nothing left to evict).
    """
    actions = {b.block_id: "KEEP" for b in blocks}
    if gpu_pressure <= target_pressure or gpu_capacity_mb <= 0:
        return actions
    if trigger_pressure is not None and gpu_pressure < trigger_pressure:
        return actions

    used_mb = gpu_pressure * gpu_capacity_mb
    target_mb = target_pressure * gpu_capacity_mb
    ordered = sorted((b for b in blocks if b.location == Tier.GPU), key=sort_key)

    for block in ordered:
        if used_mb <= target_mb:
            break
        actions[block.block_id] = "EVICT"
        used_mb -= block.size_mb

    return actions
