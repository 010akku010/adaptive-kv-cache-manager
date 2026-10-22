"""
memory_tracker.py

Tracks GPU and CPU memory capacity/usage so that the rest of the system
knows how much headroom is available and how much "pressure" the GPU
tier is currently under. GPU memory pressure is one of the five inputs
to the Adaptive Importance Engine (Zeroth Review, Slide 8/10).

Capacity is *enforced*: a block can only be placed in a tier that has
room for it. Attempting to overfill a tier raises ``CapacityError`` and
leaves the tracker's state unchanged, so callers (the Tier Placement
Manager) can implement an explicit fallback such as "CPU is full, evict
instead of migrating".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.metadata_collector import KVBlock, Tier

# Tolerance for floating-point accumulation when comparing sizes (MB).
_EPS_MB = 1e-6


class CapacityError(RuntimeError):
    """Raised when a block does not fit in the requested tier."""


@dataclass
class TierState:
    """Capacity/usage accounting for a single memory tier."""

    capacity_mb: float
    used_mb: float = 0.0

    @property
    def free_mb(self) -> float:
        return max(0.0, self.capacity_mb - self.used_mb)

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently in use, in [0, 1]."""
        if self.capacity_mb <= 0:
            return 0.0
        return min(1.0, self.used_mb / self.capacity_mb)

    def can_fit(self, size_mb: float) -> bool:
        """True if ``size_mb`` more megabytes fit within capacity."""
        return self.used_mb + size_mb <= self.capacity_mb + _EPS_MB


class MemoryTracker:
    """
    Tracks how much GPU and CPU memory is in use across all known
    KV-cache blocks, and exposes the resulting "GPU pressure" signal
    used by the Adaptive Importance Engine and Decision Engine.
    """

    def __init__(self, gpu_capacity_mb: float, cpu_capacity_mb: float) -> None:
        self.gpu = TierState(capacity_mb=gpu_capacity_mb)
        self.cpu = TierState(capacity_mb=cpu_capacity_mb)

    def _tier_state(self, location: Tier) -> TierState:
        if location == Tier.GPU:
            return self.gpu
        if location == Tier.CPU:
            return self.cpu
        raise ValueError(f"No memory accounting for tier: {location}")

    # ------------------------------------------------------------------
    # Capacity queries
    # ------------------------------------------------------------------

    def can_fit(self, location: Tier, size_mb: float) -> bool:
        """Whether ``size_mb`` fits in ``location`` right now. EVICTED always fits."""
        if location == Tier.EVICTED:
            return True
        return self._tier_state(location).can_fit(size_mb)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_block(self, block: KVBlock) -> None:
        """
        Account for a block's memory footprint in its current tier.

        Raises:
            CapacityError: if the block does not fit in its tier.
        """
        if block.location == Tier.EVICTED:
            return
        state = self._tier_state(block.location)
        if not state.can_fit(block.size_mb):
            raise CapacityError(
                f"Block '{block.block_id}' ({block.size_mb:.2f} MB) does not fit in "
                f"{block.location.value}: {state.free_mb:.2f} MB free of "
                f"{state.capacity_mb:.2f} MB"
            )
        state.used_mb += block.size_mb

    def remove_block(self, block: KVBlock) -> None:
        """Reclaim a block's memory footprint from its current tier."""
        if block.location == Tier.EVICTED:
            return
        state = self._tier_state(block.location)
        state.used_mb = max(0.0, state.used_mb - block.size_mb)

    def move_block(
        self,
        block: KVBlock,
        new_location: Tier,
        new_size_mb: Optional[float] = None,
    ) -> None:
        """
        Move a block from its current tier to a new one, updating both
        tiers' usage. Optionally update its size at the same time (e.g.
        a compress-then-place action, or an in-place resize when
        ``new_location`` equals the current tier).

        The move is atomic: if the destination cannot hold the block,
        ``CapacityError`` is raised and nothing is modified.
        """
        size = block.size_mb if new_size_mb is None else new_size_mb

        if new_location != Tier.EVICTED:
            dest = self._tier_state(new_location)
            # Moving within the same tier first releases the old footprint.
            reclaimed = block.size_mb if block.location == new_location else 0.0
            if dest.used_mb - reclaimed + size > dest.capacity_mb + _EPS_MB:
                raise CapacityError(
                    f"Block '{block.block_id}' ({size:.2f} MB) does not fit in "
                    f"{new_location.value}"
                )

        self.remove_block(block)
        block.location = new_location
        block.size_mb = size
        self.add_block(block)

    def evict_block(self, block: KVBlock) -> None:
        """Drop a block from whichever tier holds it (it must be recomputed later)."""
        self.remove_block(block)
        block.location = Tier.EVICTED

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def gpu_pressure(self) -> float:
        """
        GPU memory pressure in [0, 1]. 0 = idle, 1 = completely full.
        This is the signal fed into the Adaptive Importance Engine and
        used to modulate how aggressively the Decision Engine favors
        migrate/compress/evict over keep.
        """
        return self.gpu.utilization

    def snapshot(self) -> dict:
        """Convenience summary, useful for logging/dashboards."""
        return {
            "gpu_used_mb": round(self.gpu.used_mb, 2),
            "gpu_capacity_mb": self.gpu.capacity_mb,
            "gpu_utilization": round(self.gpu.utilization, 4),
            "cpu_used_mb": round(self.cpu.used_mb, 2),
            "cpu_capacity_mb": self.cpu.capacity_mb,
            "cpu_utilization": round(self.cpu.utilization, 4),
        }
