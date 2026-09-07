"""
memory_tracker.py

Tracks GPU and CPU memory capacity/usage so that the rest of the system
knows how much headroom is available and how much "pressure" the GPU
tier is currently under. GPU memory pressure is one of the five inputs
to the Adaptive Importance Engine (Zeroth Review, Slide 8/10).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.metadata_collector import KVBlock, Tier


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

    def add_block(self, block: KVBlock) -> None:
        """Account for a block's memory footprint in its current tier."""
        if block.location == Tier.EVICTED:
            return
        state = self._tier_state(block.location)
        state.used_mb += block.size_mb

    def remove_block(self, block: KVBlock) -> None:
        """Reclaim a block's memory footprint from its current tier."""
        if block.location == Tier.EVICTED:
            return
        state = self._tier_state(block.location)
        state.used_mb = max(0.0, state.used_mb - block.size_mb)

    def move_block(self, block: KVBlock, new_location: Tier, new_size_mb: float = None) -> None:
        """
        Move a block from its current tier to a new one, updating both
        tiers' usage. Optionally update its size at the same time (e.g.
        a compress-then-place action).
        """
        self.remove_block(block)
        block.location = new_location
        if new_size_mb is not None:
            block.size_mb = new_size_mb
        self.add_block(block)

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
