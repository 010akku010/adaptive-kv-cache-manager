"""
metadata_collector.py

Defines the KVBlock data model and the MetadataCollector, the component
responsible for tracking per-block runtime metadata (recency, access
frequency, size, current tier location) that downstream modules
(MemoryTracker, AdaptiveImportanceEngine, DecisionEngine) depend on.

This corresponds to the "Metadata Collector" module in the project
architecture (Zeroth Review, Slide 8/10):

    vLLM -> Memory Tracker -> Metadata Collector -> Adaptive Importance
    Engine -> Cost-Benefit Decision Engine -> Tier Placement Manager
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional

from src.clock import Clock, SystemClock


class Tier(str, Enum):
    """Physical location of a KV-cache block."""

    GPU = "GPU"
    CPU = "CPU"
    EVICTED = "EVICTED"


@dataclass
class KVBlock:
    """
    Metadata for a single KV-cache block.

    Attributes:
        block_id: Unique identifier for the block.
        size_mb: Memory footprint of the block, in megabytes.
        created_at: Unix timestamp when the block was created.
        last_access: Unix timestamp of the block's most recent access.
        access_count: Total number of times the block has been accessed.
        location: Current tier the block resides in (GPU/CPU/EVICTED).
        migration_cost_ms: Estimated cost (latency, ms) of moving this
            block from GPU to CPU (or vice versa). Depends on size and
            interconnect bandwidth (see Part 1 of the literature review).
        compression_benefit_mb: Estimated memory that would be freed by
            compressing (e.g. quantizing) this block.
        compressed: Whether the block is currently stored in compressed
            form.
    """

    block_id: str
    size_mb: float
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 1
    location: Tier = Tier.GPU
    migration_cost_ms: float = 0.0
    compression_benefit_mb: float = 0.0
    compressed: bool = False

    def touch(self, now: Optional[float] = None) -> None:
        """
        Record a new access to this block (updates recency/frequency).

        ``now`` lets callers (the MetadataCollector) supply the time from an
        injected clock; it defaults to wall-clock time for standalone use.
        """
        self.last_access = now if now is not None else time.time()
        self.access_count += 1

    def age_seconds(self, now: Optional[float] = None) -> float:
        """Seconds elapsed since this block was last accessed."""
        now = now if now is not None else time.time()
        return max(0.0, now - self.last_access)


class MetadataCollector:
    """
    Central registry of KVBlock metadata.

    Responsible only for bookkeeping (create/update/remove/lookup).
    It intentionally has no knowledge of memory budgets (MemoryTracker's
    job) or importance scoring (AdaptiveImportanceEngine's job) -- this
    keeps the module boundaries in the architecture diagram honest and
    independently testable.
    """

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._blocks: Dict[str, KVBlock] = {}
        self.clock: Clock = clock if clock is not None else SystemClock()

    def register_block(
        self,
        block_id: str,
        size_mb: float,
        location: Tier = Tier.GPU,
        migration_cost_ms: float = 0.0,
        compression_benefit_mb: float = 0.0,
    ) -> KVBlock:
        """Register a newly created KV-cache block and return its metadata."""
        if block_id in self._blocks:
            raise ValueError(f"Block '{block_id}' is already registered")

        now = self.clock.now()
        block = KVBlock(
            block_id=block_id,
            size_mb=size_mb,
            created_at=now,
            last_access=now,
            location=location,
            migration_cost_ms=migration_cost_ms,
            compression_benefit_mb=compression_benefit_mb,
        )
        self._blocks[block_id] = block
        return block

    def record_access(self, block_id: str) -> KVBlock:
        """Update recency/frequency counters for an accessed block."""
        block = self._require(block_id)
        block.touch(self.clock.now())
        return block

    def update_location(self, block_id: str, location: Tier) -> KVBlock:
        """Update a block's tier after the Tier Placement Manager acts on it."""
        block = self._require(block_id)
        block.location = location
        return block

    def mark_compressed(self, block_id: str, new_size_mb: float) -> KVBlock:
        """Record that a block has been compressed, shrinking its footprint."""
        block = self._require(block_id)
        block.compressed = True
        block.size_mb = new_size_mb
        block.compression_benefit_mb = 0.0
        return block

    def remove_block(self, block_id: str) -> None:
        """Remove a block's metadata (used after eviction)."""
        self._blocks.pop(block_id, None)

    def get(self, block_id: str) -> KVBlock:
        return self._require(block_id)

    def all_blocks(self) -> Iterable[KVBlock]:
        return self._blocks.values()

    def blocks_in_tier(self, location: Tier) -> Iterable[KVBlock]:
        return (b for b in self._blocks.values() if b.location == location)

    def __len__(self) -> int:
        return len(self._blocks)

    def _require(self, block_id: str) -> KVBlock:
        if block_id not in self._blocks:
            raise KeyError(f"Unknown block_id '{block_id}'")
        return self._blocks[block_id]
