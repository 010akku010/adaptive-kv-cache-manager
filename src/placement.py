"""
placement.py

The Tier Placement Manager: the single place where a chosen action
(KEEP / MIGRATE / COMPRESS / EVICT) is actually *executed* against the
bookkeeping modules (MetadataCollector + MemoryTracker).

Having exactly one implementation matters. Previously the simulator and
the evaluation harness each had their own copy, and the copies had
already drifted apart (one reset ``compression_benefit_mb`` after
compressing a block, the other did not). Every policy -- ours and the
baselines -- now goes through this class, so differences in results come
from the *policies*, never from differences in execution code.

In this simulator "execution" is bookkeeping only; real GPU<->CPU tensor
movement is out of scope until vLLM integration.

Fallback rules (documented behaviour, covered by tests)
-------------------------------------------------------
* MIGRATE when the CPU tier has no room      -> executed as EVICT
* COMPRESS on an already-compressed block    -> no-op (compress-once guard)
* MIGRATE on a block that is not on the GPU  -> no-op
* MIGRATE / COMPRESS / EVICT on a pinned block -> no-op (a running request
  is reading it)
Every result records the requested action, the executed action and the
reason for any deviation, so nothing is silently swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.memory_tracker import MemoryTracker
from src.metadata_collector import MetadataCollector, Tier

VALID_ACTIONS = ("KEEP", "MIGRATE", "COMPRESS", "EVICT")


@dataclass(frozen=True)
class PlacementResult:
    """Outcome of applying one action to one block."""

    block_id: str
    requested_action: str
    executed_action: str
    freed_gpu_mb: float
    fallback_reason: Optional[str] = None

    @property
    def was_fallback(self) -> bool:
        return self.fallback_reason is not None


class TierPlacementManager:
    """Executes Decision Engine / baseline actions on tracked blocks."""

    def __init__(self, collector: MetadataCollector, tracker: MemoryTracker) -> None:
        self.collector = collector
        self.tracker = tracker

    def apply(self, block_id: str, action: str) -> PlacementResult:
        """
        Execute ``action`` on the block ``block_id``.

        Raises:
            ValueError: for an unknown action name.
            KeyError: for an unknown block id.
        """
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        block = self.collector.get(block_id)
        gpu_before = self.tracker.gpu.used_mb

        def result(executed: str, reason: Optional[str] = None) -> PlacementResult:
            return PlacementResult(
                block_id=block_id,
                requested_action=action,
                executed_action=executed,
                freed_gpu_mb=max(0.0, gpu_before - self.tracker.gpu.used_mb),
                fallback_reason=reason,
            )

        if action == "KEEP":
            return result("KEEP")

        if block.pinned:
            return result("KEEP", "pinned")

        if action == "MIGRATE":
            if block.location != Tier.GPU:
                return result("KEEP", "not_on_gpu")
            if not self.tracker.can_fit(Tier.CPU, block.size_mb):
                # CPU tier is full: moving is impossible, so free the GPU
                # memory the only other way available.
                self.tracker.evict_block(block)
                return result("EVICT", "cpu_full")
            self.tracker.move_block(block, Tier.CPU)
            return result("MIGRATE")

        if action == "COMPRESS":
            if block.compressed:
                return result("KEEP", "already_compressed")
            new_size = max(0.0, block.size_mb - block.compression_benefit_mb)
            self.tracker.move_block(block, block.location, new_size_mb=new_size)
            self.collector.mark_compressed(block_id, new_size)
            return result("COMPRESS")

        # action == "EVICT"
        self.tracker.evict_block(block)
        return result("EVICT")
