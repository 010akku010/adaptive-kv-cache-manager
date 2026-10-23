"""
eventsim.py

The event-driven KV-cache simulator: replays a policy-independent request
trace through one cache-management policy while *time passes*, memory is
*capacity-limited*, and every outcome has a *physical cost*.

What happens in a run
---------------------
Events are processed in time order from a priority queue:

* ``ARRIVAL``  -- a request needs all existing KV blocks of its
  conversation plus a few new ones. Each block is read from wherever it
  currently lives:

    GPU       -> hit; free (a compressed block adds a small dequantize cost)
    CPU       -> swap-in over PCIe: setup + size / bandwidth, queued behind
                 other swap-ins on the same link
    EVICTED   -> recompute: tokens / prefill throughput (a true miss)

  The request's TTFT is a fixed overhead + compute for brand-new tokens +
  the sum of those block-access latencies (serial, i.e. conservative).
  While the request runs, its blocks are **pinned** so no policy can move
  them. A ``COMPLETE`` event unpins them after prefill + decode time.
* ``TICK``     -- the policy gets a chance to shed memory (proactively),
  and GPU utilization is sampled.

Whenever a block needs GPU room and none is free, the policy is asked to
make room first; if it still cannot, the simulator force-evicts LRU blocks
(counted as ``forced_evictions``, a *failure* signal for the policy) and,
if even that fails because everything is pinned, the block is served
without being cached (``oom_events``).

Platform behaviour shared by all policies (so none gets special help)
---------------------------------------------------------------------
* Load-vs-recompute choice: for a block on CPU, if recomputing it is
  cheaper than waiting for and performing the transfer, recompute instead
  (the "selective load" idea from the MTDS paper).
* CPU-tier cleanup: LRU eviction when the CPU pool passes 90% full.
* Forced eviction fallback described above.

Known simplifications (state these in the report)
-------------------------------------------------
* No request queueing / batching / GPU-compute contention, so there is no
  throughput metric; TTFT excludes scheduling delay.
* Memory freed by MIGRATE becomes available immediately (real systems
  free it when the device-to-host copy finishes). Offload time is tracked
  as ``migrate_out_ms`` but is off the request path.
* Recomputing a block costs ``block_tokens / prefill_throughput``
  regardless of position; the attention-over-prefix term is ignored.
* All block sizes are equal (one standard block).
* Hardware/speed numbers are assumed typical values (see hardware.py).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.clock import ManualClock
from src.hardware import DEFAULT_PROFILE, ServingProfile
from src.memory_tracker import MemoryTracker
from src.metadata_collector import KVBlock, MetadataCollector, Tier
from src.metrics import SimMetrics
from src.placement import TierPlacementManager
from src.policies import PolicyContext, PolicySpec
from src.signals import Signals
from src.trace import Request, Trace

# Event kinds double as tie-break priority at equal timestamps.
_COMPLETE, _TICK, _ARRIVAL = 0, 1, 2


@dataclass(frozen=True)
class SimConfig:
    """Platform configuration for a simulation run."""

    profile: ServingProfile = DEFAULT_PROFILE
    block_tokens: int = 16  # vLLM's default KV block size
    # Scaled-down memory pools (a real T4 holds ~10 GB of KV): the *ratio* of
    # working set to GPU pool is what creates pressure, and small pools keep
    # runs fast.
    gpu_pool_mb: float = 1024.0
    cpu_pool_mb: float = 4096.0
    tick_interval_s: float = 0.1
    warmup_s: float = 60.0  # excluded from all metrics (cold-start effects)
    selective_load: bool = True
    cpu_high_watermark: float = 0.90
    cpu_low_watermark: float = 0.80

    @property
    def block_mb(self) -> float:
        return self.profile.block_size_mb(self.block_tokens)


@dataclass
class SimResult:
    policy: str
    workload: str
    seed: int
    summary: Dict[str, Any]
    final_state: Dict[str, Any]
    metrics: SimMetrics = field(repr=False)


@dataclass
class _Conversation:
    block_ids: List[str] = field(default_factory=list)
    created: bool = False


class EventSimulator:
    """Replays ``trace`` against the policy built from ``policy_spec``."""

    def __init__(self, trace: Trace, config: SimConfig, policy_spec: PolicySpec) -> None:
        self.trace = trace
        self.config = config
        self.policy_spec = policy_spec

        self.clock = ManualClock(0.0)
        self.collector = MetadataCollector(clock=self.clock)
        self.tracker = MemoryTracker(config.gpu_pool_mb, config.cpu_pool_mb)
        self.placement = TierPlacementManager(self.collector, self.tracker)
        self.metrics = SimMetrics()
        self.signals = Signals()  # always-on observations for the tuner (not gated by warm-up)

        self._requests_seen = 0
        self._block_seq = 0
        self._h2d_free_at = 0.0  # when the host->GPU link is next idle
        self._conversations = [_Conversation() for _ in trace.initial_blocks]
        self._pins: Dict[int, List[str]] = {}

        self.policy = policy_spec.build(
            PolicyContext(
                collector=self.collector,
                tracker=self.tracker,
                placement=self.placement,
                clock=self.clock,
                block_mb=config.block_mb,
                requests_seen=lambda: self._requests_seen,
                signals=self.signals,
                profile=config.profile,
                block_tokens=config.block_tokens,
            )
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> SimResult:
        cfg = self.config
        heap: List[Tuple[float, int, int, Any]] = []
        seq = 0
        for req in self.trace.requests:
            heap.append((req.time_s, _ARRIVAL, seq, req))
            seq += 1
        heap.append((0.0, _TICK, seq, None))
        seq += 1
        heapq.heapify(heap)
        end_time = self.trace.config.duration_s

        while heap:
            t, kind, _, payload = heapq.heappop(heap)
            self.clock.set(t)
            self.metrics.measuring = t >= cfg.warmup_s

            if kind == _ARRIVAL:
                service_s = self._on_arrival(payload, t)
                heapq.heappush(heap, (t + service_s, _COMPLETE, seq, payload.request_id))
                seq += 1
            elif kind == _COMPLETE:
                self._on_complete(payload)
            else:
                self._on_tick()
                if t + cfg.tick_interval_s <= end_time:
                    heapq.heappush(heap, (t + cfg.tick_interval_s, _TICK, seq, None))
                    seq += 1

        return SimResult(
            policy=self.policy_spec.name,
            workload=self.trace.config.name,
            seed=self.trace.config.seed,
            summary=self.metrics.summary(),
            final_state=self.tracker.snapshot(),
            metrics=self.metrics,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_arrival(self, req: Request, now: float) -> float:
        """Process a request; returns how long (seconds) it will hold its blocks."""
        cfg, prof, m = self.config, self.config.profile, self.metrics
        self._requests_seen += 1
        self.signals.record_request()
        conv = self._conversations[req.context_id]
        is_reuse = conv.created
        pins: List[str] = []
        access_ms = 0.0
        new_tokens = 0

        if is_reuse:
            for bid in list(conv.block_ids):
                block = self.collector.get(bid)
                block.pin_count += 1  # pin first: nothing may move it mid-read
                pins.append(bid)
                access_ms += self._read_block(block, now, access_ms)
                self.signals.record_reuse_gap(now - block.last_access)
                self.collector.record_access(bid)
        else:
            conv.created = True
            for _ in range(self.trace.initial_blocks[req.context_id]):
                self._create_block(conv, pins)
                new_tokens += cfg.block_tokens

        room = self.trace.config.max_context_blocks - len(conv.block_ids)
        for _ in range(max(0, min(req.new_blocks, room))):
            self._create_block(conv, pins)
            new_tokens += cfg.block_tokens

        ttft_ms = prof.request_overhead_ms + prof.prefill_ms(new_tokens) + access_ms
        m.record_request(ttft_ms, is_reuse)
        self._pins[req.request_id] = pins
        return (ttft_ms + req.output_tokens * prof.decode_ms_per_token) / 1000.0

    def _on_complete(self, request_id: int) -> None:
        for bid in self._pins.pop(request_id, []):
            self.collector.get(bid).pin_count -= 1

    def _on_tick(self) -> None:
        self._run_policy()
        self.metrics.sample_gpu_util(self.tracker.gpu_pressure())

    # ------------------------------------------------------------------
    # Block access
    # ------------------------------------------------------------------

    def _read_block(self, block: KVBlock, now: float, elapsed_ms: float) -> float:
        """Serve one block read from wherever it lives; returns latency in ms."""
        cfg, prof, m = self.config, self.config.profile, self.metrics
        t = now + elapsed_ms / 1000.0
        decompress_ms = prof.decompress_ms_per_mb * cfg.block_mb

        if block.location == Tier.GPU:
            m.record_gpu_hit(block.compressed)
            self.signals.record_gpu_hit(block.compressed)
            return decompress_ms if block.compressed else 0.0

        if block.location == Tier.CPU:
            wait_ms = max(0.0, self._h2d_free_at - t) * 1000.0
            xfer_ms = prof.transfer_ms(block.size_mb)
            if cfg.selective_load and prof.prefill_ms(cfg.block_tokens) < wait_ms + xfer_ms:
                self.tracker.evict_block(block)  # drop the CPU copy, rebuild instead
                return self._recompute_block(block)

            was_compressed = block.compressed
            if self._ensure_room(block.size_mb):
                self.tracker.move_block(block, Tier.GPU)
            else:
                m.record_oom()  # streamed through: pays the transfer, not cached
                self.signals.record_oom()
            self._h2d_free_at = t + (wait_ms + xfer_ms) / 1000.0
            m.record_cpu_hit(block.size_mb, xfer_ms, wait_ms, was_compressed)
            self.signals.record_cpu_hit(wait_ms, was_compressed)
            return wait_ms + xfer_ms + (decompress_ms if was_compressed else 0.0)

        return self._recompute_block(block)  # EVICTED

    def _recompute_block(self, block: KVBlock) -> float:
        """Rebuild an evicted block from scratch at full size; returns latency in ms."""
        cfg, prof = self.config, self.config.profile
        self._reset_block(block)
        if self._ensure_room(block.size_mb):
            self.tracker.move_block(block, Tier.GPU)
        else:
            self.metrics.record_oom()
            self.signals.record_oom()
        recompute_ms = prof.prefill_ms(cfg.block_tokens)
        self.metrics.record_recompute(recompute_ms)
        self.signals.record_recompute()
        return recompute_ms

    def _reset_block(self, block: KVBlock) -> None:
        """Return a (non-resident) block to its pristine full-size, uncompressed state."""
        cfg = self.config
        block.size_mb = cfg.block_mb
        block.compressed = False
        block.compression_benefit_mb = cfg.block_mb * cfg.profile.compression_savings
        block.migration_cost_ms = cfg.profile.transfer_ms(cfg.block_mb)

    def _create_block(self, conv: _Conversation, pins: List[str]) -> None:
        """Allocate a brand-new block (its KV was just computed) on the GPU."""
        cfg = self.config
        bid = f"b{self._block_seq:07d}"
        self._block_seq += 1
        block = self.collector.register_block(bid, size_mb=cfg.block_mb, location=Tier.EVICTED)
        self._reset_block(block)
        block.pin_count += 1
        pins.append(bid)
        conv.block_ids.append(bid)
        if self._ensure_room(block.size_mb):
            self.tracker.move_block(block, Tier.GPU)
        else:
            self.metrics.record_oom()
            self.signals.record_oom()

    # ------------------------------------------------------------------
    # Memory management helpers
    # ------------------------------------------------------------------

    def _ensure_room(self, size_mb: float) -> bool:
        """Make ``size_mb`` of GPU room: policy first, forced LRU eviction as a last resort."""
        if self.tracker.can_fit(Tier.GPU, size_mb):
            return True
        self._run_policy()
        if self.tracker.can_fit(Tier.GPU, size_mb):
            return True
        return self._force_make_room(size_mb)

    def _run_policy(self) -> None:
        prof = self.config.profile
        for result in self.policy.relieve():
            transfer_ms = prof.transfer_ms(result.freed_gpu_mb) if result.executed_action == "MIGRATE" else 0.0
            self.metrics.record_action(result, transfer_ms)
        self._manage_cpu_tier()

    def _force_make_room(self, size_mb: float) -> bool:
        victims = sorted(
            (b for b in self.collector.blocks_in_tier(Tier.GPU) if not b.pinned),
            key=lambda b: (b.last_access, b.block_id),
        )
        for block in victims:
            if self.tracker.can_fit(Tier.GPU, size_mb):
                break
            self.placement.apply(block.block_id, "EVICT")
            self.metrics.record_forced_eviction()
            self.signals.record_forced_eviction()
        return self.tracker.can_fit(Tier.GPU, size_mb)

    def _manage_cpu_tier(self) -> None:
        """Shared host-side cleanup: LRU eviction once the CPU pool is nearly full."""
        cfg = self.config
        if self.tracker.cpu.utilization < cfg.cpu_high_watermark:
            return
        victims = sorted(
            (b for b in self.collector.blocks_in_tier(Tier.CPU) if not b.pinned),
            key=lambda b: (b.last_access, b.block_id),
        )
        for block in victims:
            if self.tracker.cpu.utilization <= cfg.cpu_low_watermark:
                break
            self.placement.apply(block.block_id, "EVICT")
            self.metrics.record_cpu_eviction()
            self.signals.record_cpu_eviction()

    # ------------------------------------------------------------------
    # Self-checks (used by the tests)
    # ------------------------------------------------------------------

    def check_invariants(self) -> List[str]:
        """Return a list of violated invariants (empty means the run is consistent)."""
        problems: List[str] = []
        used = {Tier.GPU: 0.0, Tier.CPU: 0.0}
        for block in self.collector.all_blocks():
            if block.pin_count < 0:
                problems.append(f"{block.block_id}: negative pin_count")
            if block.location in used:
                used[block.location] += block.size_mb
        for tier, state in ((Tier.GPU, self.tracker.gpu), (Tier.CPU, self.tracker.cpu)):
            if abs(used[tier] - state.used_mb) > 1e-3:
                problems.append(
                    f"{tier.value}: tracker says {state.used_mb:.3f} MB but blocks sum to {used[tier]:.3f} MB"
                )
            if state.used_mb > state.capacity_mb + 1e-3:
                problems.append(f"{tier.value}: over capacity")
        return problems
