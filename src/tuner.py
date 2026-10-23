"""
tuner.py

The Adaptive Weight Tuner: the feedback loop that makes the system
*adaptive* rather than merely cost-aware. Every ``window_s`` seconds it
looks at what actually happened (``signals.py``) and nudges parameters of
the cost-benefit engine and the shedding controller.

Mechanisms
----------
1. **Quality budget (the main one).**  Compression is the one action that
   costs *accuracy*, and the simulator cannot measure accuracy. The engine
   therefore carries an explicit exchange rate, ``quality_penalty_ms``: how
   many milliseconds of delay one read of lossy data is "worth". Left
   fixed, that constant silently decides how much of the workload is served
   from compressed data (in our experiments it swings that share from 0% to
   ~70%), and tuning it to minimise latency would just drive it to zero.
   So the operator sets a **budget** instead -- the maximum share of block
   reads that may be served from compressed data -- and the tuner holds the
   penalty at whatever level keeps the system inside it: raise it when the
   budget is exceeded, lower it (gradually) when there is slack. Latency is
   then optimised *subject to* a stated quality constraint.

2. **Shedding watermarks.**  Forced evictions or out-of-memory events mean
   the policy freed too little, too late (the simulator had to evict
   blind). That is the signal to start shedding earlier and deeper (both
   watermarks down). After several quiet windows they drift back toward
   their configured values so memory is not wasted for good.

3. **Recency time constant (optional, off by default).**  Moves the importance
   score's recency time constant toward a quantile of the observed re-use
   gaps. Our experiments found this a weak lever, so it is disabled unless
   asked for; it is kept for the ablation study.

Design choices (and why)
------------------------
* **Rule-based feedback, not learned.**  Each adjustment has a stated cause
  and an effect you can read in the log. A black-box optimiser on a noisy,
  non-stationary signal is hard to defend and hard to debug.
* **Bounded and slow.**  Step sizes are small, parameters are clamped and a
  window needs enough data before the tuner acts; a dead band between "over
  budget" and "comfortably under" stops it chattering.
* **Each mechanism can be switched off** (``adapt_*`` flags) for ablation.
* **Observes only signals a real deployment has** (counters, timestamps).
* **Deterministic**: no randomness; time comes from the injected clock.

Every step is recorded in ``history`` (``TunerRecord``) so parameter
trajectories can be plotted in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.clock import Clock
from src.controller import ControllerConfig
from src.costmodel import MsImportanceEngine
from src.signals import CounterSnapshot, Signals, WindowStats


@dataclass
class TunerConfig:
    window_s: float = 10.0
    min_accesses: int = 50  # ignore windows with too little data for the ratio rules

    # 1. Quality budget.
    adapt_quality: bool = True
    max_compressed_read_ratio: float = 0.20  # the operator's budget
    quality_slack: float = 0.5  # "comfortably under" means below slack x budget
    quality_raise_factor: float = 1.5
    quality_lower_factor: float = 1.15
    # Range the penalty may move in. The upper end defaults to the cost of
    # recomputing one block: above it, compression is never preferred over
    # evicting, so a larger value changes no decision and only winds the loop
    # up (compression is sticky: blocks stay lossy until evicted, so exposure
    # falls slowly and an unbounded integrator would overshoot far past
    # anything useful). Below ~0.25 ms compression is already near-free.
    quality_min_ms: float = 0.25
    quality_max_ms: Optional[float] = None

    # 2. Watermarks.
    adapt_watermarks: bool = True
    watermark_step: float = 0.02
    high_floor: float = 0.60
    low_floor: float = 0.40
    min_watermark_gap: float = 0.05
    quiet_windows_to_relax: int = 3

    # 3. Recency time constant (off by default).
    adapt_tau: bool = False
    tau_quantile: float = 0.9
    tau_smoothing: float = 0.3  # weight of the new target in the exponential moving average
    min_gap_samples: int = 50
    tau_min_s: float = 1.0
    tau_max_s: float = 600.0

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if not 0.0 <= self.max_compressed_read_ratio <= 1.0:
            raise ValueError("max_compressed_read_ratio must be in [0, 1]")
        if not 0.0 < self.quality_slack < 1.0:
            raise ValueError("quality_slack must be in (0, 1)")
        if self.quality_raise_factor <= 1.0 or self.quality_lower_factor <= 1.0:
            raise ValueError("quality raise/lower factors must be > 1")
        if self.quality_min_ms <= 0 or (
            self.quality_max_ms is not None and self.quality_max_ms < self.quality_min_ms
        ):
            raise ValueError("require 0 < quality_min_ms <= quality_max_ms")
        if not 0.0 < self.tau_quantile < 1.0:
            raise ValueError("tau_quantile must be in (0, 1)")
        if not 0.0 < self.tau_smoothing <= 1.0:
            raise ValueError("tau_smoothing must be in (0, 1]")
        if self.tau_min_s <= 0 or self.tau_max_s < self.tau_min_s:
            raise ValueError("require 0 < tau_min_s <= tau_max_s")


@dataclass(frozen=True)
class TunerRecord:
    """One tuner step: what it saw and the parameter values it left in force."""

    time_s: float
    stats: WindowStats
    quality_penalty_ms: float
    tau_s: float
    high_watermark: float
    low_watermark: float
    reasons: tuple = ()


def quantile(values: List[float], q: float) -> float:
    """Linear-interpolation quantile of a non-empty list."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class AdaptiveWeightTuner:
    """Feedback loop over an ``MsImportanceEngine`` and a ``ControllerConfig``."""

    engine: MsImportanceEngine
    controller_config: ControllerConfig
    signals: Signals
    clock: Clock
    config: TunerConfig = field(default_factory=TunerConfig)
    history: List[TunerRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Starting watermarks are the "home" the tuner relaxes back toward.
        self._home_high = self.controller_config.high_watermark
        self._home_low = self.controller_config.low_watermark
        self._last_snapshot: CounterSnapshot = self.signals.snapshot()
        self._last_step_time: Optional[float] = None
        self._quiet_windows = 0
        cm = self.engine.cost_model
        self._quality_max_ms = (
            self.config.quality_max_ms
            if self.config.quality_max_ms is not None
            else cm.recompute_ms_per_mb * cm.block_mb
        )

    # ------------------------------------------------------------------

    def maybe_step(self) -> Optional[TunerRecord]:
        """Run a step if a full window has elapsed since the last one."""
        now = self.clock.now()
        if self._last_step_time is None:
            self._last_step_time = now  # first call just starts the first window
            return None
        if now - self._last_step_time < self.config.window_s:
            return None
        return self.step()

    def step(self) -> TunerRecord:
        """Observe the last window and adjust parameters; returns what was done."""
        now = self.clock.now()
        snapshot = self.signals.snapshot()
        stats = snapshot.minus(self._last_snapshot)
        self._last_snapshot = snapshot
        self._last_step_time = now

        reasons: List[str] = []
        cfg = self.config
        if cfg.adapt_quality:
            self._adapt_quality(stats, reasons)
        if cfg.adapt_watermarks:
            self._adapt_watermarks(stats, reasons)
        if cfg.adapt_tau:
            self._adapt_tau(reasons)

        record = TunerRecord(
            time_s=now,
            stats=stats,
            quality_penalty_ms=self.engine.params.quality_penalty_ms,
            tau_s=self.engine.recency_half_life_s,
            high_watermark=self.controller_config.high_watermark,
            low_watermark=self.controller_config.low_watermark,
            reasons=tuple(reasons),
        )
        self.history.append(record)
        return record

    # ------------------------------------------------------------------
    # The feedback rules
    # ------------------------------------------------------------------

    def _adapt_quality(self, stats: WindowStats, reasons: List[str]) -> None:
        cfg, params = self.config, self.engine.params
        if stats.accesses < cfg.min_accesses:
            return
        ratio, budget = stats.compressed_ratio, cfg.max_compressed_read_ratio
        old = params.quality_penalty_ms
        if ratio > budget:
            new = old * cfg.quality_raise_factor
            cause = f"compressed reads {ratio:.2f} over budget {budget:.2f}"
        elif ratio < budget * cfg.quality_slack:
            new = old / cfg.quality_lower_factor
            cause = f"compressed reads {ratio:.2f} well under budget {budget:.2f}"
        else:
            return  # inside the dead band: leave it alone
        new = min(self._quality_max_ms, max(cfg.quality_min_ms, new))
        if abs(new - old) > 1e-12:
            params.quality_penalty_ms = new
            reasons.append(f"quality_penalty {old:.2f}->{new:.2f}ms ({cause})")

    def _adapt_watermarks(self, stats: WindowStats, reasons: List[str]) -> None:
        cfg, cc = self.config, self.controller_config
        too_lazy = stats.forced_evictions > 0 or stats.oom_events > 0
        if too_lazy:
            self._quiet_windows = 0
            high = max(cfg.high_floor, cc.high_watermark - cfg.watermark_step)
            low = max(cfg.low_floor, cc.low_watermark - cfg.watermark_step)
            if self._set_watermarks(high, low):
                reasons.append(
                    f"watermarks down to {cc.high_watermark:.2f}/{cc.low_watermark:.2f} "
                    f"({stats.forced_evictions} forced evictions, {stats.oom_events} OOM)"
                )
            return

        self._quiet_windows += 1
        if self._quiet_windows >= cfg.quiet_windows_to_relax:
            high = min(self._home_high, cc.high_watermark + cfg.watermark_step / 2)
            low = min(self._home_low, cc.low_watermark + cfg.watermark_step / 2)
            if self._set_watermarks(high, low):
                reasons.append(f"watermarks relaxed to {cc.high_watermark:.2f}/{cc.low_watermark:.2f}")

    def _set_watermarks(self, high: float, low: float) -> bool:
        """Apply new watermarks only if they keep the controller's invariants. True if changed."""
        cc = self.controller_config
        if high - low < self.config.min_watermark_gap or not (0.0 < low < high <= 1.0):
            return False
        if abs(high - cc.high_watermark) < 1e-12 and abs(low - cc.low_watermark) < 1e-12:
            return False
        cc.high_watermark, cc.low_watermark = high, low
        return True

    def _adapt_tau(self, reasons: List[str]) -> None:
        cfg = self.config
        gaps = self.signals.reuse_gaps()
        if len(gaps) < cfg.min_gap_samples:
            return
        target = min(cfg.tau_max_s, max(cfg.tau_min_s, quantile(gaps, cfg.tau_quantile)))
        old = self.engine.recency_half_life_s
        new = (1.0 - cfg.tau_smoothing) * old + cfg.tau_smoothing * target
        new = min(cfg.tau_max_s, max(cfg.tau_min_s, new))
        if abs(new - old) > 1e-9:
            self.engine.recency_half_life_s = new
            reasons.append(f"tau {old:.1f}->{new:.1f}s (re-use gap q{cfg.tau_quantile:.2f}={target:.1f}s)")
