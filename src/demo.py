"""
demo.py

A recorded, replayable demonstration of the KV cache manager.

``record`` runs the event-driven simulator while capturing, once per
``sample_every_s`` of simulated time, the state of both memory tiers, plus every
placement decision and every request's time-to-first-token (TTFT). The result
is a plain dict, so it can be printed in a terminal (``replay``), saved as JSON
(``--json``), or loaded by the browser demo (``docs/demo``).

Two policies are replayed on the *same* request trace so the audience can watch
the difference with their own eyes: ``LRU+offload`` (move the oldest blocks to
host memory) versus ``Ours`` (score each block and pick the best of
KEEP / MIGRATE / COMPRESS / EVICT).

Usage:
    python -m src.demo                          # terminal replay, uniform workload
    python -m src.demo --workload zipf --speed 0   # print every frame at once
    python -m src.demo --json demo.json         # export for the web demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

from src.costmodel import MsParams
from src.eventsim import EventSimulator, SimConfig
from src.metadata_collector import Tier
from src.policies import PolicySpec, lru_offload_spec, ms_adaptive_spec
from src.trace import WORKLOADS, generate_trace
from src.tuner import TunerConfig

DEMO_DURATION_S = 120.0
DEMO_WARMUP_S = 20.0
MAX_ACTIONS_PER_SECOND = 6  # logged decisions per simulated second (keeps exports small)


def ours_spec() -> PolicySpec:
    """The full engine as shipped: ms cost model + host price + quality-budget tuner."""
    return ms_adaptive_spec(
        "Ours",
        params=MsParams(cpu_price_multiplier=2.0),
        tuner_config=TunerConfig(max_compressed_read_ratio=0.4),
    )


def baseline_spec() -> PolicySpec:
    return lru_offload_spec()


class RecordingSimulator(EventSimulator):
    """EventSimulator that also records frames, decisions and request latencies."""

    def __init__(self, *args: Any, sample_every_s: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sample_every_s = sample_every_s
        self._next_sample = 0.0
        self.frames: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.action_totals = {"MIGRATE": 0, "COMPRESS": 0, "EVICT": 0}
        self.requests: List[Dict[str, Any]] = []
        self._ttft_seen = 0
        self._pending_ttft: List[float] = []
        self._logged_this_second: Dict[int, int] = {}

    def _run_policy(self) -> None:
        prof = self.config.profile
        for result in self.policy.relieve():
            transfer_ms = prof.transfer_ms(result.freed_gpu_mb) if result.executed_action == "MIGRATE" else 0.0
            self.metrics.record_action(result, transfer_ms)
            if result.executed_action in self.action_totals:
                self.action_totals[result.executed_action] += 1
                second = int(self.clock.now())
                if self._logged_this_second.get(second, 0) < MAX_ACTIONS_PER_SECOND:
                    self._logged_this_second[second] = self._logged_this_second.get(second, 0) + 1
                    self.actions.append(
                        {
                            "t": round(self.clock.now(), 2),
                            "block": result.block_id,
                            "action": result.executed_action,
                            "freed_mb": round(result.freed_gpu_mb, 3),
                            "fallback": result.fallback_reason,
                        }
                    )
        self._manage_cpu_tier()

    def _on_tick(self) -> None:
        super()._on_tick()
        now = self.clock.now()
        self._pending_ttft += self.metrics.ttft_ms[self._ttft_seen:]
        self._ttft_seen = len(self.metrics.ttft_ms)
        if now + 1e-9 >= self._next_sample:
            self._next_sample = now + self.sample_every_s
            self.frames.append(self._frame(now, self._pending_ttft))
            self._pending_ttft = []

    def _frame(self, now: float, new_ttft: Sequence[float]) -> Dict[str, Any]:
        snap = self.tracker.snapshot()
        gpu_blocks = list(self.collector.blocks_in_tier(Tier.GPU))
        cpu_blocks = list(self.collector.blocks_in_tier(Tier.CPU))
        evicted = list(self.collector.blocks_in_tier(Tier.EVICTED))
        m = self.metrics
        return {
            "t": round(now, 1),
            "gpu_used_mb": snap["gpu_used_mb"],
            "cpu_used_mb": snap["cpu_used_mb"],
            "gpu_blocks": len(gpu_blocks),
            "gpu_compressed": sum(1 for b in gpu_blocks if b.compressed),
            "cpu_blocks": len(cpu_blocks),
            "cpu_compressed": sum(1 for b in cpu_blocks if b.compressed),
            "evicted_blocks": len(evicted),
            "migrations": self.action_totals["MIGRATE"],
            "compressions": self.action_totals["COMPRESS"],
            "evictions": self.action_totals["EVICT"],
            "requests": len(m.ttft_ms),
            "ttft_recent_ms": round(sum(new_ttft) / len(new_ttft), 2) if new_ttft else None,
        }


@dataclass(frozen=True)
class DemoSetup:
    workload: str = "uniform"
    seed: int = 0
    duration_s: float = DEMO_DURATION_S
    warmup_s: float = DEMO_WARMUP_S
    cpu_pool_mb: float = 4096.0
    gpu_pool_mb: float = 1024.0


def record(spec: PolicySpec, setup: DemoSetup, sample_every_s: float = 1.0) -> Dict[str, Any]:
    """Run one policy on the demo trace and return a JSON-serialisable recording."""
    cfg = replace(WORKLOADS[setup.workload].with_seed(setup.seed), duration_s=setup.duration_s)
    sim_cfg = SimConfig(gpu_pool_mb=setup.gpu_pool_mb, cpu_pool_mb=setup.cpu_pool_mb, warmup_s=setup.warmup_s)
    sim = RecordingSimulator(generate_trace(cfg), sim_cfg, spec, sample_every_s=sample_every_s)
    result = sim.run()
    s = result.summary
    return {
        "policy": spec.name,
        "workload": setup.workload,
        "seed": setup.seed,
        "gpu_capacity_mb": setup.gpu_pool_mb,
        "cpu_capacity_mb": setup.cpu_pool_mb,
        "duration_s": setup.duration_s,
        "warmup_s": setup.warmup_s,
        "frames": sim.frames,
        "actions": sim.actions,
        "action_totals": sim.action_totals,
        "summary": {
            k: s[k]
            for k in (
                "reuse_ttft_mean_ms", "reuse_ttft_p50_ms", "reuse_ttft_p99_ms",
                "gpu_hit_ratio", "recompute_ratio", "compressed_read_ratio",
            )
        },
    }


def record_pair(setup: DemoSetup) -> Dict[str, Any]:
    return {
        "setup": setup.__dict__,
        "hardware_note": "Simulated T4-class GPU serving a ~1B model; speeds are assumed, not measured.",
        "runs": [record(baseline_spec(), setup), record(ours_spec(), setup)],
    }


# ---------------------------------------------------------------------------
# Terminal replay
# ---------------------------------------------------------------------------

def _bar(value: float, capacity: float, width: int = 24) -> str:
    frac = 0.0 if capacity <= 0 else max(0.0, min(1.0, value / capacity))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def format_frame(run: Dict[str, Any], frame: Dict[str, Any]) -> str:
    g, c = run["gpu_capacity_mb"], run["cpu_capacity_mb"]
    ttft = "   -  " if frame["ttft_recent_ms"] is None else f"{frame['ttft_recent_ms']:6.1f}"
    return (
        f"  {run['policy']:<12}"
        f" GPU {_bar(frame['gpu_used_mb'], g)} {frame['gpu_used_mb']:6.0f}/{g:.0f} MB"
        f" | CPU {_bar(frame['cpu_used_mb'], c, 12)} {frame['cpu_used_mb']:5.0f} MB"
        f" | mig {frame['migrations']:5d} cmp {frame['compressions']:5d} evi {frame['evictions']:5d}"
        f" | TTFT {ttft} ms"
    )


def replay(data: Dict[str, Any], speed: float = 1.0, out=sys.stdout, step: int = 5) -> None:
    """Print the two recordings side by side, one frame every ``step`` seconds of simulated time."""
    runs = data["runs"]
    setup = data["setup"]
    out.write(
        f"\nKV cache manager demo: workload '{setup['workload']}', seed {setup['seed']}, "
        f"{setup['duration_s']:.0f} s of traffic, identical for both policies.\n"
        f"{data['hardware_note']}\n"
    )
    frames = list(zip(*(r["frames"] for r in runs)))
    for i, group in enumerate(frames):
        if i % step:
            continue
        out.write(f"\n t = {group[0]['t']:6.1f} s\n")
        for run, frame in zip(runs, group):
            out.write(format_frame(run, frame) + "\n")
        out.flush()
        if speed > 0:
            time.sleep(step / speed * 0.2)
    out.write("\n=== Result (requests that reuse an existing conversation, after warm-up) ===\n")
    out.write(f"  {'policy':<12}{'TTFT mean':>11}{'P50':>9}{'P99':>9}{'GPU hit':>10}{'recompute':>11}{'compressed':>12}\n")
    for run in runs:
        s = run["summary"]
        out.write(
            f"  {run['policy']:<12}{s['reuse_ttft_mean_ms']:>9.1f}ms{s['reuse_ttft_p50_ms']:>7.1f}ms"
            f"{s['reuse_ttft_p99_ms']:>7.0f}ms{s['gpu_hit_ratio']:>10.3f}{s['recompute_ratio']:>11.3f}"
            f"{s['compressed_read_ratio']:>12.3f}\n"
        )
    base, ours = runs[0]["summary"]["reuse_ttft_mean_ms"], runs[1]["summary"]["reuse_ttft_mean_ms"]
    out.write(f"\n  Mean TTFT change vs {runs[0]['policy']}: {100 * (ours - base) / base:+.1f}%\n")
    out.write(
        "  Note: faster responses come partly from serving some reads from compressed (lossy) data;\n"
        f"  that share was {runs[1]['summary']['compressed_read_ratio']:.0%} here. "
        "Its accuracy cost is not measured in simulation.\n"
    )
    recent = runs[1]["actions"][-6:]
    if recent:
        out.write("\n  Last decisions by 'Ours':\n")
        for a in recent:
            fb = f"  (fallback: {a['fallback']})" if a["fallback"] else ""
            out.write(f"    t={a['t']:7.2f}s  {a['action']:<8} {a['block']:<14} frees {a['freed_mb']:.2f} MB{fb}\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Replay a recorded demo of the KV cache manager.")
    p.add_argument("--workload", default="uniform", choices=sorted(WORKLOADS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=DEMO_DURATION_S, help="seconds of simulated traffic")
    p.add_argument("--host-starved", action="store_true", help="use a 512 MB host pool instead of 4 GB")
    p.add_argument("--speed", type=float, default=4.0, help="playback speed; 0 prints everything at once")
    p.add_argument("--json", default=None, help="also write the recording to this JSON file")
    args = p.parse_args(argv)

    setup = DemoSetup(
        workload=args.workload,
        seed=args.seed,
        duration_s=args.duration,
        warmup_s=min(DEMO_WARMUP_S, args.duration / 4),
        cpu_pool_mb=512.0 if args.host_starved else 4096.0,
    )
    print("Simulating (a few seconds)...", file=sys.stderr)
    data = record_pair(setup)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(data, fh)
        print(f"Recording written to {args.json}", file=sys.stderr)
    replay(data, speed=args.speed)


if __name__ == "__main__":
    main()
