"""
experiment.py

Runs every policy on every workload, across several random seeds, on
*identical* traces, and reports mean +/- standard deviation.

Fairness guarantees
-------------------
* For a given (workload, seed) the trace is generated once and replayed
  against every policy, so differences come from the policies only.
* All policies run on the same platform (SimConfig) and share the same
  platform behaviour (CPU-tier cleanup, forced-eviction fallback,
  load-vs-recompute choice).
* Everything is deterministic: the same command prints the same numbers.
* Design decisions (tuner rules, parameter bounds) were made on *tuning
  seeds* (100-102); reported results use *held-out seeds* (0 upward).

Platform regimes
----------------
* ``default``       -- 1 GB GPU pool, 4 GB host pool (host memory is plentiful).
* ``host-starved``  -- 1 GB GPU pool, 512 MB host pool. Offloading is no longer
  a free lunch, which is where choosing between migrate / compress / evict
  actually matters.

Usage
-----
    python -m src.experiment                                   # 4 workloads, 3 seeds, default regime
    python -m src.experiment --regime host-starved --set full --seeds 5 --jobs 2 \\
        --out results/full_host_starved.json
    python -m src.analysis results/full_*.json                 # statistics on saved results

This reports *simulated* results from a model of a T4-class GPU serving a
~1B-parameter model. Hardware speeds are assumed typical values (see
src/hardware.py), not measurements.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.costmodel import MsParams
from src.eventsim import EventSimulator, SimConfig, SimResult
from src.hardware import PROFILES
from src.policies import (
    PolicySpec,
    adaptive_spec,
    default_policy_specs,
    fifo_spec,
    ladder_spec,
    lru_offload_spec,
    lru_spec,
    ms_adaptive_spec,
    mtds_spec,
)
from src.trace import WORKLOADS, WorkloadConfig, generate_trace
from src.tuner import TunerConfig

REGIMES: Dict[str, SimConfig] = {
    "default": SimConfig(),
    "host-starved": SimConfig(cpu_pool_mb=512.0),
}

TUNING_SEEDS = (100, 101, 102)  # used for design decisions only; never reported

# Metrics shown in the report table: (key, label, format).
REPORT_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("reuse_ttft_mean_ms", "TTFT mean(ms)", "{:.1f}"),
    ("reuse_ttft_p50_ms", "TTFT P50(ms)", "{:.2f}"),
    ("reuse_ttft_p99_ms", "TTFT P99(ms)", "{:.0f}"),
    ("gpu_hit_ratio", "GPU hit", "{:.3f}"),
    ("recompute_ratio", "Recompute", "{:.3f}"),
    ("compressed_read_ratio", "Compr.reads", "{:.3f}"),
)

TUNER_BUDGETS = (0.1, 0.2, 0.4, 0.6, 1.0)
LADDER_FRACTIONS = (0.25, 0.5, 1.0)
HOST_PRICE_BUDGETS = (0.2, 0.4, 0.6)
HOST_PRICE_MULTIPLIER = 2.0  # chosen on tuning seeds 100-102


def full_policy_specs() -> List[PolicySpec]:
    """
    Everything the final evaluation compares: the classical baselines, the
    static compress-first ladder at several caps, our engine in its first
    (unitless) and millisecond forms, and the tuned engine at several quality
    budgets (which together trace out its latency-vs-quality frontier).
    """
    specs: List[PolicySpec] = [
        fifo_spec(),
        lru_spec(),
        lru_offload_spec(),
        mtds_spec(),
        adaptive_spec(name="Ours-v1"),
        ms_adaptive_spec("Ours-ms"),
    ]
    specs += [ladder_spec(max_compressed_fraction=f, name=f"Ladder {f:g}") for f in LADDER_FRACTIONS]
    specs += [
        ms_adaptive_spec(f"Ours-ms+tuner b={b:g}", tuner_config=TunerConfig(max_compressed_read_ratio=b))
        for b in TUNER_BUDGETS
    ]
    # Ablation: the same tuned engine with the host-memory shadow price switched on.
    specs += [
        ms_adaptive_spec(
            f"Ours-ms+host+tuner b={b:g}",
            params=MsParams(cpu_price_multiplier=HOST_PRICE_MULTIPLIER),
            tuner_config=TunerConfig(max_compressed_read_ratio=b),
        )
        for b in HOST_PRICE_BUDGETS
    ]
    return specs


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

# Jobs are rebuilt in worker processes from indices into this table because
# PolicySpec holds lambdas, which cannot be pickled. Requires the "fork" start
# method; otherwise the runner falls back to a single process.
_JOBS: List[Tuple[WorkloadConfig, PolicySpec, SimConfig]] = []


def _run_job(index: int) -> SimResult:
    workload, spec, sim_config = _JOBS[index]
    return EventSimulator(generate_trace(workload), sim_config, spec).run()


def run_experiment(
    workloads: Sequence[WorkloadConfig],
    specs: Sequence[PolicySpec],
    seeds: Sequence[int],
    sim_config: Optional[SimConfig] = None,
    processes: int = 1,
) -> List[SimResult]:
    """Run the full workload x seed x policy matrix (optionally across processes)."""
    global _JOBS
    sim_config = sim_config or SimConfig()
    _JOBS = [
        (workload.with_seed(seed), spec, sim_config)
        for workload in workloads
        for seed in seeds
        for spec in specs
    ]
    indices = list(range(len(_JOBS)))
    if processes > 1 and "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(processes) as pool:
            return pool.map(_run_job, indices, chunksize=1)
    return [_run_job(i) for i in indices]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate(results: Sequence[SimResult]) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    """Group by (workload, policy); return {metric: (mean, std over seeds)} for numeric metrics."""
    groups: Dict[Tuple[str, str], List[SimResult]] = defaultdict(list)
    for r in results:
        groups[(r.workload, r.policy)].append(r)

    out: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for key, rs in groups.items():
        numeric = [k for k, v in rs[0].summary.items() if isinstance(v, (int, float))]
        out[key] = {k: _mean_std([r.summary[k] for r in rs]) for k in numeric}
    return out


def format_report(results: Sequence[SimResult], regime: str = "") -> str:
    agg = aggregate(results)
    workloads = list(dict.fromkeys(r.workload for r in results))
    policies = list(dict.fromkeys(r.policy for r in results))
    seeds = sorted({r.seed for r in results})
    width = max(22, max(len(p) for p in policies) + 2)

    lines: List[str] = []
    for w in workloads:
        suffix = f", regime: {regime}" if regime else ""
        lines.append(f"\n=== Workload: {w}{suffix}   (mean +/- std over {len(seeds)} seeds) ===")
        header = f"{'POLICY':<{width}}" + "".join(f"{label:>17}" for _, label, _ in REPORT_METRICS)
        lines.append(header)
        lines.append("-" * len(header))
        for p in policies:
            row = f"{p:<{width}}"
            for key, _, fmt in REPORT_METRICS:
                mean, std = agg[(w, p)][key]
                row += f"{fmt.format(mean) + ' ±' + fmt.format(std):>17}"
            lines.append(row)
    lines.append(
        "\nTTFT is for requests that reuse an existing conversation, after warm-up. "
        "'Compr.reads' is the share of block reads served from lossy compressed data. "
        "Simulated on a modelled T4-class GPU; speeds are assumed, not measured."
    )
    return "\n".join(lines)


def results_to_rows(results: Sequence[SimResult], regime: str = "") -> List[dict]:
    return [
        {
            "regime": regime,
            "workload": r.workload,
            "policy": r.policy,
            "seed": r.seed,
            "summary": r.summary,
            "final_state": r.final_state,
        }
        for r in results
    ]


def results_to_json(results: Sequence[SimResult], regime: str = "") -> str:
    return json.dumps(results_to_rows(results, regime), indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare KV-cache policies in the event-driven simulator.")
    parser.add_argument("--workloads", default="all", help=f"comma list from {sorted(WORKLOADS)} or 'all'")
    parser.add_argument("--seeds", type=int, default=3, help="number of held-out seeds, 0..N-1 (default 3)")
    parser.add_argument("--duration", type=float, default=None, help="override trace duration in seconds")
    parser.add_argument("--profile", default=None, choices=sorted(PROFILES), help="serving profile")
    parser.add_argument("--regime", default="default", choices=sorted(REGIMES), help="platform regime")
    parser.add_argument("--set", dest="policy_set", default="basic", choices=["basic", "full"],
                        help="'basic' = Ours + 4 baselines; 'full' = the whole final evaluation")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes (needs the fork start method)")
    parser.add_argument("--out", default=None, help="write raw results to this JSON file")
    args = parser.parse_args(argv)

    names = sorted(WORKLOADS) if args.workloads == "all" else [n.strip() for n in args.workloads.split(",")]
    unknown = [n for n in names if n not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workload(s): {unknown}")
    workloads = [WORKLOADS[n] for n in names]
    if args.duration is not None:
        workloads = [replace(w, duration_s=args.duration) for w in workloads]

    sim_config = REGIMES[args.regime]
    if args.profile:
        sim_config = replace(sim_config, profile=PROFILES[args.profile])

    specs = full_policy_specs() if args.policy_set == "full" else default_policy_specs()
    results = run_experiment(workloads, specs, list(range(args.seeds)), sim_config, processes=args.jobs)
    print(format_report(results, args.regime))
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(results_to_json(results, args.regime))
        print(f"\nRaw results written to {args.out}")


if __name__ == "__main__":
    main()
