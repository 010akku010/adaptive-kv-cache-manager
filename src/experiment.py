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

Usage
-----
    python -m src.experiment                       # all workloads, 3 seeds
    python -m src.experiment --workloads zipf,bursty --seeds 5
    python -m src.experiment --out results/sim.json

This reports *simulated* results from a model of a T4-class GPU serving a
~1B-parameter model. Hardware speeds are assumed typical values (see
src/hardware.py), not measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.eventsim import EventSimulator, SimConfig, SimResult
from src.hardware import PROFILES
from src.policies import PolicySpec, default_policy_specs
from src.trace import WORKLOADS, WorkloadConfig, generate_trace

# Metrics shown in the report table: (key, label, format).
REPORT_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("reuse_ttft_p50_ms", "TTFT P50(ms)", "{:.2f}"),
    ("reuse_ttft_p99_ms", "TTFT P99(ms)", "{:.1f}"),
    ("gpu_hit_ratio", "GPU hit", "{:.3f}"),
    ("cache_hit_ratio", "Cache hit", "{:.3f}"),
    ("recompute_ratio", "Recompute", "{:.3f}"),
    ("forced_evictions", "Forced ev.", "{:.0f}"),
)


def run_experiment(
    workloads: Sequence[WorkloadConfig],
    specs: Sequence[PolicySpec],
    seeds: Sequence[int],
    sim_config: Optional[SimConfig] = None,
) -> List[SimResult]:
    """Run the full workload x policy x seed matrix."""
    sim_config = sim_config or SimConfig()
    results: List[SimResult] = []
    for workload in workloads:
        for seed in seeds:
            trace = generate_trace(workload.with_seed(seed))  # one trace, shared by all policies
            for spec in specs:
                results.append(EventSimulator(trace, sim_config, spec).run())
    return results


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


def format_report(results: Sequence[SimResult]) -> str:
    agg = aggregate(results)
    workloads = list(dict.fromkeys(r.workload for r in results))
    policies = list(dict.fromkeys(r.policy for r in results))
    seeds = sorted({r.seed for r in results})

    lines: List[str] = []
    for w in workloads:
        lines.append(f"\n=== Workload: {w}   (mean +/- std over {len(seeds)} seeds) ===")
        header = f"{'POLICY':<16}" + "".join(f"{label:>17}" for _, label, _ in REPORT_METRICS)
        lines.append(header)
        lines.append("-" * len(header))
        for p in policies:
            row = f"{p:<16}"
            for key, _, fmt in REPORT_METRICS:
                mean, std = agg[(w, p)][key]
                row += f"{fmt.format(mean) + ' ±' + fmt.format(std):>17}"
            lines.append(row)
    lines.append(
        "\nTTFT is for requests that reuse an existing conversation, after warm-up. "
        "Simulated on a modelled T4-class GPU; speeds are assumed, not measured."
    )
    return "\n".join(lines)


def results_to_json(results: Sequence[SimResult]) -> str:
    rows = [
        {"workload": r.workload, "policy": r.policy, "seed": r.seed, "summary": r.summary, "final_state": r.final_state}
        for r in results
    ]
    return json.dumps(rows, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare KV-cache policies in the event-driven simulator.")
    parser.add_argument("--workloads", default="all", help=f"comma list from {sorted(WORKLOADS)} or 'all'")
    parser.add_argument("--seeds", type=int, default=3, help="number of random seeds (default 3)")
    parser.add_argument("--duration", type=float, default=None, help="override trace duration in seconds")
    parser.add_argument("--profile", default=None, choices=sorted(PROFILES), help="serving profile")
    parser.add_argument("--out", default=None, help="write raw results to this JSON file")
    args = parser.parse_args(argv)

    names = sorted(WORKLOADS) if args.workloads == "all" else [n.strip() for n in args.workloads.split(",")]
    unknown = [n for n in names if n not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workload(s): {unknown}")
    workloads = [WORKLOADS[n] for n in names]
    if args.duration is not None:
        workloads = [replace(w, duration_s=args.duration) for w in workloads]

    sim_config = SimConfig()
    if args.profile:
        sim_config = replace(sim_config, profile=PROFILES[args.profile])

    results = run_experiment(workloads, default_policy_specs(), list(range(args.seeds)), sim_config)
    print(format_report(results))
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(results_to_json(results))
        print(f"\nRaw results written to {args.out}")


if __name__ == "__main__":
    main()
