"""
analysis.py

Statistics on saved experiment results: paired comparisons with confidence
intervals, and the latency-vs-quality frontier comparison.

Why paired?  For a given (workload, seed) every policy replays the *same*
trace, so the difference between two policies on that trace removes the
(large) trace-to-trace variation. Averaging those per-trace differences and
putting a t-based 95% interval on them is the right test for "is policy A
better than policy B", far more sensitive than comparing two noisy means.

Why a frontier?  Compression buys latency by serving some reads from lossy
data, and the simulator cannot measure accuracy. So two policies are only
comparable *at equal exposure* (share of reads served compressed). Each
policy family therefore traces a curve of (exposure, latency); we compare
our engine's curve with the static ladder's curve by linear interpolation.
This is an approximation, and the table says so.

Usage:
    python -m src.analysis results/full_default.json results/full_host_starved.json
"""

from __future__ import annotations

import json
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Two-sided 95% critical values of Student's t by degrees of freedom.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
}

Row = dict


def t_critical(df: int) -> float:
    """95% two-sided t critical value (conservative: the next-lower tabulated df)."""
    if df < 1:
        raise ValueError("need at least 2 paired observations")
    keys = [k for k in sorted(_T_CRIT_95) if k <= df]
    return _T_CRIT_95[keys[-1]] if df <= 30 else 1.96


def load_rows(paths: Sequence[str]) -> List[Row]:
    rows: List[Row] = []
    for path in paths:
        with open(path) as fh:
            rows.extend(json.load(fh))
    return rows


def paired_difference(
    rows: Sequence[Row], policy_a: str, policy_b: str, metric: str, workload: str, regime: str = "",
) -> Tuple[float, float, int]:
    """
    Mean of (a - b) over seeds, with the half-width of its 95% interval.
    Returns (mean_difference, half_width, n_pairs). Negative means a is lower
    (for latency: a is faster).
    """
    def pick(policy: str) -> Dict[int, float]:
        return {
            r["seed"]: r["summary"][metric]
            for r in rows
            if r["policy"] == policy and r["workload"] == workload and r.get("regime", "") == regime
        }

    a, b = pick(policy_a), pick(policy_b)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise ValueError(f"need >= 2 common seeds for {policy_a} vs {policy_b} on {workload}/{regime}")
    diffs = [a[s] - b[s] for s in seeds]
    mean = statistics.fmean(diffs)
    half = t_critical(len(diffs) - 1) * statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, half, len(diffs)


def mean_metric(rows: Sequence[Row], policy: str, metric: str, workload: str, regime: str = "") -> float:
    vals = [
        r["summary"][metric]
        for r in rows
        if r["policy"] == policy and r["workload"] == workload and r.get("regime", "") == regime
    ]
    if not vals:
        raise ValueError(f"no rows for {policy}/{workload}/{regime}")
    return statistics.fmean(vals)


def interpolate(points: Sequence[Tuple[float, float]], x: float) -> Optional[float]:
    """Piecewise-linear interpolation of y at x; None outside the sampled range."""
    pts = sorted(points)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return (y0 + y1) / 2
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def frontier_points(rows: Sequence[Row], policies: Sequence[str], workload: str, regime: str) -> List[Tuple[float, float]]:
    """(mean compressed-read share, mean TTFT) for each policy that has data."""
    pts = []
    for p in policies:
        try:
            pts.append((
                mean_metric(rows, p, "compressed_read_ratio", workload, regime),
                mean_metric(rows, p, "reuse_ttft_mean_ms", workload, regime),
            ))
        except ValueError:
            continue
    return pts


def policies_in(rows: Sequence[Row]) -> List[str]:
    return list(dict.fromkeys(r["policy"] for r in rows))


def format_frontier(rows: Sequence[Row], workload: str, regime: str) -> List[str]:
    """Our tuned engines (with and without the host price) vs the static ladder at equal exposure."""
    all_p = policies_in(rows)
    ladder = [p for p in all_p if p.startswith("Ladder")] + ["LRU+offload"]  # LRU+offload = 0% exposure
    ours = [p for p in all_p if p.startswith("Ours-ms+tuner") or p.startswith("Ours-ms+host+tuner")]
    ladder_pts = frontier_points(rows, ladder, workload, regime)
    lines = []
    for p in ours:
        try:
            x = mean_metric(rows, p, "compressed_read_ratio", workload, regime)
            y = mean_metric(rows, p, "reuse_ttft_mean_ms", workload, regime)
        except ValueError:
            continue
        ref = interpolate(ladder_pts, x)
        if ref is None:
            lines.append(f"  {p:<26} exposure {x:5.2f}  TTFT {y:7.1f} ms   (outside the ladder's range)")
        else:
            lines.append(
                f"  {p:<26} exposure {x:5.2f}  TTFT {y:7.1f} ms   ladder at same exposure {ref:7.1f} ms"
                f"   gap {y - ref:+7.1f} ms ({100 * (y - ref) / ref:+5.1f}%)"
            )
    return lines


def format_report(rows: Sequence[Row]) -> str:
    regimes = list(dict.fromkeys(r.get("regime", "") for r in rows))
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    pols = policies_in(rows)
    out: List[str] = []
    candidates = ["Ours-ms", "Ours-ms+tuner b=0.2", "Ours-ms+host+tuner b=0.2"]
    baselines = ["Ours-v1", "LRU+offload", "MTDS-style", "Ladder 0.25"]
    for regime in regimes:
        for w in workloads:
            out.append(f"\n##### regime: {regime or '-'}   workload: {w}")
            out.append("Paired difference in mean reuse TTFT (ms), candidate - baseline, 95% interval over seeds "
                       "(negative = candidate faster; * = interval excludes 0):")
            for cand in candidates:
                for base in baselines:
                    if cand not in pols or base not in pols:
                        continue
                    try:
                        m, h, n = paired_difference(rows, cand, base, "reuse_ttft_mean_ms", w, regime)
                    except ValueError:
                        continue
                    sig = "*" if abs(m) > h else " "
                    out.append(f"  {cand:<26} vs {base:<12} {m:+8.2f} ± {h:6.2f}  (n={n}) {sig}")
            out.append("Tuned engine vs the static ladder at equal compressed-read exposure (interpolated):")
            out.extend(format_frontier(rows, w, regime) or ["  (no data)"])
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    paths = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        raise SystemExit("usage: python -m src.analysis RESULTS.json [RESULTS2.json ...]")
    print(format_report(load_rows(paths)))


if __name__ == "__main__":
    main()

