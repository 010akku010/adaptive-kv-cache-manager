"""
plots.py

Figures for the report (matplotlib is an optional dependency, imported lazily).

* ``plot_frontier``           latency vs compressed-read exposure, one panel
                              per workload: our tuned engine at several quality
                              budgets against the static compress-first ladder.
* ``plot_tuner_trajectory``   how the tuner moves its quality penalty to hold a
                              compressed-read budget over one run (two stacked
                              panels sharing the time axis; no dual axis).

Usage:
    python -m src.plots results/full_host_starved.json --out results/figures
"""

from __future__ import annotations

import argparse
import os
import statistics
from typing import List, Optional, Sequence, Tuple

from src.analysis import Row, load_rows, policies_in

# Validated categorical slots 1-2 (blue, orange) on a light surface, plus neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OURS = "#2a78d6"
LADDER = "#eb6834"


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting: pip install matplotlib") from exc
    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _point(rows: Sequence[Row], policy: str, workload: str, regime: str) -> Optional[Tuple[float, float, float]]:
    """(mean exposure, mean TTFT, std TTFT over seeds) for one policy, or None."""
    sel = [r for r in rows if r["policy"] == policy and r["workload"] == workload and r.get("regime", "") == regime]
    if not sel:
        return None
    ttft = [r["summary"]["reuse_ttft_mean_ms"] for r in sel]
    exp = [r["summary"]["compressed_read_ratio"] for r in sel]
    return statistics.fmean(exp), statistics.fmean(ttft), (statistics.stdev(ttft) if len(ttft) > 1 else 0.0)


def plot_frontier(rows: Sequence[Row], regime: str, path: str) -> str:
    plt = _pyplot()
    workloads = list(dict.fromkeys(r["workload"] for r in rows if r.get("regime", "") == regime))
    pols = policies_in(rows)
    tuned = [p for p in pols if p.startswith("Ours-ms+tuner")]
    hosted = [p for p in pols if p.startswith("Ours-ms+host+tuner")]
    ladder = [p for p in pols if p.startswith("Ladder")]

    fig, axes = plt.subplots(1, len(workloads), figsize=(3.6 * len(workloads), 3.6), facecolor=SURFACE)
    axes = [axes] if len(workloads) == 1 else list(axes)
    for ax, w in zip(axes, workloads):
        _style(ax)
        base = _point(rows, "LRU+offload", w, regime)

        def series(names: List[str]):
            pts = [(_point(rows, n, w, regime)) for n in names]
            return sorted(p for p in pts if p is not None)

        lad = series(ladder)
        if base:
            lad = sorted(lad + [base])
        ours = series(tuned)
        host = series(hosted)
        for pts, color, label, dash in (
            (lad, LADDER, "Static ladder", "-"),
            (ours, OURS, "Tuned engine", "-"),
            (host, OURS, "Tuned engine + host price", (0, (4, 2))),
        ):
            if not pts:
                continue
            xs, ys, es = zip(*pts)
            ax.errorbar(xs, ys, yerr=es, color=color, linewidth=1.6, linestyle=dash, marker="o", markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, capsize=0, elinewidth=0.8,
                        ecolor=color, label=label)
        if base:
            ax.plot([base[0]], [base[1]], marker="s", color=INK_2, markersize=5, linestyle="none",
                    label="LRU+offload (no compression)")
        ax.set_title(w, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("share of reads served from compressed data", fontsize=8, color=INK_2)
        ax.set_xlim(left=-0.03)
    axes[0].set_ylabel("mean TTFT, reused requests (ms)", fontsize=8, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        f"Latency vs quality exposure ({regime} regime). Lower and to the left is better. "
        "Bars: std over seeds.",
        fontsize=9, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_tuner_trajectory(path: str, workload: str = "zipf", seed: int = 0, budget: float = 0.2) -> str:
    """Run one host-starved simulation and plot the tuner's behaviour over time."""
    from src.eventsim import EventSimulator, SimConfig
    from src.policies import ms_adaptive_spec
    from src.trace import WORKLOADS, generate_trace
    from src.tuner import TunerConfig

    sim = EventSimulator(
        generate_trace(WORKLOADS[workload].with_seed(seed)),
        SimConfig(cpu_pool_mb=512.0),
        ms_adaptive_spec("tuned", tuner_config=TunerConfig(max_compressed_read_ratio=budget)),
    )
    sim.run()
    tuner = getattr(sim.policy, "tuner", None)
    assert tuner is not None, "policy was built without a tuner"
    hist = tuner.history

    plt = _pyplot()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, facecolor=SURFACE)
    for ax in (top, bottom):
        _style(ax)
    t = [h.time_s for h in hist]
    top.plot(t, [h.stats.compressed_ratio for h in hist], color=OURS, linewidth=1.6)
    top.axhline(budget, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    top.text(t[0], budget, f" budget {budget:.2f}", color=INK_2, fontsize=8, va="bottom", ha="left")
    top.set_ylabel("compressed-read share\n(per 10 s window)", fontsize=8, color=INK_2)
    bottom.plot(t, [h.quality_penalty_ms for h in hist], color=LADDER, linewidth=1.6, drawstyle="steps-post")
    bottom.set_ylabel("quality penalty (ms)", fontsize=8, color=INK_2)
    bottom.set_xlabel("simulated time (s)", fontsize=8, color=INK_2)
    fig.suptitle(
        f"Tuner holding a {budget:.0%} compressed-read budget ({workload}, host-starved, seed {seed})",
        fontsize=9, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the report figures from saved results.")
    parser.add_argument("results", help="JSON file written by src.experiment (host-starved regime recommended)")
    parser.add_argument("--out", default="results/figures", help="output directory")
    parser.add_argument("--no-trajectory", action="store_true", help="skip the (slow) tuner trajectory run")
    args = parser.parse_args(argv)
    rows = load_rows([args.results])
    regime = rows[0].get("regime", "")
    print(plot_frontier(rows, regime, os.path.join(args.out, f"frontier_{regime or 'run'}.png")))
    if not args.no_trajectory:
        print(plot_tuner_trajectory(os.path.join(args.out, "tuner_trajectory.png")))


if __name__ == "__main__":
    main()
