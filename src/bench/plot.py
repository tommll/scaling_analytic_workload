"""Charts + a Markdown table view of results.json.

Palette is the validated 4-slot categorical set (adjacent CVD dE 9.1, normal-vision
22.9). Two of the slots sit under 3:1 contrast on the light surface, so every mark
carries a visible direct label and RESULTS.md carries the full table -- the relief
rule, not an afterthought.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "savefig.facecolor": SURFACE,
})


def _frame(ax, xgrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def barh(ax, y, width, height, color):
    """One thin bar, anchored at the baseline. Deliberately a plain rectangle:
    a rounded-cap patch measured in data units distorts badly when the x and y
    ranges differ by three orders of magnitude, which is the normal case here."""
    if width > 0:
        ax.barh(y, width, height, color=color, zorder=3)


# --------------------------------------------------------------------------- #
def chart_speedup(res, path):
    """Headline: does adding nodes actually make it faster?

    Three lines on one axis: ideal linear (what the marketing says), the host's
    measured parallel ceiling (what this machine can physically deliver), and the
    measured Spark speedup. The gap between line 1 and line 2 is hardware; the gap
    between line 2 and line 3 is the software's coordination cost.
    """
    keys = sorted(k for k in res["experiments"] if k.startswith("strong"))
    series = []
    for k in keys:
        rows = [r for r in res["experiments"][k]
                if r["tier"] == "spark" and r["outcome"] == "ok"]
        if len(rows) > 1:
            base = rows[0]["compute_s"]
            series.append((k, [r["nodes"] for r in rows],
                           [base / r["compute_s"] for r in rows],
                           # Error bars from the fastest and slowest timed run. The
                           # host runs a desktop alongside the benchmark; showing the
                           # spread is more honest than picking a kind estimator.
                           [base / r["compute_max"] for r in rows],
                           [base / r["compute_min"] for r in rows],
                           rows[0]["rows_in"]))
    cal = res["experiments"].get("calibrate", [])
    if not series:
        return None
    nodes = series[0][1]
    cal_x = [c["nodes"] for c in cal if c["nodes"] <= max(nodes)]
    cal_y = [c["throughput_x"] for c in cal if c["nodes"] <= max(nodes)]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _frame(ax)
    ax.plot(nodes, nodes, ls=(0, (4, 3)), lw=1.5, color=MUTED, zorder=2)
    ax.annotate("ideal linear", (nodes[-1], nodes[-1]), textcoords="offset points",
                xytext=(-4, 6), ha="right", color=MUTED, fontsize=9)
    if cal_x:
        ax.plot(cal_x, cal_y, lw=2, color=ORANGE, marker="o", ms=8,
                mec=SURFACE, mew=2, zorder=4, label="host ceiling (measured)")
        ax.annotate(f"{cal_y[-1]:.1f}x", (cal_x[-1], cal_y[-1]),
                    textcoords="offset points", xytext=(8, -2), color=INK, fontsize=10,
                    fontweight="bold")
    for (key, xs, ys, lo, hi, n), color in zip(series, (BLUE, AQUA, YELLOW)):
        ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)],
                                  [h - y for y, h in zip(ys, hi)]],
                    lw=2, color=color, marker="o", ms=8, mec=SURFACE, mew=2,
                    ecolor=color, elinewidth=1.2, capsize=4, alpha=1.0,
                    zorder=5, label=f"Spark, {n / 1e6:.0f}M rows")
        ax.annotate(f"{ys[-1]:.2f}x", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(9, -3), color=INK, fontsize=10, fontweight="bold")

    ax.set_xticks(nodes)
    ax.set_xlabel("nodes (1 pinned core + 2 GB each)")
    ax.set_ylabel("speedup vs 1 node")
    ax.set_title("Strong scaling: the bigger the job, the better it scales", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(0.5, -0.02, "Median of 3 timed runs, bars show min-max. Compute time "
             "only; startup charged separately.\nHost ceiling = this machine's own "
             "measured parallel limit, from `make calibrate`.",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _biggest_strong(res):
    keys = sorted((k for k in res["experiments"] if k.startswith("strong")),
                  key=lambda k: max((r.get("rows_in", 0) for r in res["experiments"][k]),
                                    default=0))
    return res["experiments"][keys[-1]] if keys else []


def chart_tiers(res, path):
    """Magnitude comparison across execution models, including the ones that died."""
    rows = _biggest_strong(res)
    if not rows:
        return None
    labels = [r["label"] for r in rows]
    vals = [r["compute_s"] if r["outcome"] == "ok" else 0 for r in rows]
    ok = [r["outcome"] == "ok" for r in rows]
    colors = [BLUE if r["tier"] == "spark" else AQUA if r["tier"] == "chunked"
              else YELLOW for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 0.44 * len(rows) + 1.6))
    _frame(ax, xgrid=True)
    span = max(vals) or 1
    for i, (v, c, good) in enumerate(zip(vals, colors, ok)):
        y = len(rows) - 1 - i
        if good:
            barh(ax, y, v, 0.46, c)
            ax.text(v + span * 0.015, y, f"{v:.1f}s", va="center", color=INK,
                    fontsize=10, fontweight="bold")
        else:
            ax.text(span * 0.015, y, "out of memory - killed", va="center",
                    color=ORANGE, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=9.5)
    ax.set_xlim(0, span * 1.22)
    ax.set_xlabel("compute time (s), median of timed runs - lower is better")
    n = next((r["rows_in"] for r in rows if r["outcome"] == "ok"), 0)
    ax.set_title(f"Same pipeline, {len(rows)} execution models ({n:,} rows)", pad=14)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_overhead(res, path):
    """What distribution charges you: a fixed startup cost that never shrinks."""
    rows = [r for r in _biggest_strong(res) if r["tier"] == "spark" and r["outcome"] == "ok"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(rows) + 1.8))
    _frame(ax, xgrid=True)
    span = max(r["startup_s"] + r["compute_s"] for r in rows)
    gap = span * 0.004  # 2px-ish surface gap between stacked segments
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, r["compute_s"], 0.44, color=BLUE, zorder=3)
        ax.barh(y, r["startup_s"], 0.44, left=r["compute_s"] + gap, color=ORANGE, zorder=3)
        ax.text(r["compute_s"] / 2, y, f"{r['compute_s']:.0f}s", va="center",
                ha="center", color="#ffffff", fontsize=9.5, fontweight="bold")
        ax.text(r["compute_s"] + gap + r["startup_s"] / 2, y, f"{r['startup_s']:.0f}s",
                va="center", ha="center", color="#ffffff", fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['nodes']} node{'s' if r['nodes'] > 1 else ''}"
                        for r in reversed(rows)])
    ax.set_xlabel("seconds")
    ax.set_ylim(-0.8, len(rows) - 0.3)
    ax.set_title("The bill for distribution: startup never gets cheaper", pad=14)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE), plt.Rectangle((0, 0), 1, 1, fc=ORANGE)]
    ax.legend(handles, ["compute (scales with nodes)", "startup (fixed cost)"],
              frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_memory(res, path):
    """The capability difference, which is not a speed difference at all."""
    rows = res["experiments"].get("memory", [])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 0.52 * len(rows) + 1.8))
    _frame(ax, xgrid=True)
    peaks = [(r.get("peak_rss_mb") or 0) for r in rows]
    span = max(peaks + [2048]) * 1.25
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r["outcome"] == "ok":
            v = r.get("peak_rss_mb") or 0
            barh(ax, y, v, 0.44, AQUA)
            txt = (f"{v:.0f} MB peak - survived" if v
                   else "survived - streams and spills across nodes")
            ax.text((v or 0) + span * 0.02, y, txt, va="center", color=INK,
                    fontsize=9.5, fontweight="bold")
        else:
            barh(ax, y, 2048, 0.44, ORANGE)
            ax.text(2048 + span * 0.015, y, "OOM-killed (exit 137)", va="center",
                    color=ORANGE, fontsize=9.5, fontweight="bold")
    ax.axvline(2048, color=MUTED, ls=(0, (4, 3)), lw=1.5, zorder=4)
    ax.annotate("2 GB node limit", (2048, -0.62), textcoords="offset points",
                xytext=(-6, 0), ha="right", va="center", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=9.5)
    ax.set_xlim(0, span)
    ax.set_xlabel("peak resident memory (MB)")
    n = next((r["rows_in"] for r in rows if r["outcome"] == "ok"), 0)
    ax.set_ylim(-0.8, len(rows) - 0.3)
    ax.set_title(f"Memory ceiling at {n:,} rows: capability, not speed", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def table(rows, cols, headers) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c(r) for c in cols) + " |")
    return "\n".join(out)


def write_tables(res, path):
    """The table view. Required relief for the low-contrast palette slots, and the
    thing anyone reproducing these numbers will actually want."""
    L = []
    ex = res["experiments"]
    L.append("# Results (table view)\n")
    L.append(f"Host: {res['host']['cpu_model']} - {res['host']['cpus']} logical CPUs. "
             f"Node = {res['host']['node']}.\n")

    if "calibrate" in ex:
        L.append("\n## Host calibration - the machine's own parallel ceiling\n")
        L.append(table(ex["calibrate"],
                       [lambda r: str(r["nodes"]), lambda r: f"{r['wall_s']:.2f}",
                        lambda r: f"{r['throughput_x']:.2f}x"],
                       ["pinned nodes", "wall (s)", "aggregate throughput"]))
    for key, title in (*[(k, f"Strong scaling - {k.split('@')[-1]} dataset")
                         for k in sorted(ex) if k.startswith("strong")],
                       ("small", "Small data - the crossover"),
                       ("memory", "Memory ceiling"),
                       ("weak", "Weak scaling - data grows with the cluster")):
        if key not in ex:
            continue
        L.append(f"\n\n## {title}\n")
        L.append(table(ex[key],
                       [lambda r: r["label"],
                        lambda r: f"{r['rows_in']:,}" if r["outcome"] == "ok" else "-",
                        lambda r: (f"{r['compute_s']:.1f}" if r["outcome"] == "ok"
                                   else "**OOM**"),
                        lambda r: (f"{r['compute_min']:.1f}-{r['compute_max']:.1f}"
                                   if r["outcome"] == "ok" else "-"),
                        lambda r: f"{r['wall_s']:.1f}" if r["outcome"] == "ok" else "-",
                        lambda r: (f"{r['rows_per_s']:,}" if r["outcome"] == "ok" else "-"),
                        lambda r: (f"{r['peak_rss_mb']:.0f}"
                                   if r.get("peak_rss_mb") else "-")],
                       ["configuration", "rows", "compute median (s)",
                        "compute min-max (s)", "wall (s)", "rows/s",
                        "peak RSS (MB)"]))
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def main(results_path: str, out_dir: str):
    res = json.load(open(results_path))
    os.makedirs(out_dir, exist_ok=True)
    made = [chart_speedup(res, f"{out_dir}/strong_scaling.png"),
            chart_tiers(res, f"{out_dir}/tiers.png"),
            chart_overhead(res, f"{out_dir}/overhead.png"),
            chart_memory(res, f"{out_dir}/memory_ceiling.png"),
            write_tables(res, f"{out_dir}/RESULTS.md")]
    for m in made:
        print("wrote", m) if m else None


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.json",
         sys.argv[2] if len(sys.argv) > 2 else "results")
