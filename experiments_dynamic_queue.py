"""
experiments_dynamic_queue.py
=============================
Runs the SAME table/figure pipeline as experiments.py, but with a
real-world-style extension to the examination subsystem:

  PAPER'S RULE (Section 2.1.3 / Eq. 4):
    A patient needing multiple exams (e.g. X-ray + CT) processes them in a
    FIXED order -- longest report-delay first -- decided once, up front.

  THIS SCRIPT'S EXTENSION ("shortest_wait" exam_order_policy):
    Before joining EACH exam's queue, the patient dynamically checks all of
    their remaining required exam types and picks whichever device
    currently has the smallest expected wait:

        expected_wait(device) = (patients ahead at that device) x (that
                                  device's per-patient service time)

    "Patients ahead" = 1 if a patient is currently being served there
    (capacity=1) + however many are already waiting in that device's
    queue. This decision is re-made after every exam finishes, so a
    patient's exam order can be reshuffled mid-journey. This models how a
    real patient/nurse would actually route between Lab / Ultrasound /
    X-ray / CT -- by checking which line is shorter right now.

TWO CHANGES AT ONCE
-------------------
This script differs from experiments.py in TWO ways, deliberately, because
together they form "the extension":

  1. WAITING-TIME METRIC: wait_metric="extended" -- time spent queueing for
     a busy diagnostic device (rho_j) counts as waiting. experiments.py
     reports the paper's definition (physician queueing only), under which
     that same queue exists and delays the patient but is not counted.

  2. EXAM ROUTING: exam_order_policy="shortest_wait" -- the patient joins
     whichever remaining device has the shortest expected wait, instead of
     a fixed order fixed up front.

Because both change together, the gap between ./outputs/ and
./outputs_dynamic_queue/ is the COMBINED effect. To separate them, run
either script with --wait-metric to hold the measurement constant, e.g.

    python experiments.py --all --wait-metric extended   # routing effect only
    python experiments_dynamic_queue.py --all --wait-metric classic

The routing logic itself lives in ed_simulation.py's
Dispatcher._exam_process (see the `exam_order_policy` parameter); nothing
there needed to change. This script re-runs the full experiment battery
with exam_order_policy="shortest_wait", saves everything to a SEPARATE
output folder, and builds side-by-side comparison figures.

USAGE
-----
Run experiments.py FIRST if you want --compare-plots to work (it loads the
fixed-order PNGs to build the side-by-side view).

    # everything, on 8 cores
    python experiments_dynamic_queue.py --all --jobs 8

    # skip the grid search and reuse a known (k1,k2) for THIS policy:
    python experiments_dynamic_queue.py --compare --sensitivity-arrival \\
        --sensitivity-staffing --compare-plots --k1 5.0 --k2 12.0

    # just rebuild the side-by-side images from existing outputs:
    python experiments_dynamic_queue.py --compare-plots

NOTE: each examination model has its OWN optimal (k1,k2). Do not carry a
threshold pair tuned under one model over to the other without saying so --
experiments_comparison.py exists precisely to quantify that transfer.

Outputs:
    ./outputs_dynamic_queue/   <- tables & figures for the shortest_wait policy
    ./outputs_comparison/      <- compare_figure3/4/5_*.png (fixed | dynamic)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments as exp

BASE_OUT_DIR = "outputs"                # experiments.py's fixed-order results
NEW_OUT_DIR = "outputs_dynamic_queue"   # this script's results
COMPARISON_DIR = "outputs_comparison"   # side-by-side figures

# Redirect the shared pipeline: every job built by experiments.py now carries
# exam_order_policy="shortest_wait", and every output lands in a separate
# folder so nothing in ./outputs/ is touched.
exp.EXAM_ORDER_POLICY = "shortest_wait"
exp.WAIT_METRIC = "extended"      # this is the extension: device queueing counts
exp.OUT_DIR = NEW_OUT_DIR


# Panel headings on the side-by-side figures. Edit these two lines to
# change the wording everywhere.
LEFT_TITLE = "Fixed"
RIGHT_TITLE = "Congestion-aware"

# Palette sampled from the reference figure. Edit here to restyle.
LEVEL_COLOURS = {
    "Level III":    "#FFFFBF",   # cream
    "Level IV":     "#FD8E59",   # orange
    "All Patients": "#91BEDD",   # blue
}
POLICY_STYLE = {
    "IFP": ("#4E8FC0", "-",  "o"),
    "ALT": ("#FD8E59", "--", "^"),
    "SBP": ("#73C475", ":",  "s"),
}

# Metric wording under the y-axis label of each panel.
LEFT_METRIC = "physician queueing only"
RIGHT_METRIC = "physician + device queueing"

PANEL_BG = "#EFEFEF"


def _style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(axis="y", color="white", linewidth=1.1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _panel_titles():
    return LEFT_TITLE, RIGHT_TITLE


def _finish(fig, out_name):
    out_path = os.path.join(COMPARISON_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[compare-plots] Saved -> {out_path}")


def _boxplot_panel(ax, df, title):
    strategies = ["IFP", "ALT", "SBP"]
    levels = list(LEVEL_COLOURS)
    positions = np.arange(len(strategies)) * 4
    step, width = 1.0, 0.78          # step > width leaves a gap between boxes
    handles = []
    for i, lvl in enumerate(levels):
        data = [df[(df.Strategy == s) & (df.Level == lvl)].Value.values
                for s in strategies]
        bp = ax.boxplot(data, positions=positions + (i - 1) * step,
                        widths=width, patch_artist=True, showmeans=False,
                        medianprops=dict(color="#333333", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor="none",
                                        markeredgecolor="#777777"))
        for patch in bp["boxes"]:
            patch.set_facecolor(LEVEL_COLOURS[lvl])
            patch.set_edgecolor("#333333")
        handles.append(bp["boxes"][0])
    ax.set_xticks(positions)
    ax.set_xticklabels(strategies)
    ax.set_title(title, pad=8)
    _style_axes(ax)
    return handles, levels


def _line_panel(ax, df, xcol, order, title):
    handles, labels = [], []
    for strat, (colour, ls, marker) in POLICY_STYLE.items():
        sub = df[df.Strategy == strat].set_index(xcol).loc[order]
        line, = ax.plot(range(len(order)), sub.MeanWait.values, color=colour,
                        linestyle=ls, marker=marker, label=strat,
                        linewidth=2.2, markersize=7)
        handles.append(line)
        labels.append(strat)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(o) for o in order])
    ax.set_title(title, pad=8)
    _style_axes(ax)
    return handles, labels


def compare_plots():
    """Redraw the three side-by-side figures from the saved data.

    Drawing rather than stitching keeps a single set of axes per panel, so
    both panels share one y-axis range, there are no nested titles, and the
    fonts follow the settings in experiments.py.
    """
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    left, right = _panel_titles()

    def load(fname):
        a = os.path.join(BASE_OUT_DIR, fname)
        b = os.path.join(NEW_OUT_DIR, fname)
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"[compare-plots] SKIP {fname}: run both experiment "
                  f"scripts first.")
            return None, None
        return pd.read_csv(a), pd.read_csv(b)

    # --- waiting time by strategy ---
    da, db = load("figure3_waiting_data.csv")
    if da is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
        handles, labels = _boxplot_panel(axes[0], da, left)
        _boxplot_panel(axes[1], db, right)
        axes[0].set_ylabel(f"Average waiting time (min)\n({LEFT_METRIC})")
        axes[1].set_ylabel(f"({RIGHT_METRIC})")
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _finish(fig, "compare_figure3_waiting_time_boxplot.png")

    # --- arrival-rate sensitivity ---
    da, db = load("figure4_arrival_sensitivity_data.csv")
    if da is not None:
        order = sorted(da.ArrivalChangePct.unique())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
        handles, labels = _line_panel(axes[0], da, "ArrivalChangePct", order, left)
        _line_panel(axes[1], db, "ArrivalChangePct", order, right)
        for ax in axes:
            ax.set_xlabel("Arrival rate change (%)")
        axes[0].set_ylabel(f"Average waiting time (min)\n({LEFT_METRIC})")
        axes[1].set_ylabel(f"({RIGHT_METRIC})")
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _finish(fig, "compare_figure4_arrival_sensitivity.png")

    # --- staffing sensitivity ---
    da, db = load("figure5_staffing_sensitivity_data.csv")
    if da is not None:
        order = [f"S{i}" for i in range(7)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
        handles, labels = _line_panel(axes[0], da, "Scenario", order, left)
        _line_panel(axes[1], db, "Scenario", order, right)
        for ax in axes:
            ax.set_xlabel("Scenario")
        axes[0].set_ylabel(f"Average waiting time (min)\n({LEFT_METRIC})")
        axes[1].set_ylabel(f"({RIGHT_METRIC})")
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _finish(fig, "compare_figure5_staffing_sensitivity.png")


def main():
    parser = exp.build_parser()
    parser.add_argument("--compare-plots", action="store_true",
                        help="build side-by-side (fixed | congestion-aware) "
                             "figures into ./outputs_comparison/")
    args = parser.parse_args()

    action_flags = exp.ACTION_FLAGS + ("compare_plots",)
    if not any(getattr(args, f) for f in action_flags):
        args.all = True

    exp.apply_common_args(args)
    os.makedirs(NEW_OUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    k1_opt, k2_opt = args.k1, args.k2

    if args.all or args.table2:
        exp.log("=== [congestion-aware] STEP: Table 2 export ===")
        exp.export_table2()

    if args.all or args.grid_search:
        exp.log("=== [congestion-aware] STEP: SBP grid search (Tables 3 & 4) ===")
        k1_opt, k2_opt = exp.grid_search_sbp()
        exp.log(f"Optimal SBP parameters under congestion-aware routing: "
                f"k1={k1_opt}, k2={k2_opt}")
    else:
        exp.log(f"Grid search skipped -- using k1={k1_opt}, k2={k2_opt} as given. "
                "Remember that a pair tuned under fixed-order routing is not "
                "optimal here.")

    if args.all or args.compare:
        exp.log("=== [congestion-aware] STEP: Comparative analysis "
                "(Figure 3, Tables 5-10) ===")
        exp.comparative_analysis(k1_opt, k2_opt)

    if args.all or args.sensitivity_arrival:
        exp.log("=== [congestion-aware] STEP: Arrival-rate sensitivity (Figure 4) ===")
        exp.sensitivity_arrival(k1_opt, k2_opt)

    if args.all or args.sensitivity_staffing:
        exp.log("=== [congestion-aware] STEP: Staffing sensitivity (Figure 5) ===")
        exp.sensitivity_staffing(k1_opt, k2_opt)

    if args.all or args.compare_plots:
        exp.log("=== STEP: Building fixed-vs-congestion-aware comparison plots ===")
        compare_plots()

    exp.log(f"ALL DONE. Congestion-aware outputs in ./{NEW_OUT_DIR}/, "
            f"comparison figures in ./{COMPARISON_DIR}/")


if __name__ == "__main__":
    main()