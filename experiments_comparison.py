"""
experiments_comparison.py
=========================
Cross-model comparison: the tables that require BOTH examination models side
by side, which neither experiments.py (fixed order only) nor
experiments_dynamic_queue.py (congestion-aware only) can produce alone.

  Table 5 -- device queueing time and examination sojourn time, split by the
             number of examinations a patient needs, fixed order vs.
             congestion-aware routing (SBP, thresholds held constant so the
             routing rule is the only thing that changes).

  Table 6 -- the 2x2 tuning-transfer design
             {examination model} x {threshold pair}, plus the service-level
             decomposition separating the routing effect from the threshold
             effect.

By default each cell is measured the way its own examination model is
reported elsewhere: the fixed order counts the physician queues only, as
in Lv et al.; congestion-aware routing additionally counts station
queueing. The fixed-order cells therefore agree with outputs/ and the
congestion-aware cells with outputs_dynamic_queue/. Pass --wait-metric
classic or extended to force one definition on every cell instead.

IMPORTANT -- THRESHOLD PAIRS
----------------------------
--k-fixed and --k-caware default to the pairs tuned under the OLD
(physician-queueing-only) metric, and are almost certainly not optimal under
the extended metric. Re-run the grid search in each model first:

    python experiments.py --grid-search --jobs 8                  # -> pair A
    python experiments_dynamic_queue.py --grid-search --jobs 8    # -> pair B

then pass them in:

    python experiments_comparison.py --jobs 8 \\
        --k-fixed A1 A2 --k-caware B1 B2

Run order:
    python experiments.py --all                  # fixed order   -> outputs/
    python experiments_dynamic_queue.py --all    # congestion-aware
                                                 #   -> outputs_dynamic_queue/
    python experiments_comparison.py             # this file -> outputs_comparison/

Usage:
    python experiments_comparison.py                 # 100 reps (paper setting)
    python experiments_comparison.py --reps 20 --jobs 8

Outputs -> ./outputs_comparison/
    table5_exam_decomposition.csv
    table6_tuning_transfer.csv
    table6_tuning_transfer_grid.csv
    table6_service_level_decomposition.csv
    per_level_waiting.csv
    routing_effect_by_policy.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

import experiments as exp

OUT_DIR = "outputs_comparison"
BASE_SEED = 1234                 # matches experiments.py

# Defaults: the optima found under the OLD metric. Override on the command
# line once you have re-tuned each model under the extended metric.
DEFAULT_K_FIXED = (3.3, 9.0)
DEFAULT_K_CAWARE = (12.0, 24.0)

# The engine records (n_tests, device_queue_time, exam_sojourn) per examined
# patient in `exam_records`, so this script needs no instrumentation of its
# own. (An earlier version monkey-patched Dispatcher._exam_process to add
# that measurement. The patch duplicated what the engine already does and,
# critically, never wrote the per-patient `dev_queue_time` attribute -- which
# under the extended metric would have silently zeroed every device wait in
# this script alone. It is gone.)
KEEP = ("W_total", "W_init3", "W_init4", "W_follow",
        "delay_rate3", "delay_rate4", "W_device_total", "exam_records")


# Waiting-time definition per cell. "per-model" (the default) ties the
# measure to the examination model, so each cell is measured exactly as that
# model is reported elsewhere: the fixed order under the physician-queue
# definition used by Lv et al., congestion-aware routing under the extended
# definition that includes station queueing. Cells then match outputs/ and
# outputs_dynamic_queue/ exactly.
#
# The alternative, "classic" or "extended" for every cell, holds the measure
# fixed so that the routing rule is the only thing that changes between
# cells. That isolates the routing effect but makes the fixed-order cells
# disagree with outputs/. Choose with --wait-metric.
WAIT_METRIC = "per-model"


def metric_for(policy):
    if WAIT_METRIC == "per-model":
        return "classic" if policy == "paper" else "extended"
    return WAIT_METRIC


def jobs_for(policy, k1, k2, seeds, strategy="SBP", keep=KEEP):
    return [dict(strategy_name=strategy, sim_days=exp.SIM_DAYS, seed=s,
                 k1=k1, k2=k2, exam_order_policy=policy,
                 wait_metric=metric_for(policy), _keep=keep)
            for s in seeds]


def run_cell(policy, k1, k2, seeds, label):
    """One cell of the design: extended W_total, Level III delay rate and the
    per-patient examination records, for every replication."""
    res = exp.run_jobs(jobs_for(policy, k1, k2, seeds),
                       prefix=label, newline_every=50)
    W = np.array([r["W_total"] for r in res])
    d3 = np.array([r["delay_rate3"] * 100.0 for r in res])
    recs = [np.asarray(r["exam_records"], dtype=float) for r in res]
    return W, d3, recs


def per_rep_mean(recs, col, n_tests=None):
    """Mean of column `col` within each replication, optionally restricted to
    patients needing exactly `n_tests` examinations. Averaging per replication
    first keeps the paired t-tests at the replication level."""
    out = []
    for r in recs:
        if r.size == 0:
            continue
        sel = r if n_tests is None else r[r[:, 0] == n_tests]
        if len(sel):
            out.append(sel[:, col].mean())
    return np.array(out)


def table5(fixed_recs, caware_recs):
    QUEUE, SOJ = 1, 2
    all_fixed = np.concatenate([r for r in fixed_recs if r.size])
    rows = []
    for n in (1, 2, 3, 4):
        rows.append(dict(
            Tests=str(n),
            Queue_fixed=per_rep_mean(fixed_recs, QUEUE, n).mean(),
            Queue_caware=per_rep_mean(caware_recs, QUEUE, n).mean(),
            Sojourn_fixed=per_rep_mean(fixed_recs, SOJ, n).mean(),
            Sojourn_caware=per_rep_mean(caware_recs, SOJ, n).mean(),
            Share_pct=(all_fixed[:, 0] == n).mean() * 100.0,
        ))

    qf, qc = per_rep_mean(fixed_recs, QUEUE), per_rep_mean(caware_recs, QUEUE)
    sf, sc = per_rep_mean(fixed_recs, SOJ), per_rep_mean(caware_recs, SOJ)
    rows.append(dict(Tests="All", Queue_fixed=qf.mean(), Queue_caware=qc.mean(),
                     Sojourn_fixed=sf.mean(), Sojourn_caware=sc.mean(),
                     Share_pct=100.0))

    tq, ts = stats.ttest_rel(qc, qf), stats.ttest_rel(sc, sf)
    stats_ = dict(
        queue_change_pct=100 * (qc.mean() - qf.mean()) / qf.mean(),
        queue_t=tq.statistic, queue_p=tq.pvalue,
        sojourn_change_pct=100 * (sc.mean() - sf.mean()) / sf.mean(),
        sojourn_t=ts.statistic, sojourn_p=ts.pvalue,
        queue_share_of_sojourn_pct=100 * qf.mean() / sf.mean(),
        mean_tests=all_fixed[:, 0].mean(),
        share_two_or_more_pct=(all_fixed[:, 0] >= 2).mean() * 100.0,
        p90_sojourn_fixed=np.mean([np.percentile(r[:, SOJ], 90) for r in fixed_recs if r.size]),
        p90_sojourn_caware=np.mean([np.percentile(r[:, SOJ], 90) for r in caware_recs if r.size]),
    )
    return pd.DataFrame(rows), stats_


def table6(W, D3, cells, k_fixed, k_caware):
    base_key = ("paper", *k_fixed)
    base_W, base_D3 = W[base_key], D3[base_key]

    grid = pd.DataFrame(
        [[W[("paper", *k_fixed)].mean(), W[("paper", *k_caware)].mean()],
         [W[("shortest_wait", *k_fixed)].mean(), W[("shortest_wait", *k_caware)].mean()]],
        index=["Fixed order", "Congestion-aware"],
        columns=[str(tuple(k_fixed)), str(tuple(k_caware))])

    decomp, sl = [], []
    for key, label in cells.items():
        if key == base_key:
            continue
        tW = stats.ttest_rel(W[key], base_W)
        decomp.append(dict(Change=label, W_total=W[key].mean(),
                           Delta=(W[key] - base_W).mean(),
                           t=tW.statistic, p=tW.pvalue))
        tD = stats.ttest_rel(D3[key], base_D3)
        sl.append(dict(Change=label, L3_delay_rate_pct=D3[key].mean(),
                       SL3_change_pts=-(D3[key] - base_D3).mean(),
                       t=tD.statistic, p=tD.pvalue))

    return grid, pd.DataFrame(decomp), pd.DataFrame(sl)


def per_level_and_routing_effect(seeds, k_fixed, k_caware):
    """For each policy, run both examination models on the same seeds at that
    model's own thresholds, and report per-level waiting times plus the paired
    effect of switching the routing rule."""
    rows, eff = [], []
    for policy in ("IFP", "ALT", "SBP"):
        runs = {}
        for tag, pol, (k1, k2) in (("Fixed order", "paper", k_fixed),
                                   ("Congestion-aware", "shortest_wait", k_caware)):
            runs[tag] = exp.run_jobs(
                jobs_for(pol, k1, k2, seeds, strategy=policy),
                prefix=f"{policy} / {tag}", newline_every=50)
            rows.append(dict(
                Routing=tag, Policy=policy,
                W_level3=np.mean([r["W_init3"] for r in runs[tag]]),
                W_level4=np.mean([r["W_init4"] for r in runs[tag]]),
                W_device=np.mean([r["W_device_total"] for r in runs[tag]]),
                W_followup_queue=np.mean([r["W_follow"] for r in runs[tag]]),
                W_total=np.mean([r["W_total"] for r in runs[tag]]),
            ))
        a = np.array([r["W_total"] for r in runs["Fixed order"]])
        b = np.array([r["W_total"] for r in runs["Congestion-aware"]])
        t = stats.ttest_rel(b, a)
        eff.append(dict(Policy=policy, W_fixed=a.mean(), W_caware=b.mean(),
                        Delta=(b - a).mean(), t=t.statistic, p=t.pvalue))

    lvl, eff = pd.DataFrame(rows), pd.DataFrame(eff)
    lvl.to_csv(os.path.join(OUT_DIR, "per_level_waiting.csv"),
               index=False, float_format="%.4f")
    eff.to_csv(os.path.join(OUT_DIR, "routing_effect_by_policy.csv"),
               index=False, float_format="%.4f")

    print("\n=== Per-level waiting time (min, per-model metric) ===")
    print(lvl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n=== Effect of the routing rule on each policy (paired) ===")
    print(eff.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return lvl, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the replications (default 1)")
    ap.add_argument("--wait-metric", choices=["per-model", "classic", "extended"],
                    default="per-model",
                    help="per-model (default): fixed-order cells use the "
                         "physician-queue definition, congestion-aware cells "
                         "the extended one, so every cell matches outputs/ or "
                         "outputs_dynamic_queue/. classic/extended: one "
                         "definition everywhere, which isolates the routing "
                         "effect but no longer matches those folders.")
    ap.add_argument("--k-fixed", type=float, nargs=2, default=None,
                    metavar=("K1", "K2"),
                    help="threshold pair tuned under FIXED-order routing")
    ap.add_argument("--k-caware", type=float, nargs=2, default=None,
                    metavar=("K1", "K2"),
                    help="threshold pair tuned under CONGESTION-AWARE routing")
    args = ap.parse_args()

    global WAIT_METRIC
    WAIT_METRIC = args.wait_metric
    exp.JOBS = max(1, args.jobs)
    if WAIT_METRIC == "per-model":
        print("Waiting-time metric: per examination model "
              "(fixed order = classic, congestion-aware = extended)\n")
    else:
        print(f"Waiting-time metric for all cells: {WAIT_METRIC}\n")
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = [BASE_SEED + i for i in range(args.reps)]

    missing = [n for n, v in (("--k-fixed", args.k_fixed),
                              ("--k-caware", args.k_caware)) if v is None]
    k_fixed = tuple(args.k_fixed) if args.k_fixed else DEFAULT_K_FIXED
    k_caware = tuple(args.k_caware) if args.k_caware else DEFAULT_K_CAWARE
    if missing:
        print(f"NOTE: {' and '.join(missing)} not given, so the built-in "
              f"default(s) are used. Those defaults were tuned under the OLD "
              f"physician-queueing-only metric -- run the grid search in each "
              f"model and pass the pairs explicitly.\n")

    cells = {
        ("paper", *k_fixed):          f"fixed order / {k_fixed}",
        ("shortest_wait", *k_fixed):  f"congestion-aware / {k_fixed}",
        ("paper", *k_caware):         f"fixed order / {k_caware}",
        ("shortest_wait", *k_caware): f"congestion-aware / {k_caware}",
    }

    W, D3, R = {}, {}, {}
    for key, label in cells.items():
        policy, k1, k2 = key
        W[key], D3[key], R[key] = run_cell(policy, k1, k2, seeds, label)
        print(f"{label:34s} W_total = {W[key].mean():7.3f}  "
              f"(SD {W[key].std(ddof=1):5.2f})")

    t5, s5 = table5(R[("paper", *k_fixed)], R[("shortest_wait", *k_fixed)])
    t5.to_csv(os.path.join(OUT_DIR, "table5_exam_decomposition.csv"),
              index=False, float_format="%.4f")

    # The headline statistics behind Table 5 -- these are quoted directly in
    # the write-up, so they are written to disk as well as printed.
    pd.DataFrame([
        dict(Quantity="Device queueing change (%)", Value=s5["queue_change_pct"],
             t=s5["queue_t"], p=s5["queue_p"]),
        dict(Quantity="Examination sojourn change (%)", Value=s5["sojourn_change_pct"],
             t=s5["sojourn_t"], p=s5["sojourn_p"]),
        dict(Quantity="Device queueing as share of sojourn, fixed order (%)",
             Value=s5["queue_share_of_sojourn_pct"], t=float("nan"), p=float("nan")),
        dict(Quantity="Mean examinations per examined patient",
             Value=s5["mean_tests"], t=float("nan"), p=float("nan")),
        dict(Quantity="Examined patients needing >=2 tests (%)",
             Value=s5["share_two_or_more_pct"], t=float("nan"), p=float("nan")),
        dict(Quantity="Sojourn p90, fixed order (min)",
             Value=s5["p90_sojourn_fixed"], t=float("nan"), p=float("nan")),
        dict(Quantity="Sojourn p90, congestion-aware (min)",
             Value=s5["p90_sojourn_caware"], t=float("nan"), p=float("nan")),
    ]).to_csv(os.path.join(OUT_DIR, "table5_exam_summary_stats.csv"),
              index=False, float_format="%.6g")

    # Mean W_total of every cell of the design, in one file.
    pd.DataFrame([
        dict(Routing=("Fixed order" if pol == "paper" else "Congestion-aware"),
             k1=k1, k2=k2, W_total_mean=W[key].mean(),
             W_total_SD=W[key].std(ddof=1), N=len(W[key]),
             L3_delay_rate_pct=D3[key].mean())
        for key in cells for (pol, k1, k2) in [key]
    ]).to_csv(os.path.join(OUT_DIR, "cell_means.csv"),
              index=False, float_format="%.4f")

    print(f"\n=== Table 5: examination decomposition (SBP, thresholds at {k_fixed}) ===")
    print(t5.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nDevice queueing {s5['queue_change_pct']:+.2f}%  "
          f"(t = {s5['queue_t']:.2f}, p = {s5['queue_p']:.3e})")
    print(f"Exam sojourn    {s5['sojourn_change_pct']:+.2f}%  "
          f"(t = {s5['sojourn_t']:.2f}, p = {s5['sojourn_p']:.3e})")
    print(f"Queueing share of sojourn (fixed order) = "
          f"{s5['queue_share_of_sojourn_pct']:.3f}%")
    print(f"Mean tests = {s5['mean_tests']:.3f}; "
          f">=2 tests = {s5['share_two_or_more_pct']:.1f}%; "
          f"sojourn p90 {s5['p90_sojourn_fixed']:.2f} -> "
          f"{s5['p90_sojourn_caware']:.2f}")

    per_level_and_routing_effect(seeds, k_fixed, k_caware)

    grid, decomp, sl = table6(W, D3, cells, k_fixed, k_caware)
    grid.to_csv(os.path.join(OUT_DIR, "table6_tuning_transfer_grid.csv"),
                float_format="%.4f")
    decomp.to_csv(os.path.join(OUT_DIR, "table6_tuning_transfer.csv"),
                  index=False, float_format="%.4f")
    sl.to_csv(os.path.join(OUT_DIR, "table6_service_level_decomposition.csv"),
              index=False, float_format="%.4f")

    print("\n=== Table 6: tuning transfer (W_total, min) ===")
    print(grid.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nDecomposition against (fixed order, {k_fixed}):")
    print(decomp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nService-level decomposition (Level III):")
    print(sl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved -> {OUT_DIR}/  (8 CSV files, including "
          f"table5_exam_summary_stats.csv and cell_means.csv)")


if __name__ == "__main__":
    main()