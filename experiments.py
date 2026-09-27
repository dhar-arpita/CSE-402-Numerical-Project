"""
experiments.py
===============
Runs everything needed to reproduce the paper's tables & figures using the
engine in ed_simulation.py -- now under the EXTENDED waiting-time metric
(physician queueing + diagnostic-device queueing; see ed_simulation.py
revision notes v3).

Everything reported here, and everything the grid search optimizes and
constrains, uses that single metric:

    min  W_total(k1, k2)          over the extended wait
    s.t. SL3 >= SL_MIN_3, SL4 >= SL_MIN_4      also on the extended wait

Usage:
    python experiments.py --all
    python experiments.py --all --jobs 8          # use 8 CPU cores
    python experiments.py --all --smoke --jobs 8  # ~minutes, for a sanity check
    python experiments.py --grid-search
    python experiments.py --compare --k1 5 --k2 12
    python experiments.py --sensitivity-arrival
    python experiments.py --sensitivity-staffing

All outputs (CSV tables + PNG figures) are written to ./outputs/.

Dependencies: simpy, numpy, pandas, scipy, matplotlib
    pip install simpy numpy pandas scipy matplotlib

NOTE: a full reproduction (100 replications x full grid search) is
computationally heavy in pure Python. Use --jobs N to spread the
replications across CPU cores, and --smoke first to check the pipeline
end-to-end before committing to a long run.
"""

import os
import sys
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")           # headless-safe; no display needed
import matplotlib.pyplot as plt

def setup_fonts(font_dir="fonts"):
    """Register any .ttf/.otf files in ./fonts/ and prefer Book Antiqua.

    Drop the unzipped Book Antiqua files into a folder called `fonts` next
    to this script. If none are found, matplotlib falls back to Palatino
    and then to its default serif, so the figures still build.
    """
    from matplotlib import font_manager
    if os.path.isdir(font_dir):
        for fn in os.listdir(font_dir):
            if fn.lower().endswith((".ttf", ".otf")):
                try:
                    font_manager.fontManager.addfont(os.path.join(font_dir, fn))
                except Exception:
                    pass
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Book Antiqua", "Palatino Linotype",
                                  "Palatino", "URW Palladio L", "P052",
                                  "DejaVu Serif"]


setup_fonts()

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

from ed_simulation import run_simulation, TABLE2, DAYS


# ---------------------------------------------------------------------------
# Module-level configuration (overridable from the command line, and
# overwritten wholesale by experiments_dynamic_queue.py)
# ---------------------------------------------------------------------------
OUT_DIR = "outputs"
EXAM_ORDER_POLICY = "paper"  # "paper" (fixed order) | "shortest_wait" (dynamic)

# Which definition of waiting time the headline numbers use.
#   "classic"  = the paper's own Eq. 24-28: physician queueing only.
#   "extended" = physician queueing + diagnostic-device queueing (rho_j).
# This file reproduces the PAPER, so it defaults to "classic".
# experiments_dynamic_queue.py overrides it to "extended".
# Override either way with --wait-metric.
WAIT_METRIC = "classic"
JOBS = 1                     # worker processes; 1 = serial

SIM_DAYS = 7                 # one operational week, per paper
N_REPS_FINAL = 100           # paper uses 100 replications for the main comparison
N_REPS_GRID = 20             # replications per (k1,k2) candidate during grid search
N_REPS_SENSITIVITY = 20      # replications per point in the sensitivity sweeps

SL_MIN_3 = 0.90              # minimum service-level constraint (Eq. 32), extended metric
SL_MIN_4 = 0.90

BASE_SEED = 1234             # master seed; per-replication seeds derived from this

# Coarse grid (phase 1) and fine grid (phase 2) geometry
K1_RANGE = (0, 30)
K2_RANGE = (0, 40)
COARSE_STEP = 1.0
FINE_STEP = 0.1
FINE_HALFWIDTH = 1.0         # fine grid spans best +/- this, at FINE_STEP


def apply_smoke_settings():
    """Shrink every dimension so the whole pipeline finishes in minutes.

    Useful to verify the plumbing (outputs, figures, CSV schemas) before
    launching a multi-hour run. Results from a smoke run are NOT usable as
    findings -- far too few replications and far too coarse a grid.
    """
    global N_REPS_FINAL, N_REPS_GRID, N_REPS_SENSITIVITY, COARSE_STEP, FINE_HALFWIDTH
    N_REPS_FINAL = 5
    N_REPS_GRID = 3
    N_REPS_SENSITIVITY = 3
    COARSE_STEP = 10.0
    FINE_HALFWIDTH = 0.3


# Only these keys are shipped back from worker processes, to keep pickling
# cheap (a full result dict carries every patient's raw waiting time).
KEEP_GRID = ("W_total", "SL3", "SL4")
KEEP_MAIN = ("W_init3", "W_init4", "W_total", "SL3", "SL4",
             "delay_rate3", "delay_rate4",
             "W_consult3", "W_consult4", "W_consult_total",
             "W_device3", "W_device4", "W_device_total",
             "device_share3", "device_share4", "W_follow",
             "phys_utilization", "device_utilization", "device_exam_count",
             "n_completed", "exam_records")
KEEP_WAIT = ("W_total",)


# ---------------------------------------------------------------------------
# Progress logging helper
# ---------------------------------------------------------------------------
_progress_start = None


def progress(current, total, prefix="", extra="", newline_every=None):
    """Print a live progress line to the terminal (overwrites itself with \\r)."""
    global _progress_start
    if _progress_start is None or current <= 1:
        _progress_start = time.time()
    elapsed = time.time() - _progress_start
    pct = 100.0 * current / total if total else 100.0
    rate = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else 0.0
    line = (f"[{prefix}] {current}/{total} ({pct:5.1f}%) | "
            f"elapsed {elapsed:7.1f}s | ETA {eta:7.1f}s")
    if extra:
        line += f" | {extra}"
    end = "\n" if (newline_every and current % newline_every == 0) else "\r"
    sys.stdout.write(line + (" " * 10) + end)
    sys.stdout.flush()
    if current >= total:
        print()


def reset_progress():
    global _progress_start
    _progress_start = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metric_label():
    return ("physician queueing only (paper)" if WAIT_METRIC == "classic"
            else "physician + device queueing")


def rep_seeds(n, base=BASE_SEED):
    return [base + i for i in range(n)]


# ---------------------------------------------------------------------------
# Job plumbing: one "job" == one replication, described by a plain dict so it
# can be shipped to a worker process.
#
# The examination-ordering policy and the waiting-time definition travel
# INSIDE each job rather than being read from a module global at run time. That matters: on Windows/macOS,
# worker processes are spawned (not forked) and re-import this module fresh,
# which would reset any module-level override. Putting the policy in the job
# keeps experiments_dynamic_queue.py correct under every start method.
# ---------------------------------------------------------------------------

def sim_job(strategy, seed, keep, **kw):
    job = dict(strategy_name=strategy, sim_days=SIM_DAYS, seed=seed,
               exam_order_policy=EXAM_ORDER_POLICY, wait_metric=WAIT_METRIC,
               _keep=tuple(keep))
    job.update(kw)
    return job


def _sim_job(job):
    job = dict(job)
    keep = job.pop("_keep", None)
    res = run_simulation(**job)
    if keep is not None:
        res = {k: res[k] for k in keep}
    return res


def _sim_job_idx(pair):
    idx, job = pair
    return idx, _sim_job(job)


def run_jobs(jobs, prefix="", newline_every=25):
    """Run a list of replication jobs, serially or across JOBS processes.

    Results come back in the same order as `jobs` regardless of the number
    of workers, and each replication is seeded individually, so the output
    is identical whether you use --jobs 1 or --jobs 16.
    """
    total = len(jobs)
    out = [None] * total
    reset_progress()
    if JOBS and JOBS > 1 and total > 1:
        import multiprocessing as mp
        chunk = max(1, min(32, total // (JOBS * 8) or 1))
        with mp.Pool(processes=JOBS) as pool:
            for i, (idx, res) in enumerate(
                    pool.imap_unordered(_sim_job_idx, list(enumerate(jobs)),
                                        chunksize=chunk), start=1):
                out[idx] = res
                progress(i, total, prefix=prefix, newline_every=newline_every)
    else:
        for i, job in enumerate(jobs, start=1):
            out[i - 1] = _sim_job(job)
            progress(i, total, prefix=prefix, newline_every=newline_every)
    return out


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# Table 2 (arrival rates) -- just dump what's hard-coded in ed_simulation.py
# ---------------------------------------------------------------------------
def export_table2():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(TABLE2, index=[f"{h}-{(h+1)%24}" for h in range(24)])
    df.index.name = "Hours"
    df = df[DAYS]
    df.to_csv(os.path.join(OUT_DIR, "table2_arrival_rates.csv"))
    print("Saved Table 2 -> table2_arrival_rates.csv")
    return df


# ---------------------------------------------------------------------------
# Grid search for SBP thresholds (Tables 3 & 4), on the EXTENDED metric
# ---------------------------------------------------------------------------
def ci95(mean, std, n):
    if n <= 1 or np.isnan(std):
        return (float("nan"), float("nan"))
    half = 1.96 * std / np.sqrt(n)
    return (mean - half, mean + half)


def aggregate_candidate(k1, k2, reps):
    """Collapse one candidate's replications into a summary row."""
    totals = np.array([r["W_total"] for r in reps], dtype=float)
    sl3 = np.array([r["SL3"] for r in reps], dtype=float)
    sl4 = np.array([r["SL4"] for r in reps], dtype=float)
    return dict(
        k1=k1, k2=k2,
        mean_wait=float(np.nanmean(totals)),
        std_wait=float(np.nanstd(totals, ddof=1)) if len(totals) > 1 else float("nan"),
        SL3=float(np.nanmean(sl3)),
        SL4=float(np.nanmean(sl4)),
    )


def evaluate_grid(combos, seeds, prefix):
    """Run every (k1,k2) x seed combination as one flat batch of jobs."""
    jobs = [sim_job("SBP", s, KEEP_GRID, k1=float(k1), k2=float(k2))
            for (k1, k2) in combos for s in seeds]
    log(f"{prefix}: {len(combos)} (k1,k2) combinations x {len(seeds)} reps "
        f"= {len(jobs)} simulations"
        + (f"  [{JOBS} workers]" if JOBS > 1 else "  [serial]"))
    flat = run_jobs(jobs, prefix=prefix, newline_every=200)
    rows = [aggregate_candidate(k1, k2, reps)
            for (k1, k2), reps in zip(combos, chunked(flat, len(seeds)))]
    return rows


def pick_best(rows, stage):
    """Lowest mean extended wait among the pairs meeting both SL constraints."""
    feasible = [r for r in rows if r["SL3"] >= SL_MIN_3 and r["SL4"] >= SL_MIN_4]
    log(f"{stage}: {len(feasible)}/{len(rows)} combinations met the SL "
        f"constraint (SL3 >= {SL_MIN_3}, SL4 >= {SL_MIN_4}) on the extended metric.")
    if not feasible:
        log("WARNING: no (k1,k2) pair satisfied the service-level constraint. "
            "Falling back to the unconstrained best so the search can "
            "continue. Consider lowering SL_MIN_3/SL_MIN_4, raising the "
            "replication count, or checking whether the system is simply "
            "overloaded at this staffing level.")
        feasible = rows
    return feasible


def grid_search_sbp():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Phase 1: coarse grid ----
    k1_vals = np.round(np.arange(K1_RANGE[0], K1_RANGE[1] + 1e-9, COARSE_STEP), 3)
    k2_vals = np.round(np.arange(K2_RANGE[0], K2_RANGE[1] + 1e-9, COARSE_STEP), 3)
    combos = list(itertools.product(k1_vals, k2_vals))
    seeds = rep_seeds(N_REPS_GRID)

    coarse_rows = evaluate_grid(combos, seeds, "Coarse grid")
    coarse_df = (pd.DataFrame(pick_best(coarse_rows, "Phase 1 (coarse)"))
                 .sort_values("mean_wait").reset_index(drop=True))
    coarse_top5 = coarse_df.head(5).copy()
    coarse_top5["CI95"] = coarse_top5.apply(
        lambda r: ci95(r["mean_wait"], r["std_wait"], N_REPS_GRID), axis=1)
    coarse_top5.to_csv(os.path.join(OUT_DIR, "table3_coarse_grid_search.csv"), index=False)
    pd.DataFrame(coarse_rows).to_csv(
        os.path.join(OUT_DIR, "table3_coarse_grid_search_full.csv"), index=False)
    print("Saved Table 3 -> table3_coarse_grid_search.csv (+ _full.csv, every cell)")
    print(coarse_top5.to_string(index=False))

    best_k1, best_k2 = float(coarse_top5.iloc[0]["k1"]), float(coarse_top5.iloc[0]["k2"])

    # ---- Phase 2: fine grid around the coarse optimum ----
    k1_fine = np.round(np.arange(max(0.0, best_k1 - FINE_HALFWIDTH),
                                 best_k1 + FINE_HALFWIDTH + 1e-9, FINE_STEP), 1)
    k2_fine = np.round(np.arange(max(0.0, best_k2 - FINE_HALFWIDTH),
                                 best_k2 + FINE_HALFWIDTH + 1e-9, FINE_STEP), 1)
    fine_combos = list(itertools.product(k1_fine, k2_fine))
    fine_seeds = rep_seeds(N_REPS_FINAL)

    fine_rows = evaluate_grid(fine_combos, fine_seeds, "Fine grid")
    fine_df = (pd.DataFrame(pick_best(fine_rows, "Phase 2 (fine)"))
               .sort_values("mean_wait").reset_index(drop=True))
    fine_top5 = fine_df.head(5).copy()
    fine_top5["CI95"] = fine_top5.apply(
        lambda r: ci95(r["mean_wait"], r["std_wait"], N_REPS_FINAL), axis=1)
    fine_top5.to_csv(os.path.join(OUT_DIR, "table4_fine_grid_search.csv"), index=False)
    pd.DataFrame(fine_rows).to_csv(
        os.path.join(OUT_DIR, "table4_fine_grid_search_full.csv"), index=False)
    print("Saved Table 4 -> table4_fine_grid_search.csv (+ _full.csv, every cell)")
    print(fine_top5.to_string(index=False))

    best = fine_top5.iloc[0]
    return float(best["k1"]), float(best["k2"])


# ---------------------------------------------------------------------------
# Comparative analysis: Figure 3, Tables 5-10
# ---------------------------------------------------------------------------
def run_all_strategies(k1_opt, k2_opt, n_reps=None):
    n_reps = n_reps or N_REPS_FINAL
    seeds = rep_seeds(n_reps)  # SAME seeds across strategies -> paired tests
    jobs = []
    for s in seeds:
        jobs.append(sim_job("IFP", s, KEEP_MAIN))
        jobs.append(sim_job("ALT", s, KEEP_MAIN))
        jobs.append(sim_job("SBP", s, KEEP_MAIN, k1=k1_opt, k2=k2_opt))
    log(f"Comparative analysis: {n_reps} paired replications x 3 strategies "
        f"= {len(jobs)} simulations (SBP uses k1={k1_opt}, k2={k2_opt})")
    flat = run_jobs(jobs, prefix="Comparative", newline_every=30)
    results = {"IFP": flat[0::3], "ALT": flat[1::3], "SBP": flat[2::3]}
    log("Comparative analysis simulations done.")
    return results, seeds


def figure3_boxplot(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    strategies = ["IFP", "ALT", "SBP"]
    positions = np.arange(len(strategies)) * 4
    colors = {"Level III": "tab:orange", "Level IV": "tab:blue", "All Patients": "tab:green"}
    width = 0.9
    for i, level_key in enumerate(["W_init3", "W_init4", "W_total"]):
        label = {"W_init3": "Level III", "W_init4": "Level IV",
                 "W_total": "All Patients"}[level_key]
        data = [[r[level_key] for r in results[s]] for s in strategies]
        pos = positions + (i - 1) * width
        bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                        showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[label])
        bp["boxes"][0].set_label(label)
    ax.set_xticks(positions)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Average waiting time (min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figure3_waiting_time_boxplot.png"), dpi=150)
    plt.close(fig)

    # the plotted values, so the side-by-side figure can be redrawn cleanly
    rows = []
    for strat in strategies:
        for key, label in [("W_init3", "Level III"), ("W_init4", "Level IV"),
                           ("W_total", "All Patients")]:
            for v in [r[key] for r in results[strat]]:
                rows.append(dict(Strategy=strat, Level=label, Value=v))
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "figure3_waiting_data.csv"), index=False)
    print("Saved Figure 3 -> figure3_waiting_time_boxplot.png (+ _data.csv)")


def table_raw_means(results):
    """Each strategy's mean/SD/95% CI for the extended W_total."""
    rows = []
    for strat in ["IFP", "ALT", "SBP"]:
        x = np.array([r["W_total"] for r in results[strat]])
        n = len(x)
        mean, sd = x.mean(), x.std(ddof=1) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se) if n > 1 else (float("nan"),) * 2
        rows.append(dict(Strategy=strat, Mean_Wait=mean, SD=sd, N=n,
                         CI95_low=ci[0], CI95_high=ci[1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "table_raw_means.csv"), index=False)
    print("Saved -> table_raw_means.csv")
    print(df.to_string(index=False))
    return df


def table5_6_significance(results):
    pairs = [("IFP", "ALT"), ("IFP", "SBP"), ("ALT", "SBP")]
    alpha_adj = 0.05 / len(pairs)

    rows_t, rows_d = [], []
    for a, b in pairs:
        xa = np.array([r["W_total"] for r in results[a]])
        xb = np.array([r["W_total"] for r in results[b]])
        diff = xa - xb
        t_stat, p_val = stats.ttest_rel(xa, xb)
        rows_t.append(dict(Comparison=f"{a} vs. {b}", Mean_Diff=diff.mean(),
                           t_value=t_stat, p_value=p_val,
                           Significant=bool(p_val < alpha_adj)))

        # Cohen's d for PAIRED data: mean difference / SD of the differences.
        d_std = diff.std(ddof=1)
        d = diff.mean() / d_std if d_std > 0 else float("nan")
        interp = ("small" if abs(d) < 0.5 else
                  "medium" if abs(d) < 0.8 else
                  "large" if abs(d) < 1.2 else "huge")
        rows_d.append(dict(Comparison=f"{a} vs. {b}", Cohens_d=d, Interpretation=interp))

    t_df, d_df = pd.DataFrame(rows_t), pd.DataFrame(rows_d)
    t_df.to_csv(os.path.join(OUT_DIR, "table5_paired_ttest.csv"), index=False)
    d_df.to_csv(os.path.join(OUT_DIR, "table6_cohens_d.csv"), index=False)
    print("Saved Table 5 -> table5_paired_ttest.csv")
    print(t_df.to_string(index=False))
    print("Saved Table 6 -> table6_cohens_d.csv")
    print(d_df.to_string(index=False))
    return t_df, d_df


def table7_service_level(results):
    rows = []
    for strat in ["IFP", "ALT", "SBP"]:
        for level, dr_key, sl_key in [(3, "delay_rate3", "SL3"), (4, "delay_rate4", "SL4")]:
            drs = np.array([r[dr_key] for r in results[strat]])
            sls = np.array([r[sl_key] for r in results[strat]])
            rows.append(dict(
                Strategy=strat, Level=f"Level {'III' if level == 3 else 'IV'}",
                Delay_Rate_Mean=100 * np.nanmean(drs),
                Delay_Rate_SD=100 * np.nanstd(drs, ddof=1),
                Service_Level_Mean=100 * np.nanmean(sls),
                Service_Level_SD=100 * np.nanstd(sls, ddof=1),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "table7_delay_service_level.csv"), index=False)
    print("Saved Table 7 -> table7_delay_service_level.csv")
    print(df.to_string(index=False))
    return df


def table8_physician_utilization(results):
    rows = []
    for strat in ["IFP", "ALT", "SBP"]:
        u = np.array([r["phys_utilization"] for r in results[strat]]) * 100
        rows.append(dict(Strategy=strat, Mean=u.mean(), SD=u.std(ddof=1),
                         Min=u.min(), Max=u.max(), Median=np.median(u)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "table8_physician_utilization.csv"), index=False)
    print("Saved Table 8 -> table8_physician_utilization.csv")
    print(df.to_string(index=False))
    return df


def table9_device_utilization(results):
    rows = []
    for name in ["lab", "ultrasound", "xray", "ct"]:
        utils, counts = [], []
        for strat in ["IFP", "ALT", "SBP"]:
            for r in results[strat]:
                utils.append(r["device_utilization"][name] * 100)
                counts.append(r["device_exam_count"][name])
        rows.append(dict(
            Equipment=name.upper(),
            Avg_Exams_Per_Week=np.mean(counts),
            Utilization_Mean=np.mean(utils),
            Utilization_SD=np.std(utils, ddof=1),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "table9_device_utilization.csv"), index=False)
    print("Saved Table 9 -> table9_device_utilization.csv")
    print(df.to_string(index=False))
    return df


def table10_wait_decomposition(results):
    """Split a patient's queueing time into its two components.

    Consult_Queue_Wait  -- waiting for a physician (initial + follow-up queue)
    Device_Queue_Wait   -- waiting for an occupied diagnostic device (rho_j)
    Headline_Wait       -- what this run actually reports as "waiting time":
                           Consult only under wait_metric="classic" (the
                           paper's definition), Consult + Device under
                           "extended".

    Under "classic" the device column is therefore measured but NOT included
    in Headline_Wait -- that is exactly the quantity the paper leaves out,
    shown here so the two runs can be compared line by line.
    """
    rows = []
    for strat in ["IFP", "ALT", "SBP"]:
        R = results[strat]
        for label, ck, dk, hk in [("Level III", "W_consult3", "W_device3", "W_init3"),
                                  ("Level IV", "W_consult4", "W_device4", "W_init4"),
                                  ("All patients", "W_consult_total", "W_device_total", "W_total")]:
            consult = np.nanmean([r[ck] for r in R])
            device = np.nanmean([r[dk] for r in R])
            headline = np.nanmean([r[hk] for r in R])
            rows.append(dict(
                Strategy=strat, Level=label, Metric=WAIT_METRIC,
                Consult_Queue_Wait=consult,
                Device_Queue_Wait=device,
                Headline_Wait=headline,
                Device_Counted=(WAIT_METRIC == "extended"),
                Device_Share_pct=(100.0 * device / headline
                                  if WAIT_METRIC == "extended" and headline else float("nan")),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "table10_wait_decomposition.csv"), index=False)
    print("Saved Table 10 -> table10_wait_decomposition.csv")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def table11_exam_decomposition(results, strategy="SBP"):
    """Split the diagnostic stage by how many examinations a patient needs.

    Two quantities per group:
      Device_Queue_Wait -- time queueing for a busy station, summed over the
                           patient's tests. This is w^dev, the component the
                           routing rule acts on.
      Exam_Sojourn      -- from entering the diagnostic stage until every
                           result is available. Contains the queueing wait
                           plus the processing times and the reporting
                           delays, so it is NOT a waiting-time measure; it
                           shows how much of the stage the rule can touch.

    Averaged within each replication first, then across replications, so the
    row means are comparable to the paired tests elsewhere.
    """
    recs = [np.asarray(r["exam_records"], dtype=float) for r in results[strategy]]
    recs = [r for r in recs if r.size]
    if not recs:
        print("No examination records to decompose.")
        return None
    allrec = np.concatenate(recs)

    def per_rep(col, n_tests=None):
        out = []
        for r in recs:
            sel = r if n_tests is None else r[r[:, 0] == n_tests]
            if len(sel):
                out.append(sel[:, col].mean())
        return np.array(out)

    rows = []
    for n in (1, 2, 3, 4):
        q, sj = per_rep(1, n), per_rep(2, n)
        rows.append(dict(Tests=str(n),
                         Device_Queue_Wait=q.mean() if q.size else float("nan"),
                         Exam_Sojourn=sj.mean() if sj.size else float("nan"),
                         Share_pct=100.0 * (allrec[:, 0] == n).mean()))
    q, sj = per_rep(1), per_rep(2)
    rows.append(dict(Tests="All", Device_Queue_Wait=q.mean(),
                     Exam_Sojourn=sj.mean(), Share_pct=100.0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "table11_exam_decomposition.csv"),
              index=False, float_format="%.4f")

    extra = pd.DataFrame([
        dict(Quantity="Device queueing as share of exam sojourn (%)",
             Value=100.0 * q.mean() / sj.mean()),
        dict(Quantity="Mean examinations per examined patient",
             Value=allrec[:, 0].mean()),
        dict(Quantity="Examined patients needing >=2 tests (%)",
             Value=100.0 * (allrec[:, 0] >= 2).mean()),
        dict(Quantity="Exam sojourn, 90th percentile (min)",
             Value=float(np.mean([np.percentile(r[:, 2], 90) for r in recs]))),
        dict(Quantity="Replications", Value=float(len(recs))),
    ])
    extra.to_csv(os.path.join(OUT_DIR, "table11_exam_summary_stats.csv"),
                 index=False, float_format="%.6g")

    print(f"Saved Table 11 ({strategy}) -> table11_exam_decomposition.csv "
          f"(+ _summary_stats.csv)")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(extra.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def comparative_analysis(k1_opt, k2_opt):
    os.makedirs(OUT_DIR, exist_ok=True)
    results, seeds = run_all_strategies(k1_opt, k2_opt, n_reps=N_REPS_FINAL)
    figure3_boxplot(results)
    table_raw_means(results)
    table5_6_significance(results)
    table7_service_level(results)
    table8_physician_utilization(results)
    table9_device_utilization(results)
    table10_wait_decomposition(results)
    table11_exam_decomposition(results)
    return results


# ---------------------------------------------------------------------------
# Sensitivity analysis: Figure 4 (arrival rate) & Figure 5 (staffing)
# ---------------------------------------------------------------------------
def sensitivity_arrival(k1_opt, k2_opt):
    os.makedirs(OUT_DIR, exist_ok=True)
    factors_pct = list(range(-15, 16, 3))
    seeds = rep_seeds(N_REPS_SENSITIVITY, base=BASE_SEED + 500)
    cells = [(pct, strat) for pct in factors_pct for strat in ["IFP", "ALT", "SBP"]]

    jobs = []
    for pct, strat in cells:
        kw = dict(k1=k1_opt, k2=k2_opt) if strat == "SBP" else {}
        for s in seeds:
            jobs.append(sim_job(strat, s, KEEP_WAIT,
                                arrival_factor=1 + pct / 100.0, **kw))
    log(f"Arrival-rate sensitivity: {len(factors_pct)} arrival levels x 3 "
        f"strategies x {len(seeds)} reps = {len(jobs)} simulations")
    flat = run_jobs(jobs, prefix="Arrival sensitivity", newline_every=50)

    rows = []
    for (pct, strat), reps in zip(cells, chunked(flat, len(seeds))):
        waits = [r["W_total"] for r in reps]
        rows.append(dict(ArrivalChangePct=pct, Strategy=strat,
                         MeanWait=float(np.nanmean(waits)),
                         SD=float(np.nanstd(waits, ddof=1)) if len(waits) > 1 else float("nan")))
    log("Arrival-rate sensitivity simulations done.")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "figure4_arrival_sensitivity_data.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    for strat, style in [("IFP", "-o"), ("ALT", "--^"), ("SBP", ":s")]:
        sub = df[df.Strategy == strat].sort_values("ArrivalChangePct")
        ax.plot(sub.ArrivalChangePct, sub.MeanWait, style, label=strat,
                linewidth=2.2, markersize=8)
    ax.set_xlabel("Arrival rate change (%)")
    ax.set_ylabel("Average waiting time (min)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figure4_arrival_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("Saved Figure 4 -> figure4_arrival_sensitivity.png")
    return df


def sensitivity_staffing(k1_opt, k2_opt):
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = {
        "S0": (5, 5, 3), "S1": (5, 5, 4), "S2": (5, 5, 5), "S3": (5, 6, 5),
        "S4": (5, 7, 5), "S5": (6, 7, 5), "S6": (7, 7, 5),
    }
    seeds = rep_seeds(N_REPS_SENSITIVITY, base=BASE_SEED + 900)
    cells = [(name, strat) for name in scenarios for strat in ["IFP", "ALT", "SBP"]]

    jobs = []
    for name, strat in cells:
        kw = dict(k1=k1_opt, k2=k2_opt) if strat == "SBP" else {}
        for s in seeds:
            jobs.append(sim_job(strat, s, KEEP_WAIT,
                                shift_capacities=scenarios[name], **kw))
    log(f"Staffing sensitivity: {len(scenarios)} scenarios x 3 strategies x "
        f"{len(seeds)} reps = {len(jobs)} simulations")
    flat = run_jobs(jobs, prefix="Staffing sensitivity", newline_every=50)

    rows = []
    for (name, strat), reps in zip(cells, chunked(flat, len(seeds))):
        waits = [r["W_total"] for r in reps]
        rows.append(dict(Scenario=name, Strategy=strat,
                         MeanWait=float(np.nanmean(waits)),
                         SD=float(np.nanstd(waits, ddof=1)) if len(waits) > 1 else float("nan")))
    log("Staffing sensitivity simulations done.")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "figure5_staffing_sensitivity_data.csv"), index=False)

    order = list(scenarios.keys())
    fig, ax = plt.subplots(figsize=(10, 7))
    for strat, style in [("IFP", "-o"), ("ALT", "--^"), ("SBP", ":s")]:
        sub = df[df.Strategy == strat].set_index("Scenario").loc[order]
        ax.plot(order, sub.MeanWait, style, label=strat,
                linewidth=2.2, markersize=8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average waiting time (min)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figure5_staffing_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("Saved Figure 5 -> figure5_staffing_sensitivity.png")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ACTION_FLAGS = ("all", "table2", "grid_search", "compare",
                "sensitivity_arrival", "sensitivity_staffing")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--table2", action="store_true")
    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--sensitivity-arrival", action="store_true")
    p.add_argument("--sensitivity-staffing", action="store_true")
    p.add_argument("--k1", type=float, default=13.1,
                   help="use a fixed k1 instead of grid-searching")
    p.add_argument("--k2", type=float, default=2.1,
                   help="use a fixed k2 instead of grid-searching")
    p.add_argument("--jobs", type=int, default=1,
                   help="worker processes for the replications (default 1). "
                        "Results are identical for any value.")
    p.add_argument("--wait-metric", choices=["classic", "extended"], default=None,
                   help="classic = physician queueing only (the paper's "
                        "Eq. 24-28); extended = physician + device queueing. "
                        "Defaults to this script's own setting.")
    p.add_argument("--coarse-step", type=float, default=None,
                   help="coarse grid step (default 1.0). 2 or 3 makes the "
                        "grid search several times faster.")
    p.add_argument("--fine-halfwidth", type=float, default=None,
                   help="fine grid spans best +/- this (default 1.0)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny replication counts and a coarse grid, to check "
                        "the pipeline end-to-end in minutes")
    p.add_argument("--reps-final", type=int, default=None,
                   help="override N_REPS_FINAL (main comparison + fine grid)")
    p.add_argument("--reps-grid", type=int, default=None,
                   help="override N_REPS_GRID (coarse grid)")
    p.add_argument("--reps-sensitivity", type=int, default=None,
                   help="override N_REPS_SENSITIVITY")
    return p


def apply_common_args(args):
    """Apply the shared CLI options to this module's globals."""
    global JOBS, N_REPS_FINAL, N_REPS_GRID, N_REPS_SENSITIVITY, WAIT_METRIC
    global COARSE_STEP, FINE_HALFWIDTH
    if getattr(args, "wait_metric", None):
        WAIT_METRIC = args.wait_metric
    if args.smoke:
        apply_smoke_settings()
        log("SMOKE MODE: tiny replication counts and a coarse grid. "
            "Use this only to verify the pipeline, never for results.")
    if args.reps_final is not None:
        N_REPS_FINAL = args.reps_final
    if args.reps_grid is not None:
        N_REPS_GRID = args.reps_grid
    if args.reps_sensitivity is not None:
        N_REPS_SENSITIVITY = args.reps_sensitivity
    if getattr(args, "coarse_step", None) is not None:
        COARSE_STEP = args.coarse_step
    if getattr(args, "fine_halfwidth", None) is not None:
        FINE_HALFWIDTH = args.fine_halfwidth
    JOBS = max(1, args.jobs)
    if JOBS > 1:
        log(f"Running replications across {JOBS} worker processes.")
    log(f"Examination order: {EXAM_ORDER_POLICY!r} | waiting-time metric: "
        f"{WAIT_METRIC!r} ("
        + ("physician queueing only -- the paper's definition"
           if WAIT_METRIC == "classic"
           else "physician + diagnostic-device queueing") + ")")


def main():
    args = build_parser().parse_args()

    # Default to --all only when no ACTION flag was given. (The old check
    # looked at every parsed value, including --k1/--k2, whose non-zero
    # defaults are always truthy -- so a bare `python experiments.py` silently
    # did nothing at all.)
    if not any(getattr(args, f) for f in ACTION_FLAGS):
        args.all = True

    apply_common_args(args)
    os.makedirs(OUT_DIR, exist_ok=True)

    k1_opt, k2_opt = args.k1, args.k2
    t0 = time.time()

    if args.all or args.table2:
        log("=== STEP: Table 2 export ===")
        export_table2()

    if args.all or args.grid_search:
        log("=== STEP: SBP grid search on the extended metric (Tables 3 & 4) ===")
        k1_opt, k2_opt = grid_search_sbp()
        log(f"Optimal SBP parameters found: k1={k1_opt}, k2={k2_opt}")
    else:
        log(f"Grid search skipped -- using k1={k1_opt}, k2={k2_opt} as given. "
            "These are only optimal if they came from a grid search run under "
            "this same metric.")

    if args.all or args.compare:
        log("=== STEP: Comparative analysis (Figure 3, Tables 5-10) ===")
        comparative_analysis(k1_opt, k2_opt)

    if args.all or args.sensitivity_arrival:
        log("=== STEP: Arrival-rate sensitivity (Figure 4) ===")
        sensitivity_arrival(k1_opt, k2_opt)

    if args.all or args.sensitivity_staffing:
        log("=== STEP: Staffing sensitivity (Figure 5) ===")
        sensitivity_staffing(k1_opt, k2_opt)

    log(f"ALL DONE. Total elapsed: {time.time() - t0:7.1f}s. "
        f"Outputs saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()