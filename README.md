# Congestion-Aware Diagnostic Routing in Emergency Departments

A discrete-event simulation (DES) study that reproduces the emergency-department
scheduling model of Lv et al., *Discrete Event Simulation-Based Analysis and
Optimization of Emergency Patient Scheduling Strategies* (Healthcare 2026, 14, 99),
and extends its diagnostic stage with per-station queues and congestion-aware
examination routing.

This repository contains the complete simulation engine, the experiment drivers,
and every table and figure reported in the accompanying paper and course report.

---

## 1. What this code does

The base model represents a single emergency department over one operational
week. Patients are triaged as Level III or Level IV, wait for an initial
consultation, may be sent for diagnostic testing, and then return for a
follow-up consultation. Three physician-scheduling policies compete for the same
pool of physicians:

- **IFP** (Initial First Policy) — fixed priority: new patients before returning
  patients, Level III before Level IV.
- **ALT** (Alternate) — at most `floor(R/2)` of the available physicians serve
  new patients; the remainder serve follow-up patients, with unused capacity
  released to the other group.
- **SBP** (Slack-Based Policy) — priority by proximity to the clinical deadline,
  governed by two tunable anticipation margins `(k1, k2)`.

### The extension

The base model orders a patient's required examinations once, in descending
order of reporting delay, and never revises that order. This repository adds two
changes, each controlled by a single parameter:

| | Fixed-order model | Congestion-aware model |
| --- | --- | --- |
| Output folder | `outputs/` | `outputs_dynamic_queue/` |
| Examination order | fixed in advance, longest reporting delay first | re-selected before every test from the current station workload |
| Waiting-time metric | `initial + follow-up` (as reported by the base paper) | `initial + station queueing + follow-up` |
| Engine parameters | `exam_order_policy="paper"`, `wait_metric="classic"` | `exam_order_policy="shortest_wait"`, `wait_metric="extended"` |

**Routing rule.** Let `R_i(t)` be the set of examinations patient *i* still
requires and `n_j(t)` the number of patients present at station *j*, including
the one in service. The next examination is

```
j* = argmin over j in R_i(t) of  n_j(t) * tau_j
```

re-evaluated after every completed examination. Ties use the fixed modality
order laboratory, ultrasound, X-ray, CT.

**Waiting-time metric.** The base paper counts only the physician queues. Since
each diagnostic station has capacity one, a patient can also wait for a busy
station; the base paper models that delay but does not count it as waiting time.
We count it, because it is precisely the quantity the routing rule acts on.
Processing times `tau_j` and reporting delays `delta_j` are excluded under both
metrics, as they represent service rather than queueing.

### Separating the two changes

The difference between `outputs/` and `outputs_dynamic_queue/` combines the
routing change and the metric change. Either can be held fixed:

```bash
# routing effect only (extended metric on both sides)
python experiments.py --all --wait-metric extended

# metric effect only (fixed order on both sides)
python experiments_dynamic_queue.py --all --wait-metric classic
```

`experiments_comparison.py` performs the full four-way factorial separation of
the routing rule and the tuned thresholds.

---

## 2. Requirements

```bash
pip install simpy numpy pandas scipy matplotlib
```

Python 3.9 or later (the base paper used 3.13.3).

**Optional — fonts.** Place any `.ttf` or `.otf` files in a `fonts/` directory
next to the scripts and they are registered automatically; the figures prefer
Book Antiqua, falling back to Palatino and then to the default serif face.

---

## 3. Repository structure

| File | Purpose |
| --- | --- |
| `ed_simulation.py` | DES engine: arrival process, triage, consultations, examination subsystem, physician dispatcher, the three policies, and KPI aggregation. Both examination-ordering rules live in `_exam_process` behind the `exam_order_policy` parameter; `summarize()` selects the metric via `wait_metric`. |
| `experiments.py` | Driver for the fixed-order model. Runs the grid search, the policy comparison, the paired statistics and the two sensitivity sweeps. Writes to `outputs/`. |
| `experiments_dynamic_queue.py` | Driver for the congestion-aware model. Imports `experiments.py` and sets `EXAM_ORDER_POLICY` and `WAIT_METRIC`, so both models run through an identical pipeline. Writes to `outputs_dynamic_queue/` and builds the side-by-side comparison figures in `outputs_comparison/`. |
| `experiments_comparison.py` | Cross-model experiments: the per-test-count decomposition of the diagnostic stage, and the four-way threshold-transfer design. Writes to `outputs_comparison/`. |

Keeping one engine and two thin drivers guarantees that the two diagnostic
models differ only in the routing rule and the metric.

---

## 4. Reproducing the results

### Step 0 — check available cores

```bash
cd Hospital_Sim
nproc                      # Windows: echo %NUMBER_OF_PROCESSORS%
```

Pass one or two fewer than the core count to `--jobs`; on a 16-core machine use
`--jobs 14`. Results are identical regardless of the job count.

### Step 1 — fixed-order model

```bash
python experiments.py --all --jobs 14
```

Full settings, no flags required: 100 replications for the main comparison and
the fine grid, coarse grid at step 1.0 over `k1 ∈ [0,30]` and `k2 ∈ [0,40]`
(1,271 pairs × 20 replications), fine grid at step 0.1.
Output: `outputs/`.

### Step 2 — congestion-aware model

```bash
python experiments_dynamic_queue.py --all --jobs 14
```

Output: `outputs_dynamic_queue/`.

### Step 3 — cross-model comparison

Take the optimal `(k1, k2)` from the first row of each
`table4_fine_grid_search.csv`, then:

```bash
python experiments_comparison.py --jobs 14 --reps 100 \
    --k-fixed 3.3 9.0 \
    --k-caware 12.0 24.0
```

Output: `outputs_comparison/`.

### Step 4 — comparison figures

```bash
python experiments_dynamic_queue.py --compare-plots
```

Redraws the three side-by-side figures from the saved CSVs, so both panels share
one vertical axis and one set of fonts. Requires Steps 1 and 2 to have run.

### Tuned thresholds obtained

| Model | `k1` | `k2` | Mean waiting time |
| --- | --- | --- | --- |
| Fixed order | 3.3 | 9.0 | 38.33 min |
| Congestion-aware | 12.0 | 24.0 | 39.08 min |

Each model must be tuned separately; neither pair is optimal under the other
model (see `table6_tuning_transfer.csv`).

### Runtime

On 16 cores, a full run takes roughly 30–60 minutes per script, dominated by the
grid search (approximately 25,000 coarse plus 44,100 fine simulations). Live
progress and an ETA are printed for each stage. For a long run:

```bash
nohup python experiments.py --all --jobs 14 > run_paper.log 2>&1 &
tail -f run_paper.log
```

Once the grid search has been completed, the remaining stages take a few minutes:

```bash
python experiments.py --compare --sensitivity-arrival --sensitivity-staffing \
    --jobs 14 --k1 3.3 --k2 9.0
```

### Reducing cost

| Flag | Effect |
| --- | --- |
| `--jobs N` | run on N cores in parallel (results unchanged) |
| `--coarse-step 2` | coarser grid, roughly four times fewer runs |
| `--fine-halfwidth 0.5` | smaller fine grid |
| `--reps-grid 10 --reps-final 50` | fewer replications |

---

## 5. Output files

`outputs/` and `outputs_dynamic_queue/` contain the same file names:

| Item | File |
| --- | --- |
| Arrival rates | `table2_arrival_rates.csv` |
| Coarse grid, top 5 | `table3_coarse_grid_search.csv` |
| Coarse grid, every cell | `table3_coarse_grid_search_full.csv` |
| Fine grid, top 5 | `table4_fine_grid_search.csv` |
| Fine grid, every cell | `table4_fine_grid_search_full.csv` |
| Waiting time by policy (mean, SD, CI) | `table_raw_means.csv` |
| Paired *t*-tests | `table5_paired_ttest.csv` |
| Cohen's *d_z* | `table6_cohens_d.csv` |
| Delay rate and service level | `table7_delay_service_level.csv` |
| Physician utilization | `table8_physician_utilization.csv` |
| Station utilization and volume | `table9_device_utilization.csv` |
| Waiting-time decomposition | `table10_wait_decomposition.csv` |
| Examination decomposition | `table11_exam_decomposition.csv` |
| Waiting-time boxplot | `figure3_waiting_time_boxplot.png` + `figure3_waiting_data.csv` |
| Arrival-rate sensitivity | `figure4_arrival_sensitivity.png` + `_data.csv` |
| Staffing sensitivity | `figure5_staffing_sensitivity.png` + `_data.csv` |

`table10_wait_decomposition.csv` is the central file for the metric extension.
Its columns are `Consult_Queue_Wait`, `Device_Queue_Wait`, `Headline_Wait`,
`Device_Counted` and `Device_Share_pct`. Under the classic metric the station
queueing is measured but excluded from the headline figure
(`Device_Counted = False`), so the magnitude of what the base paper omits can be
read directly; under the extended metric it is included.

`outputs_comparison/` contains the cross-model results:
`table5_exam_decomposition.csv` (station queueing and diagnostic-stage time by
test count, with per-row paired *t*-tests), `table5_exam_summary_stats.csv`,
`table6_tuning_transfer*.csv`, `table6_service_level_decomposition.csv`,
`per_level_waiting.csv`, `routing_effect_by_policy.csv`, and the three
`compare_figure*.png` files.

---

## 6. Principal findings

**The routing rule reduces the quantity it targets.** Mean station queueing
falls from 0.509 to 0.365 min per examined patient, a reduction of 28.2%
(*t* = −18.61, *p* < 1e−30). The saving grows with the number of tests a patient
requires, reaching 52% for the 3.2% of patients who require all four.

**Total diagnostic-stage time nevertheless increases**, from 29.94 to 31.44 min
(+5.0%, *t* = 49.31, *p* < 1e−60). Station queueing is only 1.70% of that stage,
consistent with station utilizations of 14–19%; the remainder is dominated by
reporting delays. Because the routing rule never reads `delta_j`, it can defer a
long-reporting examination to the end of the sequence, where no remaining test
overlaps it.

**The tuned thresholds do not transfer between models.** Changing the routing
rule alone costs 4.45 min (*p* = 0.003) and changing the thresholds alone costs
2.75 min (*p* = 0.062), whereas changing both costs 0.74 min and is not
significant (*p* = 0.59). The two changes are harmful in isolation and
approximately neutral together, so they must be selected jointly.

**Station queueing is small in this configuration, and does not grow with
demand.** The physicians, not the equipment, are the binding resource: physician
utilization is approximately 86% against 14–19% at the stations. Because
patients reach the stations only after being seen by a physician, the arrival
rate at the stations is bounded by physician throughput, so additional demand
lengthens the physician queues rather than the station queues. This is an
independent confirmation of the base paper's own conclusion that the physician
pool is the bottleneck, and it bounds what any station-level routing rule can
achieve here.

---

## 7. Modelling assumptions

The base paper publishes neither its dataset nor its code, so several details
are reasonable assumptions. All are documented in the `ed_simulation.py`
docstrings.

1. Truncated exponential service times with `lambda = 1/mean` (9 min initial,
   15 min follow-up), sampled by inverse CDF.
2. Examination selection: a Bernoulli(0.6) gate, then the four modalities drawn
   independently (0.92, 0.22, 0.29, 0.55); if none is drawn, one is selected
   uniformly at random.
3. Each station has capacity one, inferred from the reported availability of
   10,080 min per week in the base paper's Table 9.
4. SBP ties are broken by earliest arrival.
5. Shift changes in physician capacity are non-preemptive.
6. The service-level bound is not stated in the base paper; 0.90 is used here
   (`SL_MIN_3` and `SL_MIN_4` in `experiments.py`).
7. Waiting time is measured over a patient's whole journey and grouped by the
   triage level assigned on arrival, rather than by visit type. Under the
   literal reading of the base paper's equations, IFP's Level III waiting time
   is almost zero, which contradicts the roughly 44 min the paper reports.
8. The SBP urgency clock runs from arrival and continues to apply in the
   follow-up queue.
9. Station queueing is part of the headline metric under the extended
   definition (Section 1).

Assumptions 7–9 apply the same principle: where the literal equations and the
reported numbers disagree, internal consistency is given priority.

---

## 8. Known limitations

- Exact values do not match the base paper, since both the assumptions and the
  metric differ.
- In this reproduction SBP does not beat ALT by the margin the base paper
  reports; the two are statistically indistinguishable at the observed arrival
  rate, and ALT overtakes SBP from +3% demand onward. Use `--reps-final 100` for
  any reported figure.
- All required examinations are known before testing begins, so a result that
  triggers a further test is not modelled.
- Each replication starts from an empty system with no warm-up period.
- The staffing sweep only adds physicians and never removes them.
