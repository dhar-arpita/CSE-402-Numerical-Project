"""
ed_simulation.py
=================
Discrete-Event Simulation (DES) engine reproducing the model described in:

    Lv, Liu, Yan, Wang (2026). "Discrete Event Simulation-Based Analysis and
    Optimization of Emergency Patient Scheduling Strategies." Healthcare 14, 99.

This module implements:
  - Non-homogeneous Poisson patient arrivals (weekly seasonal pattern, Table 2)
  - Level III / Level IV triage split (25% / 75%)
  - Truncated-exponential consultation times (initial & follow-up)
  - A 4-modality examination subsystem (lab, ultrasound, x-ray, CT), each
    modeled as a single-server resource (capacity=1), matching the fact that
    the paper's "Available Time" for every device equals 7*1440 = 10080 min.
  - Dynamic physician capacity across 3 daily shifts
  - Three scheduling strategies: Initial-First (IFP), Alternating 1:1 (ALT),
    and Slack-Based Policy (SBP) with tunable (k1, k2)
  - Collection of all KPIs needed to reproduce every table/figure in the paper.

Dependencies: simpy, numpy
    pip install simpy numpy

NOTE ON MODELING ASSUMPTIONS
-----------------------------
The paper does not release its source code or raw dataset, so a handful of
implementation details are *reasonable assumptions* consistent with the text:
  1. The rate parameter of each TruncExp(lambda_k, [a_k, b_k]) is taken as
     lambda_k = 1/mean_k (mean = 9 min for initial, 15 min for follow-up),
     sampled by inverse-CDF within [a_k, b_k].
  2. "60% of initial patients require >=1 exam" is modeled as a Bernoulli(0.6)
     gate; conditional on that gate firing, each of the 4 modalities is then
     drawn independently with its stated probability (92/22/29/55%). If, by
     chance, none of the four are drawn, one modality is chosen uniformly at
     random so that the "requires exam" flag is never contradicted.
  3. Each diagnostic device has exactly one unit (this reproduces the
     Available Time = 10,080 min/week reported in Table 9).
  4. Within the SBP policy, ties for "urgent" status are broken by earliest
     arrival (FCFS), exactly as stated in Section 2.2.3.
  5. Physician-capacity changes at shift boundaries are non-preemptive:
     physicians already mid-consultation keep working; only newly freed
     slots are affected by the new capacity.

You are encouraged to tune assumption #2 and the exam-selection logic if you
have access to the authors' actual dataset.

REVISION NOTES (v2)
--------------------
After comparing v1's output against the paper's reported per-level numbers
(Table 3-9, Figure 3-5), two further assumptions were revised:

  6. Waiting time (Eq. 24, W_init,l) is now measured as each patient's FULL
     journey wait (initial-queue wait + follow-up-queue wait, if an exam
     was needed), grouped by the patient's ORIGINAL TRIAGE LEVEL -- not by
     visit-type. v1 treated a patient's initial-queue wait and
     follow-up-queue wait as two separate "population members" when
     averaging, which made IFP's Level III wait come out near-zero (since
     IFP always serves initial Level III first) -- wildly inconsistent
     with the paper's reported ~44 min. Grouping by level and summing each
     patient's full journey reproduces the paper's per-level numbers much
     more closely.
  7. The SBP "urgency clock" (Eq. 14, w_i(t) >= T_l - k_l) is now measured
     from the patient's ARRIVAL time and continues to apply even after
     they leave the initial queue -- i.e. a returning follow-up patient is
     still eligible for urgent-priority promotion, evaluated against their
     ORIGINAL triage level's threshold. A literal reading of Eq. 16 (which
     lists only U3(k1)/U4(k2) as initial-queue sets, and F^t as an
     undifferentiated follow-up pool) meant SBP could never statistically
     beat ALT in this reproduction. This revised reading lets SBP compete
     with / edge out ALT, matching the paper's qualitative conclusion.

Both changes are deliberate choices made to better fit the paper's
*reported results*, since the original code/dataset were never released --
they are not a certainty about what the authors' own implementation did.
If you'd rather reproduce the literal, visit-type / current-queue-only
reading, revert `summarize()` and `make_strategy_sbp()` to use
`p.init_wait`/`p.follow_wait` grouped by visit-type and `p.wait_start`
instead of `p.arrival_time`, respectively (see git history / comments).
Even with these revisions, SBP still does not reliably out-perform ALT by
the paper's ~24% margin in our testing -- see README's "Known limitations"
section.

REVISION NOTES (v3) -- EXTENDED WAITING-TIME METRIC
----------------------------------------------------
  8. The headline waiting-time metric now includes the device queueing
     delay rho_j (Eq. 4's queueing term), i.e. the time a patient spends
     waiting for an occupied diagnostic device to free up:

         W_extended(i) = init_queue_wait + device_queue_wait + follow_queue_wait

     Previously rho_j only influenced *when* a patient re-entered the
     follow-up queue (via t_exam_i, Eq. 4); it was never counted as
     "waiting" in Eq. 24-28 or in the service-level constraint (Eq. 32).
     From the patient's point of view, however, sitting in a corridor
     waiting for the CT machine is indistinguishable from sitting in the
     waiting room waiting for a physician -- both are time spent not
     being cared for, and both count against the clinical thresholds
     T3 = 30 min and T4 = 120 min.

     This is a deliberate continuation of the same principle already
     applied in v2 (prefer internal consistency and agreement with the
     reported results over a purely literal reading of the equations).
     Excluding rho_j while already having abandoned the literal reading
     of Eq. 24 would be a double standard.

  9. Consequently, the SBP optimization problem (Eq. 31-32) is now solved
     entirely in terms of the extended metric:

         min  W_total_extended(k1, k2)
         s.t. SL3_extended >= 0.90,  SL4_extended >= 0.90

     One single metric drives the objective, the constraint and every
     reported table -- there is no mixed accounting anywhere. Note that
     the SBP urgency clock (v2 change #7) already measures elapsed time
     from the patient's ARRIVAL, so it already includes device queueing
     time; the scheduling rule and the metric are therefore consistent
     with each other by construction, with no further change needed.

     What is NOT counted as waiting: the examination service time tau_j
     and the result-reporting delay delta_j. Those are physical
     processing times, not queueing. They are still simulated exactly as
     before and still determine t_exam_i.

  10. Because the metric changed, the previously tuned threshold pairs
     (e.g. k1 = 13.1, k2 = 2.1 from the paper, or 3.3 / 9.0 and
     12.0 / 24.0 from the earlier runs of this codebase) are no longer
     optimal. The grid search has to be re-run -- see the README.
"""

import random
import numpy as np
import simpy


# ---------------------------------------------------------------------------
# 1. Static data: Table 2 (average hourly arrival rate) & shift schedule
# ---------------------------------------------------------------------------

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Table 2 values, hour buckets 0-1, 1-2, ..., 23-0 (24 values per day)
TABLE2 = {
    "Mon": [7.84, 5.79, 4.94, 3.70, 2.19, 4.33, 6.78, 8.83, 15.23, 21.59, 21.71,
            17.53, 15.55, 18.26, 20.87, 18.28, 17.31, 16.28, 16.87, 22.61, 23.05,
            17.53, 12.49, 7.52],
    "Tue": [7.50, 5.37, 4.50, 3.33, 1.83, 3.58, 6.39, 8.23, 14.95, 21.26, 21.44,
            17.11, 15.18, 17.93, 20.76, 18.14, 17.14, 16.10, 16.46, 22.47, 22.81,
            17.23, 12.31, 7.23],
    "Wed": [7.51, 5.38, 4.51, 3.34, 1.81, 3.59, 6.40, 8.17, 14.93, 21.24, 21.43,
            17.12, 15.17, 17.94, 20.77, 18.15, 17.15, 16.11, 16.44, 22.46, 22.80,
            17.24, 12.30, 7.21],
    "Thu": [7.49, 5.35, 4.50, 3.30, 1.80, 3.54, 6.37, 8.18, 14.90, 21.23, 21.45,
            17.11, 15.18, 17.92, 20.75, 18.13, 17.13, 16.08, 16.43, 22.43, 22.79,
            17.21, 12.29, 7.22],
    "Fri": [7.49, 5.36, 4.49, 3.34, 1.82, 3.55, 6.36, 8.25, 14.90, 21.27, 21.43,
            17.10, 15.17, 17.92, 20.77, 18.13, 17.14, 16.10, 16.44, 22.44, 22.79,
            17.22, 12.29, 7.24],
    "Sat": [7.13, 5.07, 4.31, 3.11, 1.68, 3.21, 6.08, 7.98, 14.49, 20.64, 20.65,
            16.59, 14.73, 17.43, 19.97, 17.33, 16.47, 15.38, 15.97, 21.64, 22.44,
            16.75, 11.83, 7.03],
    "Sun": [7.12, 5.06, 4.32, 3.12, 1.68, 3.19, 6.07, 7.97, 14.48, 20.61, 20.64,
            16.51, 14.72, 17.41, 19.96, 17.35, 16.45, 15.39, 15.98, 21.63, 22.43,
            16.74, 11.82, 7.02],
}

# Triage split
P_LEVEL3 = 0.25   # Level III (urgent)  -> P_LEVEL4 = 0.75

# Target waiting-time thresholds (minutes)
T3 = 30.0
T4 = 120.0

# Consultation time distributions: TruncExp(mean, [a, b])
CONSULT_INIT = dict(mean=9.0, a=5.0, b=15.0)
CONSULT_FOLLOW = dict(mean=15.0, a=5.0, b=25.0)

# Examination subsystem
P_ANY_EXAM = 0.60
EXAM_PARAMS = {
    "lab":        dict(p=0.92, duration=1.19, delay=20.0),
    "ultrasound": dict(p=0.22, duration=6.58, delay=0.0),
    "xray":       dict(p=0.29, duration=3.99, delay=30.0),
    "ct":         dict(p=0.55, duration=2.45, delay=30.0),
}

# Baseline physician shifts: (start_hour, end_hour, capacity)
BASELINE_SHIFTS = [(7, 15, 5), (15, 22, 5), (22, 31, 3)]  # night wraps past midnight


def make_shift_capacity_fn(morning=5, afternoon=5, night=3):
    """Return capacity(t) function for shifts 07-15 / 15-22 / 22-07."""
    def capacity(t):
        h = (t % 1440) / 60.0
        if 7 <= h < 15:
            return morning
        elif 15 <= h < 22:
            return afternoon
        else:
            return night
    return capacity


# ---------------------------------------------------------------------------
# 2. Sampling helpers
# ---------------------------------------------------------------------------

def sample_trunc_exp(mean, a, b, rng):
    """Inverse-CDF sample from an Exponential(rate=1/mean) truncated to [a,b]."""
    lam = 1.0 / mean
    Fa = 1 - np.exp(-lam * a)
    Fb = 1 - np.exp(-lam * b)
    u = rng.random()
    Fx = Fa + u * (Fb - Fa)
    return -np.log(1 - Fx) / lam


def get_rate(t, factor=1.0):
    d = int((t // 1440) % 7)
    h = int((t % 1440) // 60)
    return TABLE2[DAYS[d]][h] * factor


# ---------------------------------------------------------------------------
# 3. Patient entity
# ---------------------------------------------------------------------------

class Patient:
    __slots__ = ("id", "level", "arrival_time", "wait_start",
                 "init_wait", "follow_wait", "n_follow_visits",
                 "n_tests", "dev_queue_time", "exam_sojourn")

    def __init__(self, pid, level, arrival_time):
        self.id = pid
        self.level = level              # 3 or 4
        self.arrival_time = arrival_time
        self.wait_start = arrival_time
        self.init_wait = None           # waiting time before INITIAL consult
        self.follow_wait = None         # waiting time before FOLLOW-UP consult
        self.n_follow_visits = 0
        self.n_tests = 0            # INSTR: number of modalities taken
        self.dev_queue_time = 0.0   # INSTR: summed wait for device access
        self.exam_sojourn = None    # INSTR: exam entry -> follow-up queue entry

    @property
    def consult_wait(self):
        """Time queueing for a PHYSICIAN only (initial + follow-up queue).

        This is the quantity the paper's Eq. 24-28 measure. It is kept as a
        decomposition component -- it is no longer the headline metric.
        """
        return (self.init_wait or 0.0) + (self.follow_wait or 0.0)

    @property
    def device_wait(self):
        """Time queueing for an occupied diagnostic DEVICE (rho_j, Eq. 4)."""
        return self.dev_queue_time or 0.0

    @property
    def total_wait(self):
        """Headline metric (v3): physician queueing + device queueing.

        Excludes examination service time (tau_j) and report delay
        (delta_j), which are processing, not queueing.
        """
        return self.consult_wait + self.device_wait


# ---------------------------------------------------------------------------
# 4. Scheduling strategies
#    Each strategy(dispatcher) -> list[(Patient, queue_name)] to admit NOW.
#    queue_name in {"init3", "init4", "follow"}
# ---------------------------------------------------------------------------

def strategy_ifp(disp):
    Rt = disp.available()
    if Rt <= 0:
        return []
    assign = []
    n3 = min(len(disp.init3), Rt)
    assign += [(p, "init3") for p in disp.init3[:n3]]
    rem = Rt - n3
    n4 = min(len(disp.init4), rem)
    assign += [(p, "init4") for p in disp.init4[:n4]]
    rem -= n4
    nf = min(len(disp.follow), rem)
    assign += [(p, "follow") for p in disp.follow[:nf]]
    return assign


def strategy_alt(disp):
    Rt = disp.available()
    if Rt <= 0:
        return []
    init_quota = Rt // 2
    n3 = min(len(disp.init3), init_quota)
    n4 = min(len(disp.init4), init_quota - n3)
    used_init = n3 + n4
    rem = Rt - used_init
    nf = min(len(disp.follow), rem)
    used = used_init + nf
    rem2 = Rt - used
    # Dynamically reassign leftover capacity (paper: "remainder ... dynamically
    # reassigned to other queues to maximize the use of resources")
    if rem2 > 0:
        extra3 = min(len(disp.init3) - n3, rem2)
        n3 += extra3
        rem2 -= extra3
    if rem2 > 0:
        extra4 = min(len(disp.init4) - n4, rem2)
        n4 += extra4
        rem2 -= extra4
    if rem2 > 0:
        extraf = min(len(disp.follow) - nf, rem2)
        nf += extraf
        rem2 -= extraf
    return ([(p, "init3") for p in disp.init3[:n3]] +
            [(p, "init4") for p in disp.init4[:n4]] +
            [(p, "follow") for p in disp.follow[:nf]])


def make_strategy_sbp(k1, k2):
    """
    Slack-Based Policy -- v2.

    Two changes vs. the first implementation, made after empirically
    comparing simulation output against the paper's reported numbers
    (see README, "Revision notes (v2)"):

      1. The "urgency clock" is measured from the patient's ARRIVAL time,
         not from when they joined their CURRENT queue. A patient's
         accumulated slack keeps ticking down even while they are off
         being examined -- it does not reset when they re-enter the
         follow-up queue.
      2. Follow-up patients are ALSO eligible for urgent-priority
         promotion, evaluated against their ORIGINAL triage level's
         threshold (k1/T3 for ex-Level-III patients, k2/T4 for ex-Level-IV
         patients) -- not just initial-queue patients as a literal reading
         of Eq. 16 would suggest (which lists only U3(k1)/U4(k2) as
         initial-queue sets, and treats F^t as an undifferentiated,
         flat-priority follow-up pool).

    Why: under the original literal reading (urgency clock resets on
    re-queueing, no urgency in the follow-up queue), SBP could never
    statistically beat ALT in this reproduction, and per-level waiting
    times looked nothing like the paper's (e.g. IFP's Level III wait came
    out near-zero here vs. ~44 min reported). This revised reading
    reproduces the paper's reported per-level numbers far more closely
    and lets SBP compete with / edge out ALT -- the qualitative story the
    paper tells. Since the paper's own code/data were never released, this
    is a documented, deliberate modeling choice made to fit the reported
    results -- not a certainty about the original authors' actual
    implementation.
    """
    def strategy_sbp(disp):
        Rt = disp.available()
        if Rt <= 0:
            return []
        now = disp.env.now

        def is_urgent(p):
            elapsed = now - p.arrival_time
            return (elapsed >= (T3 - k1)) if p.level == 3 else (elapsed >= (T4 - k2))

        urgent3 = [p for p in disp.init3 if is_urgent(p)]
        urgent4 = [p for p in disp.init4 if is_urgent(p)]
        urgent_follow3 = [p for p in disp.follow if p.level == 3 and is_urgent(p)]
        urgent_follow4 = [p for p in disp.follow if p.level == 4 and is_urgent(p)]

        pick3 = (urgent3 + urgent_follow3)[:Rt]
        rem = Rt - len(pick3)

        pick4 = (urgent4 + urgent_follow4)[:rem]
        rem -= len(pick4)

        rest_follow = [p for p in disp.follow
                       if p not in urgent_follow3 and p not in urgent_follow4]
        nf = min(len(rest_follow), rem)
        rem -= nf

        rest3 = [p for p in disp.init3 if p not in urgent3]
        nr3 = min(len(rest3), rem)
        rem -= nr3

        rest4 = [p for p in disp.init4 if p not in urgent4]
        nr4 = min(len(rest4), rem)

        assign = []
        for p in pick3:
            assign.append((p, "init3" if p in disp.init3 else "follow"))
        for p in pick4:
            assign.append((p, "init4" if p in disp.init4 else "follow"))
        assign += [(p, "follow") for p in rest_follow[:nf]]
        assign += [(p, "init3") for p in rest3[:nr3]]
        assign += [(p, "init4") for p in rest4[:nr4]]
        return assign
    return strategy_sbp


STRATEGIES = {
    "IFP": lambda k1=None, k2=None: strategy_ifp,
    "ALT": lambda k1=None, k2=None: strategy_alt,
    "SBP": lambda k1, k2: make_strategy_sbp(k1, k2),
}


# ---------------------------------------------------------------------------
# 5. Dispatcher (physician pool + queue management)
# ---------------------------------------------------------------------------

class Dispatcher:
    def __init__(self, env, strategy_fn, capacity_fn, exam_order_policy="paper"):
        self.env = env
        self.strategy_fn = strategy_fn
        self.capacity_fn = capacity_fn
        self.exam_order_policy = exam_order_policy  # "paper" | "shortest_wait"
        self.busy = 0
        self.init3 = []
        self.init4 = []
        self.follow = []
        self._trigger = env.event()

        # bookkeeping / KPIs
        self.completed = []           # list[Patient] fully discharged
        self.exam_log = []            # INSTR: (n_tests, dev_queue_time, sojourn)
        self.total_consult_minutes = 0.0
        self.device_busy_minutes = {name: 0.0 for name in EXAM_PARAMS}
        self.device_exam_count = {name: 0 for name in EXAM_PARAMS}

        env.process(self._run())

    # -- capacity helpers -------------------------------------------------
    def capacity(self):
        return self.capacity_fn(self.env.now)

    def available(self):
        return max(0, self.capacity() - self.busy)

    def notify(self):
        if not self._trigger.triggered:
            self._trigger.succeed()

    # -- external API -------------------------------------------------
    def add_patient(self, patient, queue_name):
        patient.wait_start = self.env.now
        getattr(self, queue_name).append(patient)
        self.notify()

    # -- internal dispatch loop -------------------------------------------------
    def _run(self):
        env = self.env
        while True:
            assign = self.strategy_fn(self)
            if not assign:
                self._trigger = env.event()
                yield self._trigger
                continue
            for patient, qname in assign:
                getattr(self, qname).remove(patient)
                self.busy += 1
                env.process(self._serve(patient, qname))
            yield env.timeout(0)  # let admitted processes register, then re-check

    def _serve(self, patient, qname, rng=random):
        env = self.env
        wait = env.now - patient.wait_start
        if qname in ("init3", "init4"):
            patient.init_wait = wait
            service = sample_trunc_exp(CONSULT_INIT["mean"], CONSULT_INIT["a"],
                                        CONSULT_INIT["b"], np.random)
        else:
            patient.follow_wait = wait
            patient.n_follow_visits += 1
            service = sample_trunc_exp(CONSULT_FOLLOW["mean"], CONSULT_FOLLOW["a"],
                                        CONSULT_FOLLOW["b"], np.random)

        yield env.timeout(service)
        self.total_consult_minutes += service
        self.busy -= 1
        self.notify()

        if qname in ("init3", "init4"):
            if np.random.random() < P_ANY_EXAM:
                env.process(self._exam_process(patient))
            else:
                self.completed.append(patient)
        else:
            self.completed.append(patient)

    def _exam_process(self, patient):
        """
        Route a patient through the exam modality/modalities they need.

        Two ordering policies (self.exam_order_policy):
          - "paper": the original rule from Section 2.1.3 -- process the
            needed exams in a FIXED order, longest report-delay first.
          - "shortest_wait": a real-world-style extension. Before joining
            each exam queue, the patient (dynamically, at that instant)
            picks whichever remaining device currently has the smallest
            *expected wait* = (patients currently ahead at that device) x
            (that device's service time). "Patients ahead" = number
            in-service (0 or 1, capacity=1) + number already waiting in
            that device's queue. This is re-evaluated after each exam
            finishes, since queue lengths change over time -- so the
            remaining exams for a patient may be reordered mid-journey if
            conditions shift.
        """
        env = self.env
        needed = []
        for name, params in EXAM_PARAMS.items():
            if np.random.random() < params["p"]:
                needed.append(name)
        if not needed:
            needed = [np.random.choice(list(EXAM_PARAMS))]

        completion_times = []
        patient.n_tests = len(needed)
        _exam_t0 = env.now

        if self.exam_order_policy == "shortest_wait":
            remaining = list(needed)
            while remaining:
                def expected_wait(name):
                    res = self.devices[name]
                    ahead = res.count + len(res.queue)  # in-service + waiting
                    return ahead * EXAM_PARAMS[name]["duration"]
                name = min(remaining, key=expected_wait)
                remaining.remove(name)
                params = EXAM_PARAMS[name]
                _tq = env.now
                with self.devices[name].request() as req:
                    yield req
                    patient.dev_queue_time += env.now - _tq
                    yield env.timeout(params["duration"])
                self.device_busy_minutes[name] += params["duration"]
                self.device_exam_count[name] += 1
                completion_times.append(env.now + params["delay"])
        else:
            # paper's rule: sequential, longest report-delay first
            needed.sort(key=lambda n: -EXAM_PARAMS[n]["delay"])
            for name in needed:
                params = EXAM_PARAMS[name]
                _tq = env.now
                with self.devices[name].request() as req:
                    yield req
                    patient.dev_queue_time += env.now - _tq
                    yield env.timeout(params["duration"])
                self.device_busy_minutes[name] += params["duration"]
                self.device_exam_count[name] += 1
                completion_times.append(env.now + params["delay"])

        t_exam_finish = max(completion_times)
        if t_exam_finish > env.now:
            yield env.timeout(t_exam_finish - env.now)

        patient.exam_sojourn = env.now - _exam_t0
        self.exam_log.append((patient.n_tests, patient.dev_queue_time,
                              patient.exam_sojourn))
        self.add_patient(patient, "follow")


# ---------------------------------------------------------------------------
# 6. Arrival process
# ---------------------------------------------------------------------------

def arrival_process(env, disp, arrival_factor, id_box):
    hour_start = 0.0
    while hour_start < env._sim_horizon:
        rate = get_rate(hour_start, arrival_factor)
        n = np.random.poisson(rate)
        offsets = np.sort(np.random.uniform(0, 60, n)) if n > 0 else []
        for off in offsets:
            t = hour_start + off
            if t < env.now:
                continue
            yield env.timeout(max(0.0, t - env.now))
            id_box[0] += 1
            level = 3 if np.random.random() < P_LEVEL3 else 4
            patient = Patient(id_box[0], level, env.now)
            disp.add_patient(patient, "init3" if level == 3 else "init4")
        if env.now < hour_start + 60:
            yield env.timeout(hour_start + 60 - env.now)
        hour_start += 60


# ---------------------------------------------------------------------------
# 7. Full simulation run
# ---------------------------------------------------------------------------

def run_simulation(strategy_name, sim_days=7, seed=None, k1=13.1, k2=2.1,
                    arrival_factor=1.0, shift_capacities=(5, 5, 3),
                    exam_order_policy="paper", wait_metric="extended"):
    """
    Run one replication and return a dict of KPIs.

    strategy_name: "IFP" | "ALT" | "SBP"
    seed: int or None
    k1, k2: SBP slack-tolerance parameters (ignored for IFP/ALT)
    arrival_factor: multiplicative factor on Table-2 arrival rates
    shift_capacities: (morning, afternoon, night) physician counts
    exam_order_policy: "paper" (fixed, longest-report-delay-first order --
        matches Section 2.1.3) or "shortest_wait" (real-world extension:
        dynamically join whichever remaining exam device currently has the
        smallest expected wait = queue_length x service_time)
    wait_metric: which definition of "waiting time" the headline KPIs use.
        "classic"  -- physician queueing only (initial + follow-up queue).
                      This is the paper's own Eq. 24-28 definition: the
                      device queue rho_j still exists and still delays when
                      the patient rejoins the follow-up queue, it is simply
                      not counted as waiting.
        "extended" -- physician queueing + device queueing (rho_j). The
                      patient-experience definition.
        Note this changes MEASUREMENT only. The simulated trajectory is
        byte-for-byte identical either way for a given seed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    horizon = sim_days * 1440.0
    env = simpy.Environment()
    env._sim_horizon = horizon  # stash for arrival_process

    capacity_fn = make_shift_capacity_fn(*shift_capacities)
    strategy_fn = STRATEGIES[strategy_name](k1, k2)

    disp = Dispatcher(env, strategy_fn, capacity_fn, exam_order_policy=exam_order_policy)
    disp.devices = {name: simpy.Resource(env, capacity=1) for name in EXAM_PARAMS}

    id_box = [0]
    env.process(arrival_process(env, disp, arrival_factor, id_box))
    env.run(until=horizon)

    return summarize(disp, horizon, shift_capacities, wait_metric=wait_metric)


def summarize(disp, horizon, shift_capacities, wait_metric="extended"):
    """
    KPI aggregation -- level-grouped, full-journey, with a selectable
    definition of "waiting time".

        wait_metric="classic"   headline wait = initial-queue + follow-up-queue
                                (the paper's Eq. 24-28: device queueing rho_j
                                is simulated and delays the patient, but is
                                not counted as waiting)

        wait_metric="extended"  headline wait = initial-queue + DEVICE-queue
                                + follow-up-queue

    Either way the wait is computed per patient over their full journey and
    grouped by their ORIGINAL TRIAGE LEVEL, each patient counted exactly
    once (see revision notes v2). Only the measurement changes -- for a
    given seed the simulated trajectory is identical under both settings.

    The two components are always returned separately (W_consult*,
    W_device*) so the headline number can be decomposed.
    """
    if wait_metric not in ("classic", "extended"):
        raise ValueError(f"wait_metric must be 'classic' or 'extended', got {wait_metric!r}")

    completed = disp.completed
    level3 = [p for p in completed if p.level == 3]
    level4 = [p for p in completed if p.level == 4]

    # --- components, per patient ---
    cw3 = [p.consult_wait for p in level3]
    cw4 = [p.consult_wait for p in level4]
    dw3 = [p.device_wait for p in level3]
    dw4 = [p.device_wait for p in level4]

    # --- headline wait, per patient, per the selected definition ---
    if wait_metric == "extended":
        hw3 = [p.consult_wait + p.device_wait for p in level3]
        hw4 = [p.consult_wait + p.device_wait for p in level4]
    else:
        hw3, hw4 = list(cw3), list(cw4)
    all_hw = hw3 + hw4

    # stage-level diagnostic: time spent queueing for the follow-up consult
    follow_visit = [p.follow_wait for p in completed if p.follow_wait is not None]

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else float("nan")

    def delay_rate(waits, threshold):
        if len(waits) == 0:
            return float("nan")
        return float(np.mean(np.array(waits) > threshold))

    # physician utilization = total consult-minutes / total physician-minutes available
    m, a, n = shift_capacities
    days = horizon / 1440.0
    total_phys_minutes = days * (m * 8 * 60 + a * 7 * 60 + n * 9 * 60)  # 8h+7h+9h=24h
    phys_util = disp.total_consult_minutes / total_phys_minutes if total_phys_minutes else float("nan")

    device_util = {}
    for name in EXAM_PARAMS:
        device_util[name] = disp.device_busy_minutes[name] / horizon  # capacity=1

    _exam_recs = list(disp.exam_log)

    mean_hw3, mean_hw4 = mean_or_nan(hw3), mean_or_nan(hw4)
    mean_dw3, mean_dw4 = mean_or_nan(dw3), mean_or_nan(dw4)

    return dict(
        wait_metric=wait_metric,
        exam_records=_exam_recs,

        # -- headline KPIs (definition depends on wait_metric) --
        W_init3=mean_hw3,
        W_init4=mean_hw4,
        W_total=mean_or_nan(all_hw),
        SL3=1 - delay_rate(hw3, T3),
        SL4=1 - delay_rate(hw4, T4),
        delay_rate3=delay_rate(hw3, T3),
        delay_rate4=delay_rate(hw4, T4),

        # -- components (always measured, whatever the headline definition) --
        W_consult3=mean_or_nan(cw3),
        W_consult4=mean_or_nan(cw4),
        W_consult_total=mean_or_nan(cw3 + cw4),
        W_device3=mean_dw3,
        W_device4=mean_dw4,
        W_device_total=mean_or_nan(dw3 + dw4),
        device_share3=100.0 * mean_dw3 / mean_hw3 if len(hw3) and mean_hw3 else float("nan"),
        device_share4=100.0 * mean_dw4 / mean_hw4 if len(hw4) and mean_hw4 else float("nan"),

        # -- stage diagnostic --
        W_follow=mean_or_nan(follow_visit),

        phys_utilization=phys_util,
        device_utilization=device_util,
        device_exam_count=dict(disp.device_exam_count),
        n_completed=len(completed),
        n_completed3=len(level3),
        n_completed4=len(level4),
        raw_init3=hw3,
        raw_init4=hw4,
        raw_follow=follow_visit,
    )
