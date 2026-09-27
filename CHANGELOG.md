# CHANGELOG — v3 (extended waiting-time metric)

> **v3.1 আপডেট:** metric এখন switchable। `experiments.py` চালায় পেপারের
> হিসাব (`--wait-metric classic`, default), আর
> `experiments_dynamic_queue.py` চালায় extension (`--wait-metric extended`,
> default)। দুই জায়গাতেই flag দিয়ে বদলানো যায়, যাতে metric আর exam-routing
> — দুটোর প্রভাব আলাদা করে দেখা যায়। `wait_metric` প্রতিটা job-এর ভেতরে
> যায়, তাই multiprocessing-এ নিরাপদ।

আগের ভার্সন থেকে কী কী বদলালো, ফাইল ধরে ধরে।

---

## `ed_simulation.py`

### ১. Extended waiting-time metric (মূল পরিবর্তন)

`Patient`-এ তিনটা property:

| property | মানে |
| --- | --- |
| `consult_wait` | initial-queue wait + follow-up-queue wait (আগের হিসাব) |
| `device_wait` | ডিভাইসের লাইনে অপেক্ষা (ρⱼ) |
| `total_wait` | উপরের দুটোর যোগ ⬅ **এখন এটাই headline metric** |

`dev_queue_time` ফিল্ডটা আগে থেকেই ছিল আর `_exam_process`-এ accumulate
হচ্ছিল — শুধু কোনো KPI-তে ব্যবহার হতো না। এখন হয়।

### ২. `summarize()` পুরো নতুন করে লেখা

সব headline KPI (`W_init3`, `W_init4`, `W_total`, `SL3`, `SL4`,
`delay_rate3/4`) এখন extended wait থেকে আসে।

নতুন decomposition key: `W_consult3/4/total`, `W_device3/4/total`,
`device_share3/4`, `n_completed3/4`।

**বাদ দেওয়া হয়েছে** (তোমার সিদ্ধান্ত অনুযায়ী): `W_init3_visitonly`,
`W_init4_visitonly` — পুরনো classic কলামগুলো। `W_follow` রাখা হয়েছে,
কারণ ওটা alternative metric না, একটা stage-level diagnostic (follow-up
queue-তে কতক্ষণ বসে থাকতে হয়)।

### ৩. Docstring

Revision notes v3 যোগ — কী বদলালো, কেন, আর কী গোনা হয় *না* (τⱼ, δⱼ)।
সাথে সতর্কবাণী যে পুরনো (k1, k2) আর optimal না।

---

## `experiments.py` — প্রায় পুরোটা নতুন

### ৪. `--jobs N` — parallel execution

প্রতিটা replication এখন একটা plain dict ("job")। `run_jobs()` সেগুলো
serial বা `multiprocessing.Pool`-এ চালায়।

- ফলাফল **হুবহু একই** যেকোনো `--jobs` মানে: প্রতিটা replication আলাদা
  seed নিয়ে চলে, আর ফলাফল সবসময় input order-এ ফিরে আসে (`imap_unordered`
  + index)।
- Worker process থেকে পুরো result dict ফেরত আসে না, শুধু দরকারি key
  (`_keep`) — প্রতিটা রোগীর raw waiting time pickle করা এড়ানোর জন্য।
- `exam_order_policy` এখন **job-এর ভেতরে** যায়, module global থেকে পড়া হয়
  না। কারণ Windows/macOS-এ worker process spawn হয় আর module নতুন করে
  import করে — global override হারিয়ে যেত, `experiments_dynamic_queue.py`
  চুপচাপ ভুল (fixed-order) ফলাফল দিত।

### ৫. 🐛 Bug fix: কোনো flag ছাড়া চালালে কিছুই হতো না

```python
if not any(vars(args).values()): args.all = True    # আগে
```

`vars(args)`-এ `--k1`/`--k2`-র default (13.1, 2.1) থাকে, যা truthy — তাই
`any(...)` সবসময় True, `args.all` কখনো True হতো না। `python experiments.py`
চুপচাপ exit করত। এখন শুধু action flag-গুলো দেখা হয় (`ACTION_FLAGS`)।

### ৬. নতুন: `table10_wait_decomposition.csv`

Strategy × level অনুযায়ী consultation-wait | device-wait | extended total |
device share (%)। extension-এর মূল ফলাফলের টেবিল।

### ৭. নতুন: grid search-এর সম্পূর্ণ ফলাফল সেভ

`table3/4_..._full.csv` — শুধু top-5 না, প্রতিটা (k1,k2) ঘরের mean wait,
SD, SL3, SL4। heat-map বা contour আঁকতে চাইলে এটা লাগবে।

### ৮. নতুন: `--smoke`, `--reps-grid`, `--reps-final`, `--reps-sensitivity`

`--smoke` সব ডাইমেনশন ছোট করে দেয় (২-৩ মিনিটে পুরো pipeline), যাতে লম্বা
রান শুরু করার আগে plumbing যাচাই করা যায়।

### ৯. Grid search পুনর্গঠন

- সব (k1,k2) × seed একসাথে একটা flat job list — parallel efficiency অনেক ভালো।
- coarse grid geometry এখন constant (`K1_RANGE`, `K2_RANGE`, `COARSE_STEP`,
  `FINE_STEP`, `FINE_HALFWIDTH`) — hardcoded `range()` না।
- SL constraint চেক এখন extended SL3/SL4 দিয়ে।

### ১০. ছোট জিনিস

- `matplotlib.use("Agg")` — SSH/server-এ display ছাড়া চলবে।
- grid search স্কিপ করলে log-এ সতর্কবার্তা: "এই (k1,k2) শুধু তখনই optimal
  যদি একই metric-এ grid search থেকে এসে থাকে।"
- axis label-এ লেখা "physician + device queueing", যাতে ফিগার দেখেই বোঝা
  যায় কোন metric।

---

## `experiments_comparison.py`

### ১১. 🐛 Bug fix: monkey-patch সরানো হয়েছে

স্ক্রিপ্টটা `Dispatcher._exam_process`-কে নিজের instrumented ভার্সন দিয়ে
replace করত। কিন্তু সেটা `patient.dev_queue_time` **সেট করত না** (শুধু
local variable রাখত)। extended metric চালু হওয়ার পর এর ফল হতো: এই
স্ক্রিপ্টে (আর শুধু এই স্ক্রিপ্টেই) প্রতিটা রোগীর device-wait = 0, অর্থাৎ
নীরবে classic metric-এ ফিরে যাওয়া — কোনো error ছাড়াই ভুল সংখ্যা।

engine নিজেই এখন হুবহু একই tuple `exam_log`-এ রাখে, তাই পুরো patch-টাই
অপ্রয়োজনীয় ছিল। সরিয়ে দেওয়া হয়েছে — divergence-এর সুযোগই নেই।

### ১২. Threshold pair আর hardcoded না

`K_FIXED = (3.3, 9.0)` আর `K_CAWARE = (12.0, 24.0)` ছিল পুরনো metric-এর
optimum। এখন `--k-fixed` / `--k-caware` CLI argument। default ওই পুরনো
মানেই, কিন্তু default ব্যবহার করলে স্ক্রিপ্ট নিজেই সতর্ক করে দেয়।

### ১৩. `--jobs` support + নতুন কলাম

`experiments.py`-র job runner ব্যবহার করে। `per_level_waiting.csv`-এ নতুন
`W_device` কলাম, আর `W_followup` → `W_followup_queue` (নামটা স্পষ্ট করতে)।

---

## `experiments_dynamic_queue.py`

### ১৪. Monkey-patch → explicit policy

আগে `exp.run_simulation`-কে wrapper দিয়ে replace করা হতো। এখন শুধু
`exp.EXAM_ORDER_POLICY = "shortest_wait"` — আর প্রতিটা job সেই policy
নিজের ভেতরে বহন করে, তাই multiprocessing-এ (spawn mode-এও) নিরাপদ।

### ১৫. একই bug fix + `--jobs`/`--smoke`

`any(vars(args).values())` বাগটা এখানেও ছিল, ঠিক করা হয়েছে (`--compare-plots`
সহ action flag list)। `experiments.py`-র parser reuse করায় সব নতুন flag
এখানেও পাওয়া যায়।

### ১৬. Docstring সংশোধন

default `--k1/--k2` আর docstring-এর উদাহরণ আগে আলাদা ছিল (13.1/2.1 vs
3.3/9.0)। এখন সামঞ্জস্যপূর্ণ, আর স্পষ্ট করে লেখা যে প্রতিটা exam-model-এর
নিজস্ব optimum আছে।

---

## যা **করা হয়নি** (ইচ্ছাকৃত)

- **এক রানে দুই metric পাশাপাশি দেখানো হয় না** — প্রতিটা রান একটাই
  definition ব্যবহার করে, আর সেটা Table 10-এর `Metric` / `Device_Counted`
  কলামে লেখা থাকে। তুলনা করতে হয় দুই ফোল্ডারের একই ফাইল পাশাপাশি রেখে।
- **"constraint extended vs classic" robustness টেবিল বানানো হয়নি** — চাইলে
  `experiments.py --grid-search --wait-metric extended` আর
  `--wait-metric classic` আলাদা চালিয়ে দুটো optimal (k1,k2) নিজেই মিলিয়ে
  নিতে পারো।
- **Scheduling logic-এ কোনো পরিবর্তন নেই** — IFP, ALT, SBP-র সিদ্ধান্তের
  নিয়ম অপরিবর্তিত। SBP-র urgency clock আগে থেকেই arrival time থেকে চলে,
  তাই সেটা এমনিতেই device-wait গোনে; নতুন metric-এর সাথে consistent।
- **Engine-এর random draw-এর ক্রম বদলায়নি** — তাই একই seed-এ patient flow
  হুবহু আগের মতোই; শুধু মাপার নিয়ম বদলেছে।
