# Scaling a single-node pipeline out

A trip-analytics pipeline written three ways — **single-process pandas**, **multi-process
pandas on one box**, and **Spark on a cluster of pinned single-core nodes** — with a
benchmark harness that measures what each rewrite actually bought.

The interesting part is not that Spark exists. It is *when distribution wins, when it
loses, and how to measure the difference without fooling yourself*. The suite therefore
reports the configurations where the distributed version is **slower**, and the one where
the single-node version does not merely lose but **dies**.

Everything is free and open source: Python, pandas, PyArrow, Apache Spark, Docker Compose,
matplotlib. No cloud account, one laptop.

Read [`DESIGN.md`](DESIGN.md) first — it was written before the code.

---

## Quickstart

```bash
make build             # one image, every role
make up                # master + 6 pinned worker nodes + driver + baselines
make gen SCALE=m       # 10M synthetic trips as 32 Parquet files
make calibrate         # measure the HOST's parallel ceiling  <-- do not skip
make bench             # the full suite -> results/results.json
make plot              # charts + results/RESULTS.md
```

Then the demos:

```bash
make demo-fault        # kill a node mid-job; output must stay byte-identical
make demo-skew         # hot-key shuffle join, with and without salting
make plan              # dump the physical plan: broadcast join, filter pushdown
```

`make help` lists everything.

---

## The three implementations

The pipeline is identical in all three: read Parquet → drop invalid trips → derive
duration/speed/tip% → join a 265-row zone dimension → aggregate by (borough, hour) →
rank the top 5 pickup zones per borough → write.

| Tier | File | Parallelism | Memory model |
|---|---|---|---|
| **T1** | `src/baseline/pandas_pipeline.py` | 1 core | whole dataset in RAM |
| **T2** | `src/baseline/chunked_pipeline.py` | P processes, 1 machine | one partition at a time |
| **T3** | `src/distributed/spark_pipeline.py` | N nodes × 1 core | streams, spills to disk |

T2 is the honest middle rung. Comparing naive pandas straight to Spark would credit
distribution with a win that mostly came from using more than one core. **T1→T2 is
vertical scaling; T2→T3 is horizontal.**

T2 is also a distributed system in miniature — map each partition to a *partial
aggregate*, reduce the partials — which is exactly what T3 does with the process boundary
moved across machines. Writing it makes Spark's map/shuffle/reduce structure obvious
instead of magic.

## Architecture

```
                  shared storage (bind-mounted volume, read by every node)
                  /data/trips/part-*.parquet · /data/zones · /data/out
                         ▲            ▲            ▲            ▲
   ┌──────────────┐      │            │            │            │
   │ spark-client │  ┌───┴───┐   ┌────┴───┐   ┌────┴───┐   ┌────┴───┐
   │ driver, on   │◄─┤ node1 │   │ node2  │   │ node3  │   │ node4  │
   │ the P-cores  │  │ 1 core│   │ 1 core │   │ 1 core │   │ 1 core │
   └──────┬───────┘  │ 2 GB  │   │ 2 GB   │   │ 2 GB   │   │ 2 GB   │
          │          └───┬───┘   └────┬───┘   └────┬───┘   └────┬───┘
          │        ┌─────┴────────────┴────────────┴────────────┘
          └───────►│  spark-master (scheduler)
                   └──────────────────────────────────────────────
   control group:  baseline-big — 4 cores / 8 GB in ONE box,
                   the same total resources as the 4-node cluster.
```

Every node is one **dedicated physical core** + 2 GB. The only way to add capacity is to
add a node. `baseline-big` is the control group that keeps "distributed is faster" a
falsifiable claim rather than a slogan: same silicon, different topology.

---

## What the benchmark found

Full tables in [`results/RESULTS.md`](results/RESULTS.md); charts in `results/`.
Host: i7-1365U, node = 1 pinned core + 2 GB. Median of 3 timed runs, warm-up discarded.

![strong scaling](results/strong_scaling.png)

**Scaling gains plateau; the coordination tax is what shrinks.** Adding data does not keep
buying scalability. The 4-node speedup goes 0.80x (10M) → 2.19x (30M) → **2.24x (300M)**:
it saturates around 2.2x, or 56% efficiency against the host's own 3.94x ceiling, and 300M
rows buys essentially nothing over 30M. Since the plateau is identical whether or not the
data fits in page cache, it is structural — shuffle and coordination — not disk.

What *does* improve monotonically is how much distribution costs you in absolute terms:

| dataset | one 4-core box | Spark, 4 nodes | Spark is |
|---|---|---|---|
| 10M rows | 4.2s | 46.2s | 11.1x slower |
| 30M rows | 9.0s | 32.1s | 3.6x slower |
| **300M rows** | **72.5s** | **112.2s** | **1.6x slower** |

Same code, same total cores, only the job size changed. The fixed costs — JVM startup,
scheduling, plan construction — are a fatal overhead at 10M and nearly amortised by 300M.

> **Correction.** An earlier version of this README read the 30M column as a steady-state
> result and reported a "3.6x coordination tax", concluding the cluster would need ~3.6x
> more cores than the biggest single box to break even. The 300M run shows that number was
> inflated by fixed costs. The steady-state tax is **1.55x** (1.04M vs 0.67M rows/s/core),
> so the real break-even is about **6 nodes**, not 15. Measuring one point and calling it
> an asymptote was the mistake.

**The single-node control group still wins, and that is the point of having it.**

| 300M rows | compute | wall |
|---|---|---|
| T1 pandas, 1 node | **OOM-killed** (skipped at this scale) | — |
| T2 chunked, 1 node (1 core) | 209.6s | 209.6s |
| **T2 chunked, one 4-core box** | **72.5s** | **72.5s** |
| T3 spark, 1 node | 251.4s | 270.4s |
| T3 spark, 2 nodes | 139.6s | 157.2s |
| T3 spark, 4 nodes | 112.2s | 131.3s |

Four cores in one box still beat four one-core nodes on identical total resources — but by
1.55x, not the 11x seen at 10M. (Persisting the joined frame to make one pass instead of
two was measured too, and was *worse*: 86s vs 22s at 1 node, because the frame does not fit
in a 1.4 GB executor and spills.)

So the honest guidance on this hardware: **if your data fits on one machine, put it on one
machine** — but the penalty for being wrong about that shrinks fast as the job grows.

One caveat specific to the 300M run: at ~7 GB the dataset exceeds spare page cache, so it
is the first size measuring real disk I/O — and all four simulated nodes share one NVMe
where a real cluster would have four. That asymmetry favours the single-box tiers, and the
1.55x figure should be read as a floor for the cluster, not a verdict.

**Where the single machine simply cannot follow.** At 30M rows on a 2 GB node, T1 is
OOM-killed (exit 137, `OOMKilled=true`) because peak RSS is O(dataset). T2 peaks at 366 MB
and T3 finishes, because both stream partitions instead of materialising the whole frame.
That is a capability difference, not a speed difference, and it is the only argument for
distribution that survives this benchmark.

![memory ceiling](results/memory_ceiling.png)

**Small data: distribution is a pure loss.** At 200K rows, pandas takes 0.6s and the
4-node cluster takes 30s of compute inside 67.6s wall — about **100x slower** for the same
answer.

![overhead](results/overhead.png)

**Losing a node costs correctness nothing.** `make demo-fault` kills 25% of the cluster
mid-job. The job finishes and the answer is unchanged — every integer column exactly
equal, floats within rtol 1e-9. It is *not* byte-identical, and that is expected:
recomputed partitions are combined in a different order and float addition is not
associative. Byte-equality was the demo's original pass condition; it was the wrong bar.

**Skew hurts in proportion to how parallel you are.** A job cannot finish before its
largest task does, so makespan ≥ `max(hot_share, 1/cores)` of the total work — salting can
win back at most `max(hot_share x cores, 1)`:

| hot key holds | predicted ceiling | salting measured |
|---|---|---|
| 30% of rows | 1.21x | 1.01x — not worth its own shuffle |
| 85% of rows | 3.40x | **1.75x** |

On 4 cores a 30% hot key is nearly harmless (30% of the work versus 25% if perfectly
balanced) and salting costs more than it saves. The same key on a 100-node cluster leaves
99 nodes waiting. **Skew is not a property of the data alone; it is a property of the data
and the cluster width together.**

---

## Principles, and where to see each one

| Principle | Where it is visible |
|---|---|
| Partitions are the unit of parallelism | `scan_partitions` in every result; the default file-packing bug below |
| Narrow vs wide transformations | filter/derive scale; the two shuffles are where efficiency goes |
| Avoid the shuffle you can avoid | `make plan` → `BroadcastHashJoin`, not `SortMergeJoin` |
| Lazy evaluation, pushdown | same plan dump → `PushedFilters` |
| Amdahl's law | `results/strong_scaling.png`: measured curve vs ideal vs host ceiling |
| Horizontal beats vertical *past a point* | `results/memory_ceiling.png` — T1 is OOM-killed, T3 finishes |
| Fault tolerance | `make demo-fault` — kill 25% of the cluster mid-job, byte-identical output |
| Data skew | `make demo-skew` — one key with 30% of rows, plain vs salted join |
| Correctness before speed | `make parity` gates the benchmark; ints exact, floats to rtol 1e-9 |
| Benchmark honesty | warm-up discarded, median of 3, min/max kept, failures reported |

---

## Three things this project got wrong first

Kept in, because they are the actual content.

**1. A bigger cluster looked *slower*.** The first action of a job was paying for executor
JVMs still booting, so a 4-node run "measured" 4 JVMs starting up. Fixed by
`spark.scheduler.minRegisteredResourcesRatio=1.0` plus an explicit
[`await_cluster`](src/distributed/spark_pipeline.py) barrier and a warm-up action before
the clock starts. Startup is still reported — as its own bar, because it never shrinks.

**2. 64 input files became 8 tasks.** Spark packs small Parquet files together up to
`spark.sql.files.maxPartitionBytes` (128 MB default). 246 MB of input collapsed into 8
partitions, so a 4-node cluster had nothing to do. **Your parallelism budget is your
partition count, not your core count.**

**3. The machine, not the software.** Every scaling curve came out flat. The host is an
i7-1365U: 2 fast P-cores + 8 slow E-cores under a 15 W package limit, where *two*
unpinned CPU-bound threads run **~3× slower** than one. Pinning each node to its own
E-core (`cpuset` in the compose file) restored near-linear host scaling — so
`make calibrate` now runs first and every speedup is plotted against that measured
ceiling as well as against ideal linear.

That third one is the transferable lesson: **calibrate the machine before you blame the
software.** A flat scaling curve is more often the hardware than the framework.


### Task slots: oversubscribing one core

`make bench EXPERIMENTS=slots` puts N task slots on a single pinned core and races it
against every other approach. Shuffle partitions are pinned at 16 across all Spark rows so
slot count is the only variable. 10M rows, median of 3 timed runs:

![task slots](results/slots.png)

| configuration | compute | wall |
|---|---|---|
| T1 pandas (1 core) | **OOM-killed** | — |
| T2 chunked (1 core) | 7.7s | 7.7s |
| **T2 chunked (4-core box)** | **2.5s** | **2.5s** |
| T3 spark (1 node, 1 slot) | 24.8s | 39.3s |
| T3 spark (1 node, 2 slots) | 25.2s | 40.2s |
| T3 spark (1 node, 4 slots) | 25.5s | 41.3s |
| T3 spark (1 node, 10 slots) | 28.0s | 44.3s |
| T3 spark (4 nodes, 1 slot) | 22.7s | 44.1s |

**Slots do not create capacity.** Going 1 → 10 slots on one core is monotonically *worse*,
ending 13% slower, with run-to-run ranges tight enough (27.4-28.8s at 10 slots) that it is
signal, not noise. A slot is a thread inside one executor JVM; ten threads on one core
timeslice the same silicon while adding context switches, ten sets of task bookkeeping, and
ten concurrent shuffle writers competing for one 1400 MB heap.

Oversubscription pays only when slots *wait* rather than compute — remote object-store
reads, or stragglers leaving cores idle. This pipeline is CPU-bound on a warm page cache,
so every extra slot is pure overhead. Real capacity came only from the real fourth node
(22.7s), and even that is beaten 9x by four cores in one box.

Related limits, both discovered the hard way: a standalone worker fits
`floor(worker_memory / executor_memory)` executors, and Spark rejects any
`spark.executor.memory` below 450 MB. A 1600 MB worker therefore tops out at three executor
*processes* — and three JVMs sharing one core measured 45.3s, far worse than ten threads.

---

## Layout

```
DESIGN.md                        design, written first
docker/                          one Dockerfile, one compose file (nodes are pinned here)
src/common/config.py             paths, scales, business rules — one source of truth
src/gen/generate.py              synthetic data, skew-controlled
src/baseline/{pandas,chunked}_pipeline.py    T1, T2
src/distributed/spark_pipeline.py            T3
src/distributed/skew_demo.py                 hot-key join, plain vs salted
src/bench/{runner,parity,plot,fault_demo}.py harness, correctness gate, charts, chaos
results/                         results.json, RESULTS.md, *.png, spark_plan.txt
```

## Non-goals

Streaming, a workflow scheduler, real multi-host networking, autoscaling, cost modelling.
Each would add infrastructure without adding a lesson.
