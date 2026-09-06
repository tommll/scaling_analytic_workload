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

**Scalability is a property of the job size, not of the code.** At 30M rows Spark goes
70.3s → 35.9s → 32.1s on 1 → 2 → 4 nodes: **1.96x at two nodes** (98% efficiency, right on
the host's own measured ceiling) then flattening to 2.19x at four. Run the identical code
on 10M rows and the curve is flat, because the per-job fixed costs stop being noise.
Nothing about the pipeline changed between those two lines — only how much work each node
was given.

**The single-node control group wins, and that is the point of having it.**

| 30M rows | compute | wall |
|---|---|---|
| T1 pandas, 1 node | **OOM-killed** | — |
| T2 chunked, 1 node | 22.9s | 22.9s |
| **T2 chunked, one 4-core box** | **9.0s** | **9.0s** |
| T3 spark, 4 nodes | 32.1s | 53.9s |

Four cores in one box beat four one-core nodes by **3.6x on compute and 6x end-to-end**,
using identical total resources. Per core, Spark moved 234K rows/s against chunked
pandas's 833K — a **3.6x coordination tax** for shuffles, serialisation, JVM execution and
two passes over the input where pandas makes one. (Persisting the joined frame to get one
pass was measured too, and was *worse*: 86s vs 22s at 1 node, because the frame does not
fit in a 1.4 GB executor and spills.)

So the honest crossover on this hardware: **the cluster needs ~3.6x more cores than the
biggest single box to break even**, plus ~15-25s of startup on every job. If your data
fits on one machine, put it on one machine.

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


### Tuning note: task slots per core

Two different knobs get called "more parallelism on one node", and they behave nothing
alike. Measured on one pinned core, 10M rows:

| configuration | JVMs on the node | compute |
|---|---|---|
| 1 task slot (the default here) | 2 (worker + 1 executor) | 26.6s |
| 4 slots as **threads** in one executor (`--cores 4`, `executor.cores=4`) | 2 | 25.9s |
| 3 slots as separate **executor JVMs** (`executor.cores=1`, `cores.max=3`) | 4 (worker + 3 executors) | **45.3s** |

Task slots are *threads inside an executor JVM*, so raising `--cores` on a worker costs
almost nothing and buys almost nothing for CPU-bound work — the core is already saturated.
Splitting the same core across several executor *processes* is actively harmful: three JVMs
mean three heaps, three sets of GC threads and real context switching, for a 70% slowdown.

Oversubscription pays only when task slots spend time *waiting* rather than computing (I/O,
remote reads) or when a few stragglers leave cores idle — smaller, more numerous tasks
schedule more evenly. This pipeline is CPU-bound on a warm page cache, so it pays nothing.

Two hard limits worth knowing before you try: a standalone worker fits
`floor(worker_memory / executor_memory)` executors, and Spark refuses any
`spark.executor.memory` below 450 MB. A 1600 MB worker therefore tops out at three.


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
