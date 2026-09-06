# Design: From One Machine to Many

**Goal.** Take a real analytics pipeline that runs as a single-process pandas script on one
machine, re-architect it to run distributed on a Spark cluster, and *measure* the difference
under controlled conditions — not just "it's faster", but *how* it scales, *where* it stops
scaling, and *what the distributed version buys you that a bigger machine cannot*.

Everything here is free and open source: Python, pandas, PyArrow, Apache Spark, Docker Compose,
matplotlib. No cloud account required.

---

## 1. The workload

A trip-analytics pipeline over synthetic ride-hailing data (NYC-taxi shaped). It is deliberately
chosen to contain one of *each* class of distributed-computing operation, so the benchmark
exercises the interesting parts of the system rather than a single trivially-parallel map.

**Inputs**
| Dataset | Shape | Role |
|---|---|---|
| `trips` (fact) | N rows × 11 cols, Parquet, many files | The big table. Scales with `SCALE`. |
| `zones` (dim) | 265 rows × 3 cols | Small lookup. Forces a join decision. |

**Stages**

| # | Stage | Operation class | Why it's in the pipeline |
|---|---|---|---|
| 1 | Read Parquet | I/O, columnar | Demonstrates projection + predicate pushdown |
| 2 | Filter invalid trips | **Narrow** (map-only) | Embarrassingly parallel — the easy win |
| 3 | Derive columns (duration, speed, tip %) | **Narrow** | Pure CPU, scales linearly |
| 4 | Join `zones` | **Broadcast join** | Small side is replicated, *avoids* a shuffle |
| 5 | Group by (borough, hour) | **Wide / shuffle** | Network-bound; the real scaling limiter |
| 6 | Top-5 zones per borough by revenue | **Window over shuffle** | Shuffle + sort, the most expensive stage |
| 7 | Write Parquet + CSV summary | I/O | Parallel write, one file per partition |

The output is ~1 KB of aggregates from GBs of input — the classic funnel shape that makes
distribution worthwhile at all.

---

## 2. Three implementations (deliberately three, not two)

Jumping straight from "naive pandas" to "Spark" overstates the win: most of the speedup would
come from using more than one core, not from distribution. So the benchmark has a middle rung.

| Tier | Implementation | Parallelism | Memory model | Ceiling |
|---|---|---|---|---|
| **T1** | `baseline/pandas_pipeline.py` | 1 core | Whole dataset in RAM | Dies when data > RAM |
| **T2** | `baseline/chunked_pipeline.py` | `multiprocessing`, P cores, 1 machine | Streams chunks, partial aggregates | Bounded by one box's cores/RAM/disk |
| **T3** | `distributed/spark_pipeline.py` | K executors × C cores, N machines | Streams + spills to disk | Bounded by network / shuffle |

T2 is the honest comparison point. **T1 → T2 is vertical scaling. T2 → T3 is horizontal
scaling.** Reporting only T1 → T3 would be a benchmark lie, and the results section says so
explicitly.

Notably, T2 is already a *distributed system in miniature*: partition the input, map, combine
partial aggregates. Spark is the same algorithm with the process boundary moved across machines.
Writing T2 first makes the map/shuffle/reduce structure of T3 obvious rather than magic.

---

## 3. Architecture

```
                       ┌──────────────────────────────────────────┐
                       │  Shared storage layer (bind-mounted vol)  │
   Storage/compute     │  /data/trips/part-*.parquet   (N files)   │
   separation: compute │  /data/zones/zones.parquet    (265 rows)  │
   scales without      │  /data/out/...                            │
   moving the data     └───────────▲──────────────▲────────────▲───┘
                                   │              │            │
        ┌──────────────┐    ┌──────┴─────┐ ┌──────┴─────┐ ┌────┴───────┐
        │ spark-client │    │  worker-1  │ │  worker-2  │ │  worker-N  │
        │ (driver,     │◄──►│ 2 cores    │ │ 2 cores    │ │ 2 cores    │
        │  builds DAG) │    │ 2 GB       │ │ 2 GB       │ │ 2 GB       │
        └──────┬───────┘    └────────────┘ └────────────┘ └────────────┘
               │                   ▲              ▲            ▲
               │            ┌──────┴──────────────┴────────────┴───┐
               └───────────►│        spark-master (scheduler)       │
                            └──────────────────────────────────────┘
```

Each Spark worker is a separate container with its **own CPU and memory limit** — a stand-in for
a separate machine. Adding capacity means adding containers, never resizing one. That is the
whole point of the exercise, and it is what `--scale spark-worker=N` does.

The baseline containers get the **same per-node limits** (2 cores / 2 GB), so the comparison is
"one node vs N nodes", not "small box vs big box".

### Why a shared volume instead of HDFS/S3
Storage/compute separation matters more than *which* storage. A bind-mounted volume readable by
every worker gives the same property (compute nodes are stateless and disposable; data outlives
them) with one moving part instead of five. A MinIO + `s3a://` profile is included in the compose
file for anyone who wants the object-store version — the pipeline code only ever sees a URI, so
switching is a config change, not a rewrite.

---

## 4. Principles being demonstrated (and how each is made visible)

| Principle | How the project shows it |
|---|---|
| **Partitioning is the unit of parallelism** | Input is written as many Parquet files; the scaling curve flattens once `partitions < total cores` |
| **Narrow vs wide transformations** | Stages 2–4 scale near-linearly; stage 5–6 (shuffle) is where efficiency is lost. Per-stage timings are recorded separately |
| **Avoid the shuffle when you can** | The `zones` join is a broadcast join; the plan is dumped to `results/spark_plan.txt` showing `BroadcastHashJoin`, not `SortMergeJoin` |
| **Lazy evaluation / whole-query optimization** | Spark builds a DAG and pushes filters into the Parquet reader; the plan output shows `PushedFilters` |
| **Amdahl's law** | Speedup is plotted against ideal linear speedup; the gap is the serial fraction (driver planning, final collect, write commit) |
| **Strong vs weak scaling** | Strong: fixed data, more workers. Weak: data grows with workers — the curve that matters for "can I keep up with growth" |
| **Horizontal beats vertical past a point** | The memory-ceiling test: with a dataset larger than one node's RAM, T1 **fails** and T3 **succeeds**. A capability difference, not a speed difference |
| **Fault tolerance / disposable compute** | `make demo-fault` kills a worker mid-job; Spark re-schedules the lost partitions and the job still produces byte-identical output |
| **Data skew is the classic distributed failure mode** | The generator makes one zone hold ~30% of trips; `--salt` enables salted aggregation. Both timings are reported |
| **Correctness before speed** | `bench/parity.py` asserts T1, T2 and T3 outputs match within float tolerance. A fast wrong answer scores zero |

---

## 5. Benchmark methodology

Fair-benchmark rules, enforced by the runner:

1. **Same data, same output.** Parity check runs before the timings are published.
2. **Cold-ish cache.** Page cache is dropped between runs where permitted; otherwise every tier
   pays the same warm-cache cost and the caveat is recorded in the results.
3. **Discard the first run** (JVM JIT warm-up, Parquet footer caching), then take the **median of
   3**, and report min/max as error bars.
4. **Per-node resources held constant** across tiers (2 cores / 2 GB).
5. **Report the failures too** — including the configurations where distribution *loses*.

**Metrics:** wall-clock, rows/sec, peak RSS (baseline tiers), speedup `S(N)=T₁/T_N`, parallel
efficiency `E(N)=S(N)/N`, and per-stage breakdown.

**Experiments**
| Name | Independent variable | Question it answers |
|---|---|---|
| `strong` | workers ∈ {1,2,4}, data fixed | How much faster on the same job? |
| `weak` | workers ∈ {1,2,4}, data ∝ workers | Can I keep up as data grows? |
| `small-data` | tiny dataset, all tiers | When is Spark *slower*? (spoiler: usually) |
| `memory-ceiling` | data > node RAM | What can the cluster do that one box cannot? |
| `skew` | skewed keys, ±salting | How does a bad key distribution destroy scaling? |

### The result I expect to have to explain
On a laptop-sized dataset Spark will **lose** to chunked pandas: ~5–10 s of JVM startup,
scheduling and shuffle-write dominate a job that takes seconds. That is not a bug in the
benchmark, it is the actual engineering lesson — distribution buys you *headroom and survivability*
and charges you *latency and complexity*, and the crossover point is a number you should measure
before you rewrite anything. The small-data experiment exists specifically to find that crossover
and put it in the README.

---

## 6. Layout

```
DESIGN.md              this document
Makefile               make setup / gen / bench / plot / demo-fault / clean
docker/
  Dockerfile           one image: Spark 3.5 + pandas/pyarrow/matplotlib
  docker-compose.yml   master, workers (scalable), client, optional minio
src/
  common/config.py     paths, scales, schema — one source of truth
  gen/generate.py      synthetic data generator (vectorised, skew-controlled)
  baseline/pandas_pipeline.py    T1
  baseline/chunked_pipeline.py   T2
  distributed/spark_pipeline.py  T3
  bench/runner.py      orchestrates experiments, writes results/results.json
  bench/parity.py      cross-tier correctness check
  bench/plot.py        speedup / efficiency / wall-clock charts
results/               results.json, *.png, spark_plan.txt, RESULTS.md
```

## 7. Non-goals
Streaming, a scheduler (Airflow), real multi-host networking, autoscaling, and cost modelling.
This project is about the scaling principles, and each of those would add infrastructure without
adding a lesson.

---

## 8. What changed once it met real hardware

The design above was written first and is left as written. These are the deltas, with the
measurement that forced each one — the gap between the two is most of what the project
turned out to be about.

| Designed | Built | Why |
|---|---|---|
| Nodes of 2 cores / 2 GB | **1 pinned core** / 2 GB | Unpinned, the host's P-core/E-core asymmetry made two threads ~3x slower than one and every scaling curve came out flat. Pinning each node to its own E-core restored 1.00 / 2.09 / 3.94x host scaling at 1 / 2 / 4 nodes. |
| `--scale spark-worker=N` | six named services with fixed `cpuset` | Compose gives every replica of a scaled service the same config, so replicas cannot be pinned to different cores. |
| Per-stage timings from the Spark job | one `startup` / `compute` split, plus a `--staged` diagnostic mode | Splitting a lazy DAG into timed stages requires materialising between them, which changes the thing being measured. Startup vs compute is the split that actually matters, because startup never shrinks with more nodes. |
| Optional MinIO / `s3a://` profile | shared bind mount only | The volume already gives storage/compute separation. MinIO would have added a service and two jars for no additional lesson. Cut. |
| Salting the borough/hour aggregation | salting a **join**, at two hot-key shares | Spark's map-side combine neutralises skew for decomposable aggregates like `sum` — the reducer gets one partial row per partition, so there was nothing to fix. Skew needs a join, or a non-decomposable aggregate, to bite. |
| Fault demo asserts byte-identical output | asserts equality within rtol 1e-9 | Recomputed partitions reassociate float sums. Ints stay exact. Bit-reproducibility is not what a lineage-based engine promises, and asserting it produces a false failure. |
| — | **`make calibrate`, run before anything else** | Not in the original design at all. It is now the first experiment, because three days of "Spark does not scale" was actually a 15 W laptop. |

Two bugs worth keeping in the record because both are silent and both look like
"distribution does not work":

1. **Cold executors.** The first action of a job paid for JVMs that were still registering,
   so a 4-node run measured 4 JVMs booting and came out *slower* than 1 node. Fixed with
   `spark.scheduler.minRegisteredResourcesRatio=1.0` and an explicit barrier plus warm-up
   before the clock starts.
2. **File packing.** Spark packs small Parquet files up to `maxPartitionBytes` (128 MB), so
   64 input files became 8 tasks and a 4-node cluster had nothing to do. Parallelism is
   bounded by partition count, not core count.
