"""T3 — horizontal scaling: the same pipeline as a Spark job.

The code is deliberately a near-line-for-line translation of T1 so the diff you would
show a reviewer is about *execution model*, not about business logic:

  pandas                      spark
  ---------------------------------------------------------------
  read every file into RAM -> lazy scan, one task per file
  boolean mask             -> filter, pushed down into the Parquet reader
  assign columns           -> narrow transformation, no data movement
  merge(zones)             -> BROADCAST join: replicate the 265-row side, no shuffle
  groupby                  -> map-side partial agg + shuffle + reduce
  sort + cumcount          -> window function over a shuffled, sorted partition
  to_csv                   -> collect the ~150-row result to the driver

Only the final, tiny result crosses back to the driver. Moving 10 GB to one process to
call .to_csv() would rebuild the bottleneck the rewrite was meant to remove.
"""
import argparse
import json
import os
import sys
import time

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import config as C  # noqa: E402


def build_session(app: str, shuffle_partitions: int,
                  max_partition_bytes: int = 16 * 1024 * 1024) -> SparkSession:
    b = (SparkSession.builder.appName(app)
         .master(os.environ.get("SPARK_MASTER_URL", "local[*]"))
         .config("spark.sql.shuffle.partitions", shuffle_partitions)
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
         .config("spark.sql.parquet.filterPushdown", "true")
         # Parallelism is capped by PARTITION count, not core count. Spark packs
         # small input files together up to maxPartitionBytes (128 MB by default),
         # which silently collapsed 64 files into 8 tasks -- a 4-node cluster then
         # has nothing to do. Keep one partition per file.
         .config("spark.sql.files.maxPartitionBytes", str(max_partition_bytes))
         # Don't start scheduling until the cluster we asked for actually exists,
         # otherwise a 4-worker run measures 4 JVMs booting, not 4 nodes computing.
         .config("spark.scheduler.minRegisteredResourcesRatio", "1.0")
         .config("spark.scheduler.maxRegisteredResourcesWaitingTime", "60s")
         # Only affects how much of the plan is printed, never execution. Without
         # it Spark elides the pushed-filter list at 100 chars and `make plan` can
         # no longer show what it claims to show.
         .config("spark.sql.maxMetadataStringLength", "2000")
         .config("spark.ui.showConsoleProgress", "false"))
    for key, env in (("spark.driver.host", "SPARK_DRIVER_HOST"),
                     ("spark.executor.memory", "SPARK_EXECUTOR_MEMORY"),
                     ("spark.executor.cores", "SPARK_EXECUTOR_CORES"),
                     ("spark.cores.max", "SPARK_CORES_MAX")):
        if os.environ.get(env):
            b = b.config(key, os.environ[env])
    return b.getOrCreate()


def await_cluster(spark, expected: int, timeout_s: int = 90) -> int:
    """Block until every expected executor has registered, then warm the JVMs.

    Without this, a cold run attributes executor startup to the first stage, which
    makes a *bigger* cluster look *slower* -- the classic way to accidentally
    disprove your own scaling result.
    """
    sc = spark.sparkContext
    deadline = time.time() + timeout_s
    live = 0
    while time.time() < deadline:
        live = max(sc._jsc.sc().getExecutorMemoryStatus().size() - 1, 0)
        if expected and live >= expected:
            break
        time.sleep(0.5)
    spark.range(10_000).repartition(max(live, 1) * 2).count()  # JIT + task warm-up
    return live


def transform(spark):
    """Stages 1-4. Lazy: nothing has executed when this returns."""
    trips = spark.read.parquet(C.TRIPS)
    dur = F.col("dropoff_ts") - F.col("pickup_ts")

    clean = trips.filter(
        F.col("trip_distance_km").between(C.MIN_DISTANCE_KM, C.MAX_DISTANCE_KM)
        & (F.col("fare_amount") >= C.MIN_FARE)
        & dur.between(C.MIN_DURATION_S, C.MAX_DURATION_S)
    )
    derived = (clean
               .withColumn("duration_min", dur / 60.0)
               .withColumn("speed_kmh", F.col("trip_distance_km") / (dur / 3600.0))
               .withColumn("total_amount", F.col("fare_amount") + F.col("tip_amount"))
               .withColumn("tip_pct", F.col("tip_amount") / F.col("fare_amount") * 100.0)
               .withColumn("pickup_hour", F.floor((F.col("pickup_ts") % 86400) / 3600)))

    zones = spark.read.parquet(C.ZONES).select("zone_id", "borough")
    # F.broadcast is the whole point: ship 265 rows to every executor instead of
    # shuffling hundreds of millions of trip rows across the network.
    return derived.join(F.broadcast(zones),
                        derived.pickup_zone_id == zones.zone_id, "inner")


def aggregate(df):
    """Stage 5 — the wide transformation. This is where the network gets used."""
    return (df.groupBy("borough", "pickup_hour")
            .agg(F.count("trip_id").alias("trips"),
                 F.sum("total_amount").alias("total_revenue"),
                 F.avg("fare_amount").alias("avg_fare"),
                 F.avg("tip_pct").alias("avg_tip_pct"),
                 F.avg("speed_kmh").alias("avg_speed_kmh"),
                 F.max("trip_distance_km").alias("max_distance_km"),
                 F.countDistinct("vendor_id").alias("n_vendors"))
            .orderBy("borough", "pickup_hour")
            .select(*C.AGG_COLS))


def top_zones(df):
    """Stage 6 — window function: shuffle by borough, sort, rank, cut at N."""
    per_zone = (df.groupBy("borough", "pickup_zone_id")
                .agg(F.sum("total_amount").alias("zone_revenue"),
                     F.count("trip_id").alias("zone_trips"))
                .withColumnRenamed("pickup_zone_id", "zone_id"))
    w = Window.partitionBy("borough").orderBy(
        F.col("zone_revenue").desc(), F.col("zone_id").asc())
    return (per_zone.withColumn("rank", F.row_number().over(w))
            .filter(F.col("rank") <= C.TOP_N_ZONES)
            .orderBy("borough", "rank")
            .select(*C.TOP_COLS))


def summarise_plan(plan: str) -> None:
    """Print what the plan proves, so `make plan` is evidence and not homework."""
    import re
    join = "BroadcastHashJoin" if "BroadcastHashJoin" in plan else "SortMergeJoin"
    trips = next((l for l in plan.splitlines()
                  if "FileScan" in l and "/data/trips" in l), "")
    cols = re.search(r"FileScan parquet \[([^]]*)\]", trips)
    pushed = re.search(r"PushedFilters: \[([^]]*)\]", trips)
    n_read = len(cols.group(1).split(",")) if cols else 0
    print(f"  join strategy      : {join}"
          f"{'  <- 265-row dimension replicated, no shuffle' if 'Broadcast' in join else ''}")
    print(f"  columns read       : {n_read} of 11 (projection pushdown)")
    print(f"  filters pushed into the Parquet reader: "
          f"{pushed.group(1) if pushed else 'none'}")


def run(out_dir: str, shuffle_partitions: int = 32, staged: bool = False,
        dump_plan: str = "", expect_executors: int = 0, cache: bool = False,
        max_partition_bytes: int = 16 * 1024 * 1024) -> dict:
    t_start = time.perf_counter()
    spark = build_session("scaling-demo", shuffle_partitions, max_partition_bytes)
    live = await_cluster(spark, expect_executors)
    t_startup = round(time.perf_counter() - t_start, 3)

    t_compute = time.perf_counter()
    joined = transform(spark)
    if cache:
        # The joined frame feeds two different aggregations. pandas holds it in RAM
        # for free; Spark would otherwise re-scan and re-filter the whole input once
        # per action, so persisting is what makes the comparison like-for-like.
        joined = joined.persist(StorageLevel.MEMORY_AND_DISK)
    timings = {"startup": t_startup}

    if staged:
        # Diagnostic mode: force the narrow stages to materialise so their cost can
        # be separated from the shuffle. Slower than the benchmark path by design.
        joined = joined.persist()
        t = time.perf_counter()
        timings["narrow_stages"] = None
        joined.count()
        timings["narrow_stages"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    agg_df = aggregate(joined)
    agg = agg_df.toPandas()
    timings["aggregate"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    top = top_zones(joined).toPandas()
    timings["top_zones"] = round(time.perf_counter() - t, 3)

    if dump_plan:
        os.makedirs(os.path.dirname(dump_plan) or ".", exist_ok=True)
        plan = ("=== aggregate ===\n" + agg_df._jdf.queryExecution().toString()
                + "\n\n=== top_zones ===\n"
                + top_zones(joined)._jdf.queryExecution().toString())
        with open(dump_plan, "w") as fh:
            fh.write(plan)
        summarise_plan(plan)

    t = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(f"{out_dir}/agg.csv", index=False)
    top.to_csv(f"{out_dir}/top_zones.csv", index=False)
    timings["write"] = round(time.perf_counter() - t, 3)

    timings["compute"] = round(time.perf_counter() - t_compute, 3)
    scan_parts = spark.read.parquet(C.TRIPS).rdd.getNumPartitions()
    sc = spark.sparkContext
    total_cores = int(sc.defaultParallelism)
    # rows_in is metadata-only for Parquet; rows_kept falls out of the aggregate for
    # free, so neither number costs an extra pass over the data.
    rows_in = spark.read.parquet(C.TRIPS).count()
    rows_kept = int(agg["trips"].sum())
    spark.stop()

    return dict(tier="spark", wall_s=round(time.perf_counter() - t_start, 3),
                compute_s=timings["compute"], startup_s=timings["startup"],
                rows_in=rows_in, rows_kept=rows_kept, stages=timings,
                executors=live, default_parallelism=total_cores,
                scan_partitions=scan_parts, cached=cache,
                shuffle_partitions=shuffle_partitions)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{C.OUT}/spark")
    ap.add_argument("--shuffle-partitions", type=int, default=32)
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--dump-plan", default="")
    ap.add_argument("--expect-executors", type=int,
                    default=int(os.environ.get("SPARK_EXPECT_EXECUTORS", 0)))
    ap.add_argument("--cache", action="store_true",
                    help="persist the joined frame instead of re-scanning per action")
    ap.add_argument("--max-partition-bytes", type=int, default=16 * 1024 * 1024)
    a = ap.parse_args()
    print(json.dumps(run(a.out, a.shuffle_partitions, a.staged, a.dump_plan,
                         a.expect_executors, a.cache, a.max_partition_bytes)))
