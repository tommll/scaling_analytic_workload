"""Data skew: the failure mode that only appears once you distribute.

On one machine, an uneven key distribution costs you nothing -- one core processes
every key regardless. Distributed, the partition holding the hot key becomes a
straggler: 3 nodes finish in seconds and the whole job waits on the fourth. Adding
nodes does not help, because the problem is not throughput, it is that one task
cannot be split.

The generator gives zone 132 about 30% of all trips. This joins trips against a
per-zone campaign table (many rows per zone), which is exactly the shape that
explodes on a hot key, and compares:

  plain   -- sort-merge join, one reducer gets ~30% of the data
  salted  -- hot-side keys get a random suffix, small side is replicated across
             suffixes, so the hot key's work is spread over `salts` tasks

Broadcast is disabled on purpose. With a small dimension you would just broadcast it
and skew would not matter -- which is itself the first thing to try in real life.
"""
import argparse
import json
import os
import sys
import time

from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import config as C  # noqa: E402
from src.distributed.spark_pipeline import build_session, await_cluster  # noqa: E402

SALTS = 16


def campaigns(spark, per_zone: int = 40):
    """A dimension with several rows per zone: joining it multiplies the fact side."""
    return (spark.range(C.N_ZONES * per_zone)
            .withColumn("zone_id", (F.col("id") % C.N_ZONES + 1).cast("int"))
            .withColumn("campaign_budget", (F.col("id") % 97 + 1).cast("double"))
            .select("zone_id", "campaign_budget"))


def plain_join(trips, camps):
    return (trips.join(camps, trips.pickup_zone_id == camps.zone_id, "inner")
            .groupBy("pickup_zone_id")
            .agg(F.sum("campaign_budget").alias("budget"),
                 F.count("*").alias("n")))


def salted_join(trips, camps, salts: int = SALTS):
    """Split each key into `salts` sub-keys on the big side; replicate the small side.

    The hot key's rows are spread over `salts` reduce tasks instead of one. The cost
    is a `salts`-fold blow-up of the *small* side -- cheap, because it is small.
    """
    big = trips.withColumn("salt", (F.rand() * salts).cast("int"))
    small = (camps.withColumn("salt", F.explode(F.array(*[F.lit(i) for i in range(salts)]))))
    return (big.join(small, (big.pickup_zone_id == small.zone_id) & (big.salt == small.salt),
                     "inner")
            .groupBy("pickup_zone_id")
            .agg(F.sum("campaign_budget").alias("budget"),
                 F.count("*").alias("n")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=int(os.environ.get("SPARK_CORES_MAX", 4)))
    ap.add_argument("--salts", type=int, default=SALTS)
    ap.add_argument("--per-zone", type=int, default=200,
                    help="rows per zone in the dimension; controls how much the "
                         "join amplifies the hot key, i.e. how much skew hurts")
    ap.add_argument("--out", default="/results/skew.json")
    a = ap.parse_args()

    spark = build_session("skew-demo", a.nodes * 4)
    # Force a real shuffle join. Broadcasting would sidestep the whole lesson.
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    # AQE's skew handling would also hide it; turn it off so the mitigation measured
    # is the one written here, not the one Spark does behind your back.
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")
    await_cluster(spark, a.nodes)

    trips = spark.read.parquet(C.TRIPS).select("pickup_zone_id")
    camps = campaigns(spark, a.per_zone)

    counts = (trips.groupBy("pickup_zone_id").count()
              .orderBy(F.desc("count")).limit(3).collect())
    total = sum(r["count"] for r in trips.groupBy("pickup_zone_id").count().collect())
    hot = counts[0]
    share = hot["count"] / total
    # How much a hot key can possibly cost you: the job cannot finish before its
    # biggest single task does, so makespan >= max(hot_share, 1/cores) * total work,
    # while a balanced job costs 1/cores. Everything above 1.0 is what salting can
    # win back -- and it has to beat its own extra shuffle to be worth doing.
    predicted = max(share * a.nodes, 1.0)
    print(f"hottest key: zone {hot['pickup_zone_id']} holds {hot['count']:,} rows "
          f"({share * 100:.1f}% of all trips)")
    print(f"  {a.nodes} cores -> one task holds {share * 100:.0f}% of the work where "
          f"{100 / a.nodes:.0f}% would be balanced")
    print(f"  predicted ceiling for salting: {predicted:.2f}x")

    results = {}
    for name, fn in (("plain", lambda: plain_join(trips, camps)),
                     ("salted", lambda: salted_join(trips, camps, a.salts))):
        fn().count()  # warm-up
        t = time.perf_counter()
        rows = fn().count()
        results[name] = round(time.perf_counter() - t, 2)
        print(f"  {name:7s} {results[name]:6.2f}s  ({rows} output rows)")

    speedup = results["plain"] / results["salted"] if results["salted"] else 0
    print(f"salting the hot key: {speedup:.2f}x measured vs {predicted:.2f}x "
          f"predicted ceiling, on {a.nodes} nodes (salts={a.salts})")
    out = dict(nodes=a.nodes, salts=a.salts, per_zone=a.per_zone,
               predicted_ceiling=round(predicted, 3),
               hot_key=int(hot["pickup_zone_id"]),
               hot_share=round(share, 4),
               plain_s=results["plain"], salted_s=results["salted"],
               speedup=round(speedup, 3))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    spark.stop()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
