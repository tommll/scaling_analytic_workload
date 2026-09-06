"""T2 — vertical scaling: same box, all the cores, bounded memory.

This is the honest intermediate step, and it is already a distributed system in
miniature: map each input partition to a *partial aggregate*, then reduce the partials
into the final answer. Peak memory is O(one partition), not O(dataset), so it survives
inputs far larger than RAM. What it cannot do is use a second machine — which is
exactly the wall that motivates T3.
"""
import glob
import json
import multiprocessing as mp
import os
import resource
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import config as C  # noqa: E402
from src.baseline import pandas_pipeline as P  # noqa: E402

# Columns that are summed during the reduce; the averages are reconstructed at the
# end from these sums, because you cannot average an average.
_SUMS = ["trips", "total_revenue", "sum_fare", "sum_tip_pct", "sum_speed"]


def _map_partition(path: str) -> tuple:
    """Map phase: one input file -> two small partial-aggregate frames."""
    df = pd.read_parquet(path)
    n_in = len(df)
    df = P.stage_derive(P.stage_clean(df))
    df = P.stage_join(df)

    agg = df.groupby(["borough", "pickup_hour"], as_index=False).agg(
        trips=("trip_id", "count"),
        total_revenue=("total_amount", "sum"),
        sum_fare=("fare_amount", "sum"),
        sum_tip_pct=("tip_pct", "sum"),
        sum_speed=("speed_kmh", "sum"),
        max_distance_km=("trip_distance_km", "max"),
        vendors=("vendor_id", lambda s: frozenset(s.unique())),
    )
    zones = df.groupby(["borough", "pickup_zone_id"], as_index=False).agg(
        zone_revenue=("total_amount", "sum"), zone_trips=("trip_id", "count"))
    return agg, zones, n_in, len(df)


def _reduce(aggs: list, zones: list) -> tuple:
    """Reduce phase: combine partials. Associative + commutative, so order-free."""
    a = pd.concat(aggs, ignore_index=True)
    vendors = (a.groupby(["borough", "pickup_hour"])["vendors"]
               .apply(lambda s: len(frozenset().union(*s))).rename("n_vendors"))
    g = a.groupby(["borough", "pickup_hour"]).agg(
        **{c: (c, "sum") for c in _SUMS}, max_distance_km=("max_distance_km", "max"))
    g = g.join(vendors).reset_index()
    g["avg_fare"] = g["sum_fare"] / g["trips"]
    g["avg_tip_pct"] = g["sum_tip_pct"] / g["trips"]
    g["avg_speed_kmh"] = g["sum_speed"] / g["trips"]
    agg = g.sort_values(["borough", "pickup_hour"], ignore_index=True)[C.AGG_COLS]

    z = (pd.concat(zones, ignore_index=True)
         .groupby(["borough", "pickup_zone_id"], as_index=False)
         .agg(zone_revenue=("zone_revenue", "sum"), zone_trips=("zone_trips", "sum"))
         .rename(columns={"pickup_zone_id": "zone_id"}))
    z = z.sort_values(["borough", "zone_revenue", "zone_id"],
                      ascending=[True, False, True], ignore_index=True)
    z["rank"] = z.groupby("borough").cumcount() + 1
    top = z[z["rank"] <= C.TOP_N_ZONES].reset_index(drop=True)[C.TOP_COLS]
    return agg, top


def run(out_dir: str, workers: int = 0) -> dict:
    workers = workers or mp.cpu_count()
    files = sorted(glob.glob(f"{C.TRIPS}/*.parquet"))
    t_start = time.perf_counter()

    t = time.perf_counter()
    with mp.Pool(workers) as pool:
        parts = pool.map(_map_partition, files)
    t_map = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    agg, top = _reduce([p[0] for p in parts], [p[1] for p in parts])
    t_reduce = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(f"{out_dir}/agg.csv", index=False)
    top.to_csv(f"{out_dir}/top_zones.csv", index=False)
    t_write = round(time.perf_counter() - t, 3)

    rss = max(resource.getrusage(r).ru_maxrss
              for r in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return dict(
        tier="chunked", workers=workers, wall_s=round(time.perf_counter() - t_start, 3),
        rows_in=sum(p[2] for p in parts), rows_kept=sum(p[3] for p in parts),
        stages=dict(map=t_map, reduce=t_reduce, write=t_write),
        peak_rss_mb=round(rss / 1024, 1),
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{C.OUT}/chunked"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(json.dumps(run(out, w)))
