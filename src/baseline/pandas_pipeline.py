"""T1 — the original single-process script. One core, everything in RAM.

This is the "before" picture: correct, readable, and completely unable to grow. It
reads every part-file into one DataFrame, which means peak memory is O(dataset) and
the wall clock is O(dataset) on exactly one core no matter how many the box has.
"""
import glob
import json
import os
import resource
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import config as C  # noqa: E402


def stage_read() -> pd.DataFrame:
    files = sorted(glob.glob(f"{C.TRIPS}/*.parquet"))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def stage_clean(df: pd.DataFrame) -> pd.DataFrame:
    dur = df["dropoff_ts"] - df["pickup_ts"]
    keep = (
        df["trip_distance_km"].between(C.MIN_DISTANCE_KM, C.MAX_DISTANCE_KM)
        & (df["fare_amount"] >= C.MIN_FARE)
        & dur.between(C.MIN_DURATION_S, C.MAX_DURATION_S)
    )
    return df.loc[keep].copy()


def stage_derive(df: pd.DataFrame) -> pd.DataFrame:
    dur_s = df["dropoff_ts"] - df["pickup_ts"]
    df["duration_min"] = dur_s / 60.0
    df["speed_kmh"] = df["trip_distance_km"] / (dur_s / 3600.0)
    df["total_amount"] = df["fare_amount"] + df["tip_amount"]
    df["tip_pct"] = df["tip_amount"] / df["fare_amount"] * 100.0
    df["pickup_hour"] = (df["pickup_ts"] % 86400) // 3600
    return df


def stage_join(df: pd.DataFrame) -> pd.DataFrame:
    zones = pd.read_parquet(C.ZONES)[["zone_id", "borough"]]
    return df.merge(zones, left_on="pickup_zone_id", right_on="zone_id", how="inner")


def stage_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["borough", "pickup_hour"], as_index=False).agg(
        trips=("trip_id", "count"),
        total_revenue=("total_amount", "sum"),
        avg_fare=("fare_amount", "mean"),
        avg_tip_pct=("tip_pct", "mean"),
        avg_speed_kmh=("speed_kmh", "mean"),
        max_distance_km=("trip_distance_km", "max"),
        n_vendors=("vendor_id", "nunique"),
    )
    return g.sort_values(["borough", "pickup_hour"], ignore_index=True)[C.AGG_COLS]


def stage_top_zones(df: pd.DataFrame) -> pd.DataFrame:
    z = df.groupby(["borough", "pickup_zone_id"], as_index=False).agg(
        zone_revenue=("total_amount", "sum"), zone_trips=("trip_id", "count"))
    z = z.rename(columns={"pickup_zone_id": "zone_id"})
    # zone_id is the tiebreaker so the ranking is deterministic across engines.
    z = z.sort_values(["borough", "zone_revenue", "zone_id"],
                      ascending=[True, False, True], ignore_index=True)
    z["rank"] = z.groupby("borough").cumcount() + 1
    return z[z["rank"] <= C.TOP_N_ZONES].reset_index(drop=True)[C.TOP_COLS]


def run(out_dir: str) -> dict:
    timings, t_start = {}, time.perf_counter()

    def timed(name, fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        timings[name] = round(time.perf_counter() - t, 3)
        return r

    df = timed("read", stage_read)
    n_in = len(df)
    df = timed("clean", stage_clean, df)
    df = timed("derive", stage_derive, df)
    df = timed("join", stage_join, df)
    agg = timed("aggregate", stage_aggregate, df)
    top = timed("top_zones", stage_top_zones, df)

    t = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(f"{out_dir}/agg.csv", index=False)
    top.to_csv(f"{out_dir}/top_zones.csv", index=False)
    timings["write"] = round(time.perf_counter() - t, 3)

    return dict(
        tier="pandas", wall_s=round(time.perf_counter() - t_start, 3),
        rows_in=n_in, rows_kept=len(df), stages=timings,
        peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{C.OUT}/pandas"
    print(json.dumps(run(out)))
