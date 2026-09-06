"""Generate the synthetic trip dataset.

Written as many Parquet part-files rather than one big file, because the number of
files *is* the parallelism budget for every downstream engine. Generation itself is
parallelised across processes so that making 80M rows doesn't dominate the workday.
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import config as C  # noqa: E402

# A single "hot" zone that receives a disproportionate share of pickups. Real key
# distributions are never uniform, and the skew experiment needs a victim.
HOT_ZONE = 132
HOT_SHARE = 0.30


def _make_part(args):
    part, rows, out_dir, seed, skew, hot_share = args
    rng = np.random.default_rng(seed)

    pickup = rng.integers(1, C.N_ZONES + 1, rows, dtype=np.int32)
    if skew:
        hot = rng.random(rows) < hot_share
        pickup[hot] = HOT_ZONE

    # Timestamps spread over 30 days with a realistic double-humped daily profile.
    day = rng.integers(0, 30, rows)
    hour = np.clip(rng.normal(rng.choice([8.5, 18.0], rows), 3.0), 0, 23.999)
    start = (day * 86400 + hour * 3600 + rng.random(rows) * 60).astype(np.int64)
    base = np.int64(1704067200)  # 2024-01-01T00:00:00Z

    distance = np.round(rng.lognormal(0.7, 0.85, rows), 2)
    duration = np.clip(distance / rng.uniform(8, 45, rows) * 3600 + rng.normal(120, 60, rows),
                       5, 8 * 3600).astype(np.int64)
    fare = np.round(3.0 + distance * 2.6 + duration / 60.0 * 0.45, 2)
    tip = np.round(fare * rng.choice([0.0, 0.1, 0.15, 0.2, 0.25], rows,
                                     p=[0.25, 0.15, 0.25, 0.25, 0.10]), 2)

    df = pd.DataFrame({
        "trip_id": np.arange(part * rows, (part + 1) * rows, dtype=np.int64),
        "vendor_id": rng.integers(1, 4, rows, dtype=np.int16),
        "pickup_zone_id": pickup,
        "dropoff_zone_id": rng.integers(1, C.N_ZONES + 1, rows, dtype=np.int32),
        "pickup_ts": base + start,
        "dropoff_ts": base + start + duration,
        "passenger_count": rng.integers(1, 7, rows, dtype=np.int16),
        "trip_distance_km": distance.astype(np.float64),
        "fare_amount": fare.astype(np.float64),
        "tip_amount": tip.astype(np.float64),
        "payment_type": rng.choice(["card", "cash", "wallet"], rows,
                                   p=[0.7, 0.2, 0.1]).astype(object),
    })

    # ~4% dirty records, so the cleaning stage has something to actually remove.
    bad = rng.random(rows) < 0.04
    df.loc[bad, "trip_distance_km"] = 0.0
    df.loc[bad, "fare_amount"] = -1.0

    path = os.path.join(out_dir, f"part-{part:05d}.parquet")
    df.to_parquet(path, index=False, compression="snappy")
    return os.path.getsize(path)


def generate(scale: str, skew: bool = False, workers: int = 0,
             hot_share: float = HOT_SHARE) -> None:
    cfg = C.scale_cfg(scale)
    rows_per_part = cfg["rows"] // cfg["files"]
    workers = workers or min(mp.cpu_count(), cfg["files"])

    for d in (C.TRIPS, os.path.dirname(C.ZONES)):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(0)
    zones = pd.DataFrame({
        "zone_id": np.arange(1, C.N_ZONES + 1, dtype=np.int32),
        "borough": rng.choice(C.BOROUGHS, C.N_ZONES, p=[.35, .25, .2, .13, .05, .02]),
        "zone_name": [f"Zone {i:03d}" for i in range(1, C.N_ZONES + 1)],
    })
    zones.to_parquet(C.ZONES, index=False)

    t0 = time.time()
    tasks = [(p, rows_per_part, C.TRIPS, 1000 + p, skew, hot_share)
             for p in range(cfg["files"])]
    with mp.Pool(workers) as pool:
        sizes = pool.map(_make_part, tasks)

    print(f"scale={scale} skew={'%.2f' % hot_share if skew else False}: {cfg['files']} files, "
          f"{rows_per_part * cfg['files']:,} rows, "
          f"{sum(sizes) / 1e6:.0f} MB on disk, in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="m")
    ap.add_argument("--skew", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--hot-share", type=float, default=HOT_SHARE,
                    help="fraction of trips forced onto one pickup zone")
    a = ap.parse_args()
    generate(a.scale, a.skew, a.workers, a.hot_share)
