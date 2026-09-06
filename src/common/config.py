"""Single source of truth for paths, scales and the pipeline's business rules.

Every tier (pandas / chunked / spark) imports from here so that "same job, different
engine" is enforced by construction rather than by hoping three files stay in sync.
"""
import os

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")

TRIPS = f"{DATA_ROOT}/trips"
ZONES = f"{DATA_ROOT}/zones/zones.parquet"
OUT = f"{DATA_ROOT}/out"

# Dataset sizes. `files` drives the number of Parquet parts, which is the unit of
# parallelism: a cluster can never use more cores than there are input partitions.
SCALES = {
    "xs": dict(rows=200_000, files=8),
    "s": dict(rows=2_000_000, files=16),
    "m": dict(rows=10_000_000, files=32),
    "l": dict(rows=30_000_000, files=64),
    "xl": dict(rows=80_000_000, files=128),
}

N_ZONES = 265
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]

# --- business rules, shared by all three implementations ---------------------
MIN_DISTANCE_KM = 0.1
MAX_DISTANCE_KM = 200.0
MIN_FARE = 2.5
MIN_DURATION_S = 60
MAX_DURATION_S = 6 * 3600
TOP_N_ZONES = 5

# Aggregate columns produced by stage 5, in output order.
AGG_COLS = [
    "borough", "pickup_hour", "trips", "total_revenue", "avg_fare",
    "avg_tip_pct", "avg_speed_kmh", "max_distance_km", "n_vendors",
]
TOP_COLS = ["borough", "rank", "zone_id", "zone_revenue", "zone_trips"]


def scale_cfg(name: str) -> dict:
    if name not in SCALES:
        raise SystemExit(f"unknown scale {name!r}; pick one of {list(SCALES)}")
    return SCALES[name]
