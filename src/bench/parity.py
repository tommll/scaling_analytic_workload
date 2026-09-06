"""Correctness gate: a fast wrong answer scores zero.

The three engines sum floats in different orders, so bit-identical output is neither
achievable nor the right bar. What must hold is that every integer column matches
exactly and every float column matches to relative 1e-9 -- i.e. the only difference
is floating-point associativity, not different arithmetic.
"""
import sys

import numpy as np
import pandas as pd

RTOL = 1e-9
INT_COLS = {"pickup_hour", "trips", "n_vendors", "rank", "zone_id", "zone_trips"}


def compare(ref_dir: str, other_dir: str, name: str) -> list:
    a = pd.read_csv(f"{ref_dir}/{name}.csv")
    b = pd.read_csv(f"{other_dir}/{name}.csv")
    problems = []
    if a.shape != b.shape:
        return [f"{name}: shape {a.shape} != {b.shape}"]
    if list(a.columns) != list(b.columns):
        return [f"{name}: columns {list(a.columns)} != {list(b.columns)}"]

    for col in a.columns:
        x, y = a[col], b[col]
        if col in INT_COLS or x.dtype == object:
            bad = int((x != y).sum())
            if bad:
                problems.append(f"{name}.{col}: {bad} exact mismatches")
        else:
            close = np.isclose(x, y, rtol=RTOL, atol=1e-9)
            if not close.all():
                worst = np.nanmax(np.abs((x - y) / np.where(y == 0, 1, y)))
                problems.append(
                    f"{name}.{col}: {(~close).sum()} rows exceed rtol={RTOL} "
                    f"(worst relative error {worst:.2e})")
    return problems


def main(root: str, tiers: list) -> int:
    ref, others = tiers[0], tiers[1:]
    failures = []
    for other in others:
        for name in ("agg", "top_zones"):
            failures += [f"[{ref} vs {other}] {p}"
                         for p in compare(f"{root}/{ref}", f"{root}/{other}", name)]

    if failures:
        print("PARITY FAILED")
        for f in failures:
            print("  " + f)
        return 1
    print(f"PARITY OK: {', '.join(tiers)} agree on agg.csv and top_zones.csv "
          f"(ints exact, floats within rtol={RTOL})")
    return 0


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/data/out"
    tiers = sys.argv[2:] or ["pandas", "chunked", "spark"]
    sys.exit(main(root, tiers))
