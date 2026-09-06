"""Fault tolerance: kill a node mid-job and check the answer is still correct.

The single-node script has no story here at all -- if the process dies, the run is
gone and you start over. A distributed engine treats compute as disposable: the
partitions that were in flight on the dead executor are simply recomputed elsewhere,
because the lineage that produced them is known and the input is immutable.

The bar is not "the job survived" but "the answer is the same". Note the difference:
byte-identical is the WRONG bar and this demo originally used it and failed. When
lost partitions are recomputed they are combined in a different order, and floating
point addition is not associative, so sums differ in the last ulp. Every integer
column -- counts, ranks, ids -- is exactly identical; the float columns agree to
rtol 1e-9. That is precisely the guarantee a lineage-based engine actually makes,
and confusing it with bit-reproducibility will send you hunting a bug that is not
there.
"""
import filecmp
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = int(os.environ.get("FAULT_NODES", "4"))
VICTIM = os.environ.get("FAULT_VICTIM", f"spark-worker-{NODES}")
KILL_AT = float(os.environ.get("FAULT_KILL_AFTER", "35"))


def spark_run(out_dir: str):
    return subprocess.Popen(
        ["docker", "exec", "-e", f"SPARK_CORES_MAX={NODES}",
         "-e", f"SPARK_EXPECT_EXECUTORS={NODES}", "spark-client", "python3", "-m",
         "src.distributed.spark_pipeline", "--out", out_dir,
         "--shuffle-partitions", str(NODES * 4)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def main() -> int:
    print(f"1. reference run on {NODES} healthy nodes ...", flush=True)
    if spark_run("/data/out/fault_ref").wait() != 0:
        print("   reference run failed; is the cluster up?")
        return 1

    print(f"2. same job, killing {VICTIM} after {KILL_AT:.0f}s ...", flush=True)
    t0 = time.time()
    proc = spark_run("/data/out/fault_kill")
    time.sleep(KILL_AT)
    if proc.poll() is not None:
        print("   job finished before the kill; raise FAULT_KILL_AFTER")
        return 1
    subprocess.run(["docker", "kill", VICTIM], capture_output=True)
    print(f"   killed {VICTIM} at t+{time.time() - t0:.0f}s "
          f"({100 // NODES}% of the cluster is now gone)")
    rc = proc.wait()
    print(f"   job exited rc={rc} after {time.time() - t0:.0f}s total")

    subprocess.run(["docker", "start", VICTIM], capture_output=True)
    if rc != 0:
        print("VERDICT: job did NOT survive the node loss")
        return 1

    identical = all(filecmp.cmp(f"{ROOT}/data/out/fault_ref/{f}",
                                f"{ROOT}/data/out/fault_kill/{f}", shallow=False)
                    for f in ("agg.csv", "top_zones.csv"))
    parity = subprocess.run(
        ["docker", "exec", "baseline", "python3", "-m", "src.bench.parity",
         "/data/out", "fault_ref", "fault_kill"], capture_output=True, text=True)
    print(f"3. byte-identical to the reference run? "
          f"{'yes' if identical else 'no - float sums reassociated on recompute'}")
    print(f"4. same answer within tolerance? {parity.stdout.strip().splitlines()[0]}")
    ok = parity.returncode == 0
    print("VERDICT: " + ("lost 25% of the cluster mid-flight; the partitions in "
                         "progress were recomputed elsewhere and the answer is "
                         "unchanged (ints exact, floats to rtol 1e-9)." if ok else
                         "job completed but the answer changed materially!"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
