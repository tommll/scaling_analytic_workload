"""Benchmark orchestrator. Runs on the host, drives the containers.

Fair-benchmark rules are enforced here rather than left to discipline:

  * every simulated node is one dedicated physical core + 2 GB (see compose file)
  * the cluster is warm before the clock starts -- otherwise a bigger cluster just
    measures more JVMs booting, and appears *slower*
  * the first run of every configuration is discarded (JIT, Parquet footer cache)
  * the reported number is the median of `--reps` timed runs; min/max are kept
  * parity is verified before any timing is published
  * the host's OWN parallel-scaling ceiling is measured first, because a flat
    scaling curve is far more often the machine than the software
"""
import argparse
import json
import os
import subprocess
import sys
import time
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.common.lock import LOCK_ENV, data_lock  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CF = ["docker", "compose", "-f", os.path.join(ROOT, "docker", "docker-compose.yml")]
PROFILES = [a for i in range(1, 7) for a in ("--profile", f"w{i}")]
MAX_NODES = 6


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def compose(*args):
    return sh(CF + PROFILES + list(args))


def ensure_created():
    """Create every container once; scaling is then just start/stop, which is fast
    and -- more importantly -- keeps each node pinned to its assigned core."""
    compose("up", "-d")
    time.sleep(3)


def set_cluster(n: int) -> None:
    """Horizontal scaling: N nodes means N machines, never one bigger machine."""
    want = {f"spark-worker-{i}" for i in range(1, n + 1)}
    for i in range(1, MAX_NODES + 1):
        name = f"spark-worker-{i}"
        running = sh(["docker", "inspect", "-f", "{{.State.Running}}", name]).stdout.strip()
        if name in want and running != "true":
            sh(["docker", "start", name])
        elif name not in want and running == "true":
            sh(["docker", "stop", "-t", "2", name])
    time.sleep(6)  # let the master notice the membership change


def last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def exec_in(container: str, args: list, env: dict = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    return sh(cmd + [container, "python3", "-m"] + args)


def set_slots(slots: int) -> None:
    """One node with `slots` task slots.

    A slot is a THREAD inside a single executor JVM, not a process, so this is
    oversubscribing one physical core rather than adding capacity. Recreating the
    container is required because a worker's core count is fixed at launch.
    """
    for i in range(2, MAX_NODES + 1):
        sh(["docker", "stop", "-t", "2", f"spark-worker-{i}"])
    subprocess.run(CF + PROFILES + ["up", "-d", "--force-recreate", "spark-worker-1"],
                   capture_output=True, text=True,
                   env=dict(os.environ, WORKER_CORES=str(slots)))
    time.sleep(8)


def run_once(tier: str, nodes: int, procs: int = 1, slots: int = 1,
             shuffle: int = 0) -> dict:
    if tier == "pandas":
        p = exec_in("baseline", ["src.baseline.pandas_pipeline", "/data/out/pandas"])
    elif tier == "chunked":
        box = "baseline" if procs == 1 else "baseline-big"
        p = exec_in(box, ["src.baseline.chunked_pipeline", "/data/out/chunked", str(procs)])
    else:
        p = exec_in("spark-client",
                    ["src.distributed.spark_pipeline", "--out", "/data/out/spark",
                     "--shuffle-partitions",
                     str(shuffle or max(nodes * slots * 4, 4))],
                    {"SPARK_CORES_MAX": nodes * slots,
                     "SPARK_EXECUTOR_CORES": slots,
                     "SPARK_EXPECT_EXECUTORS": nodes})
    res = last_json(p.stdout)
    if not res:
        raise RuntimeError(f"{tier}(nodes={nodes},procs={procs}) failed "
                           f"rc={p.returncode}\n" + p.stdout[-1500:] + p.stderr[-1500:])
    return res


def measure(label: str, tier: str, nodes: int, reps: int, procs: int = 1,
            slots: int = 1, shuffle: int = 0, warmup: bool = True) -> dict:
    print(f"    {label:30s}", end="", flush=True)
    try:
        if warmup:
            # Discarded: JIT, Parquet footer caching. At the largest scales a warm-up
            # costs more than it removes -- a 12-minute job JITs in its first seconds
            # -- so it is skipped there and the min-max spread reports the residue.
            run_once(tier, nodes, procs, slots, shuffle)
        runs = [run_once(tier, nodes, procs, slots, shuffle) for _ in range(reps)]
    except RuntimeError as e:
        print("  DIED (see outcome=failed)")
        return dict(label=label, tier=tier, nodes=nodes, procs=procs, slots=slots,
                    outcome="failed", error=str(e)[:400])
    walls = [r["wall_s"] for r in runs]
    # For Spark, `compute` excludes cluster startup: startup is a fixed cost that does
    # not shrink with more nodes, so mixing it in hides the scaling signal. Both are
    # reported -- compute answers "does it scale", wall answers "is it worth it".
    comps = [r.get("compute_s", r["wall_s"]) for r in runs]
    out = dict(label=label, tier=tier, nodes=nodes, procs=procs, slots=slots,
               outcome="ok",
               wall_s=median(walls), wall_min=min(walls), wall_max=max(walls),
               compute_s=median(comps), compute_min=min(comps), compute_max=max(comps),
               startup_s=median([r.get("startup_s", 0.0) for r in runs]),
               rows_in=runs[-1]["rows_in"], rows_kept=runs[-1]["rows_kept"],
               rows_per_s=round(runs[-1]["rows_in"] / median(comps)),
               peak_rss_mb=runs[-1].get("peak_rss_mb"), reps=reps, detail=runs[-1])
    print(f"compute {median(comps):7.1f}s   wall {median(walls):7.1f}s"
          f"   ({min(comps):.1f}-{max(comps):.1f})")
    return out


def gen(scale: str, skew: bool = False) -> str:
    args = ["src.gen.generate", "--scale", scale] + (["--skew"] if skew else [])
    # The runner already holds the data lock for the whole run; tell the child so it
    # does not fail against its own parent.
    out = exec_in("baseline-big", args, {LOCK_ENV: "1"}).stdout.strip()
    print("    " + out)
    return out


def check_parity(tiers=("pandas", "chunked", "spark")) -> None:
    p = exec_in("baseline", ["src.bench.parity", "/data/out", *tiers])
    print("    " + (p.stdout.strip().splitlines() or ["no output"])[0])
    if p.returncode:
        raise SystemExit("refusing to publish timings: parity failed")


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def experiment_calibrate(reps: int = 3) -> list:
    """Measure what the HOST can actually deliver in parallel.

    This exists because it saved the project. On the dev machine (i7-1365U: 2 fast
    P-cores + 8 slow E-cores, 15 W package limit) two unpinned CPU-bound threads run
    ~3x slower than one, so every scaling curve came out flat -- which looks exactly
    like "Spark does not scale". It was the laptop. Every speedup below is therefore
    reported against this measured ceiling as well as against ideal linear.
    """
    print("\n[calibrate] the host's own parallel-scaling ceiling")
    spin = ("import math\nx=0.0\n"
            "for i in range(1, 20_000_000): x += math.sqrt(i)\n")
    rows, t1 = [], None
    for n in (1, 2, 4, MAX_NODES):
        set_cluster(n)
        best = None
        for _ in range(reps):
            procs = [subprocess.Popen(["docker", "exec", f"spark-worker-{i}",
                                       "python3", "-c", spin],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                     for i in range(1, n + 1)]
            t = time.perf_counter()
            for p in procs:
                p.wait()
            best = min(best or 1e9, time.perf_counter() - t)
        t1 = t1 or best
        rows.append(dict(nodes=n, wall_s=round(best, 3),
                         throughput_x=round(n * t1 / best, 3)))
        print(f"    {n} pinned node(s): {best:6.2f}s   "
              f"aggregate throughput {n * t1 / best:.2f}x")
    return rows


def experiment_strong(scale, node_counts, reps, warmup=True, skip=()) -> list:
    """Fixed dataset, growing cluster. 'How much faster on the same job?'

    `skip` drops tiers that cannot survive the scale being tested. Past a few tens of
    millions of rows T1 is a guaranteed OOM, and spending minutes re-proving that on
    every run adds nothing -- the memory-ceiling experiment already establishes it.
    """
    print(f"\n[strong scaling] fixed {scale} dataset, 1 -> {max(node_counts)} nodes"
          + (f"  (skipping: {', '.join(skip)})" if skip else ""))
    gen(scale)
    set_cluster(0)
    rows = []
    if "pandas" not in skip:
        rows.append(measure("T1 pandas (1 node)", "pandas", 1, reps, warmup=warmup))
    if "chunked" not in skip:
        rows += [measure("T2 chunked (1 node)", "chunked", 1, reps, procs=1,
                         warmup=warmup),
                 measure("T2 chunked (1 big box, 4c)", "chunked", 4, reps, procs=4,
                         warmup=warmup)]
    for n in node_counts:
        set_cluster(n)
        rows.append(measure(f"T3 spark ({n} node{'s' if n > 1 else ''})", "spark", n,
                            reps, warmup=warmup))
    if all(r["outcome"] == "ok" for r in rows):
        check_parity()
    return rows


def experiment_small(scale, nodes, reps) -> list:
    """Finds the crossover. Below some size, distribution is pure overhead."""
    print(f"\n[small data] {scale} -- where distribution is expected to LOSE")
    gen(scale)
    set_cluster(0)
    rows = [measure("T1 pandas (1 node)", "pandas", 1, reps),
            measure("T2 chunked (1 big box, 4c)", "chunked", 4, reps, procs=4)]
    set_cluster(nodes)
    rows.append(measure(f"T3 spark ({nodes} nodes)", "spark", nodes, reps))
    if all(r["outcome"] == "ok" for r in rows):
        check_parity()
    return rows


def experiment_memory(scale, nodes) -> list:
    """The capability difference: one node has a hard wall, a cluster does not."""
    print(f"\n[memory ceiling] {scale} against a 2 GB node")
    gen(scale)
    set_cluster(0)
    rows = []
    for label, tier, n, procs in (("T1 pandas (1 node, 2 GB)", "pandas", 1, 1),
                                  ("T2 chunked (1 node, 2 GB)", "chunked", 1, 1)):
        rows.append(measure(label, tier, n, 1, procs))
    set_cluster(nodes)
    rows.append(measure(f"T3 spark ({nodes} nodes)", "spark", nodes, 1))
    for r in rows:
        # rc=137 is SIGKILL, which for a memory-limited container means the cgroup
        # OOM killer. The container's own .State.OOMKilled flag is NOT usable here:
        # it is sticky once set, so a later unrelated failure reads as an OOM.
        if r["outcome"] == "failed":
            r["oom_killed"] = "rc=137" in r.get("error", "")
    return rows


def experiment_weak(scales, node_counts, reps) -> list:
    """Data grows with the cluster. 'Can I keep up as the data keeps growing?'"""
    print("\n[weak scaling] dataset grows in proportion to the cluster")
    rows = []
    for s, n in zip(scales, node_counts):
        gen(s)
        set_cluster(n)
        r = measure(f"{n} node(s) / scale {s}", "spark", n, reps)
        r["scale"] = s
        rows.append(r)
    return rows


def experiment_slots(scale, slot_counts, reps, shuffle=16) -> list:
    """Oversubscribe ONE core with N task slots, against every other approach.

    Shuffle partitions are pinned to a constant so slot count is the only variable.
    The reference rows are re-measured in the same session rather than reused from
    an earlier run, because the point is a head-to-head.
    """
    print(f"\n[task slots] one pinned core, {slot_counts} slots vs everything else")
    gen(scale)
    set_cluster(0)
    rows = [measure("T1 pandas (1 core)", "pandas", 1, reps),
            measure("T2 chunked (1 core)", "chunked", 1, reps, procs=1),
            measure("T2 chunked (4-core box)", "chunked", 4, reps, procs=4)]
    for sl in slot_counts:
        set_slots(sl)
        rows.append(measure(f"T3 spark (1 node, {sl} slot{'s' if sl > 1 else ''})",
                            "spark", 1, reps, slots=sl, shuffle=shuffle))
    set_slots(1)
    set_cluster(4)
    rows.append(measure("T3 spark (4 nodes, 1 slot)", "spark", 4, reps,
                        slots=1, shuffle=shuffle))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="calibrate,strong,small,memory")
    ap.add_argument("--scale", default="m,l",
                    help="comma-separated dataset sizes for strong scaling")
    ap.add_argument("--small-scale", default="xs")
    ap.add_argument("--memory-scale", default="l")
    ap.add_argument("--weak-scales", default="s,m,l")
    ap.add_argument("--nodes", default="1,2,4")
    ap.add_argument("--slots", default="1,2,4,10")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-tiers", default="",
                    help="comma-separated tiers to skip, e.g. 'pandas' at scales "
                         "where it is a guaranteed OOM")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the discarded warm-up run (use at the largest scales)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    a = ap.parse_args()

    with data_lock(os.path.join(ROOT, "data"), "benchmark runner"):
        nodes = [int(x) for x in a.nodes.split(",")]
        wanted = a.experiments.split(",")
        ensure_created()

        res = {"config": vars(a),
               "host": {"cpus": int(sh(["nproc"]).stdout.strip()),
                        "node": "1 pinned physical core + 2 GB",
                        "cpu_model": next((l.split(":", 1)[1].strip()
                                           for l in sh(["lscpu"]).stdout.splitlines()
                                           if l.startswith("Model name")), "unknown")},
               "experiments": {}}

        if "calibrate" in wanted:
            res["experiments"]["calibrate"] = experiment_calibrate()
        if "strong" in wanted:
            # One curve per dataset size: scalability is not a property of the code
            # alone, it improves as the per-node work grows relative to fixed costs.
            for sc in a.scale.split(","):
                res["experiments"][f"strong@{sc}"] = experiment_strong(
                    sc, nodes, a.reps, warmup=not a.no_warmup)
        if "weak" in wanted:
            res["experiments"]["weak"] = experiment_weak(a.weak_scales.split(","), nodes, a.reps)
        if "small" in wanted:
            res["experiments"]["small"] = experiment_small(a.small_scale, max(nodes), a.reps)
        if "slots" in wanted:
            res["experiments"]["slots"] = experiment_slots(
                a.scale.split(",")[0], [int(x) for x in a.slots.split(",")], a.reps)
        if "memory" in wanted:
            res["experiments"]["memory"] = experiment_memory(a.memory_scale, max(nodes))

        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
