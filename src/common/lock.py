"""Advisory lock over the shared data directory.

Every tier reads one `data/trips` path, and `generate()` starts by deleting it. Two
things touching that path at once -- two benchmark runs, or a run and a manual
`make gen` -- silently destroy each other's data and then produce numbers that look
entirely plausible and mean nothing. That is exactly what happened on 2026-09-06: a
`make bench` and a 300M-row generation overlapped, and the resulting dataset was a
mix of 8 files from one scale and 111 from another.

Silent corruption is the worst failure mode a benchmark can have, so the second
writer now fails fast and says who holds the lock.
"""
import contextlib
import fcntl
import os

# Set by the benchmark runner when it shells `generate` into a container, so the
# child does not deadlock against the lock its own parent is already holding.
LOCK_ENV = "SCALING_DATA_LOCK_HELD"


@contextlib.contextmanager
def data_lock(root: str, what: str):
    if os.environ.get(LOCK_ENV):
        yield
        return

    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, ".lock")
    # O_CREAT|O_RDWR rather than "w": truncating first would destroy the holder's
    # identity before we can read it back for the error message.
    fh = os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR, 0o644), "r+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = fh.read().strip() or "another process"
        fh.close()
        raise SystemExit(
            f"\nrefusing to start {what}: the data directory is already in use by\n"
            f"  {holder}\n"
            f"Wait for it to finish, or stop it. Running both would corrupt "
            f"data/trips and silently invalidate every number produced.\n")

    fh.seek(0)
    fh.truncate()
    fh.write(f"{what} (pid {os.getpid()})\n")
    fh.flush()
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
