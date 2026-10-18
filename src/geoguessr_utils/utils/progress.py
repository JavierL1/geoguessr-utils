"""
Generic progress reporting for long-running loops.

No project-specific knowledge — takes counts, a start time, and an
optional free-form suffix. Used by both the discovery scanner and the
extraction pipeline, which previously each had their own copy of this
arithmetic.
"""

import sys
import time


def print_eta_progress(
    *,
    current: int,
    total: int,
    start_time: float,
    suffix: str = "",
    stream=sys.stdout,
) -> None:
    """Print a single-line in-place progress bar with ETA.

    Args:
        current: Items processed so far.
        total: Total items to process.
        start_time: time.time() taken when processing began.
        suffix: Extra text appended to the line (e.g. "Batches: 3").
        stream: File-like object to write to. Defaults to stdout.

    Writes a carriage-return-prefixed line so repeated calls animate
    in place. Does nothing if not enough progress has been made yet
    for the numbers to be meaningful.
    """
    elapsed = time.time() - start_time
    if elapsed <= 0 or current <= 0 or total <= 0:
        return

    pct = min(100.0, (current / total) * 100)
    rate = current / elapsed
    remaining = total - current
    eta_sec = remaining / rate if rate > 0 else 0

    eta_mins, eta_secs = divmod(int(eta_sec), 60)
    elapsed_mins, elapsed_secs = divmod(int(elapsed), 60)

    line = (
        f"\rProgress: [{current}/{total}] ({pct:5.1f}%) | "
        f"Rate: {rate:5.2f}/s | "
        f"Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d} | "
        f"ETA: {eta_mins:02d}:{eta_secs:02d}"
    )
    if suffix:
        line += f" | {suffix}"

    stream.write(line + "  ")  # trailing spaces clear stale chars
    stream.flush()
