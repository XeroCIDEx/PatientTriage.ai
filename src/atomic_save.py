"""
atomic_save.py
===============
Crash-safe joblib saving.

Problem this fixes: joblib.dump(obj, path) writes directly to the target
path. If the process is interrupted mid-write (browser tab closed,
terminal killed, machine sleeps, disk hiccup) while that write is still
in progress, `path` is left containing a truncated/corrupted file. The
NEXT time the app starts, `models_exist()` sees the file is present and
skips training — but loading it then fails with an unpickling error
like "input stream corrupted".

Fix: write to a temporary file in the same directory, then use
`os.replace()` to atomically rename it into place. os.replace is a
single filesystem operation — it either completes fully (temp file
becomes the real file) or doesn't happen at all if interrupted. There is
no state in between where `path` is half-written, so a crash can never
corrupt the file that's actually sitting at `path`.
"""

from __future__ import annotations
import os
import tempfile
import joblib


def atomic_joblib_dump(obj, path: str):
    """Save `obj` to `path` such that `path` never contains a partially
    written file, even if the process is killed mid-save."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".joblib")
    try:
        os.close(fd)
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows
    except BaseException:
        # clean up the temp file on any failure (including Ctrl-C) so it
        # doesn't linger; the real `path` is untouched either way
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
