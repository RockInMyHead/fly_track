"""Run a script and print every file under data/ or output/ it opened for reading.

    python desktop/trace_reads.py scripts/p062_neuron_run.py --video X.AVI --tag t --duration 4

Used to find which shared files a fresh install needs (desktop/build_bundle.py SEED_FILES).
"""

from __future__ import annotations

import atexit
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEN: set[str] = set()


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
        return
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
    if any(c in mode for c in "wax+"):
        return
    try:
        p = Path(args[0] if not isinstance(args[0], bytes) else args[0].decode()).resolve()
        rel = p.relative_to(ROOT)
    except Exception:
        return
    if rel.parts and rel.parts[0] in ("data", "output") and p.is_file():
        SEEN.add(rel.as_posix())


@atexit.register
def report():
    with open(ROOT / "dist" / "reads.txt", "a", encoding="utf-8") as fh:
        for r in sorted(SEEN):
            fh.write(r + "\n")


sys.addaudithook(hook)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
