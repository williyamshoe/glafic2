#!/usr/bin/env python
"""Run a cluster notebook for every cluster in Caminha et al. (2021), one after the other.

    python run_caminha.py                       # cluster_cosmo_models.ipynb for rj2129, then the others
    python run_caminha.py --notebook cluster_cosmo_hmc.ipynb
    python run_caminha.py --clusters rj2129 mj1931
    python run_caminha.py --dry-run             # list what would run
    python run_caminha.py --smoke               # tiny chains (SMOKE_TEST = True), to test the pipeline
    python run_caminha.py --clusters as1063 --model shear+mpole32+mpole42
                                                # force a mass model -> results/as1063_shear_mpole32_mpole42/

The clusters are the rows marked in the 'Caminha' column of
example_data/venus_clusters_summary.csv (folder names from 'glafic_name'),
with rj2129 first.  Each cluster runs in a fresh kernel with GLAFIC_CLUSTER
set, exactly as the notebook does interactively; the executed notebook is
saved next to the cluster's other results, as
results/<cluster>/<notebook>.executed.ipynb (results_smoke/ with --smoke).

A cluster that fails (for example the results-folder guard, or a model that
fails the image check) is logged and skipped; the others still run.  Re-running
is cheap for finished clusters: every stage loads its cached result, and the
samplers resume from their checkpoints, so the script can be restarted after a
batch-job time limit.  Progress goes to results/run_caminha.log.

On NERSC, run it inside a batch job from this folder, in the environment that
has glafic_jax and JAX (its 'python3' kernel is used), e.g.

    srun python run_caminha.py
"""
import argparse
import csv
import os
import re
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_cluster as build  # noqa: E402

FIRST = "rj2129"


def caminha_clusters():
    """Folder names of the clusters marked in the summary table's 'Caminha' column, rj2129 first."""
    path = os.path.join(build.DATA, build.SUMMARY)
    rows = list(csv.DictReader(open(path)))
    names = [r["glafic_name"].strip() for r in rows if (r.get("Caminha") or "").strip()]
    if not names:
        raise SystemExit(f"no rows marked in the 'Caminha' column of {path}")
    return sorted(names, key=lambda n: (n != FIRST, names.index(n)))


def log(msg, logfile):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logfile, "a") as f:
        f.write(line + "\n")


def run_one(cluster, notebook, results, smoke, logfile, model=""):
    """Execute the notebook for one cluster; returns (ok, seconds)."""
    nb = nbformat.read(os.path.join(HERE, notebook), as_version=4)
    if smoke:
        n = 0
        for c in nb.cells:
            if c.cell_type == "code" and "SMOKE_TEST = False" in c.source:
                c.source = c.source.replace("SMOKE_TEST = False", "SMOKE_TEST = True"); n += 1
        if n != 1:
            raise RuntimeError(f"{notebook}: could not switch on SMOKE_TEST")
    outdir = os.path.join(HERE, results, build.results_name(cluster, model))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, notebook.replace(".ipynb", ".executed.ipynb"))
    os.environ["GLAFIC_CLUSTER"] = cluster              # the kernel inherits it
    os.environ["GLAFIC_MODEL"] = model                  # '' = automatic model choice
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": HERE}})
    t0 = time.time()
    try:
        client.execute()
        ok = True
    except CellExecutionError as err:
        ok = False
        plain = re.sub(r"\x1b\[[0-9;]*m", "", str(err))   # drop the traceback's terminal colours
        last = [ln for ln in plain.strip().splitlines() if ln.strip()]
        log(f"{cluster}: FAILED in a notebook cell: {last[-1] if last else err}", logfile)
    finally:
        nbformat.write(nb, out)                         # keep the outputs, also of a failed run
    log(f"{cluster}: {'done' if ok else 'stopped'} in {(time.time() - t0) / 60:.1f} min; "
        f"executed notebook: {os.path.relpath(out, HERE)}", logfile)
    return ok, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--notebook", default="cluster_cosmo_models.ipynb",
                    help="notebook in this folder to run (default: cluster_cosmo_models.ipynb)")
    ap.add_argument("--clusters", nargs="+", help="clusters to run instead of Caminha's (folder names)")
    ap.add_argument("--smoke", action="store_true", help="tiny chains: set SMOKE_TEST = True (results_smoke/)")
    ap.add_argument("--model", default="",
                    help="force this mass model for every cluster, e.g. shear+mpole32+mpole42 (components: shear, "
                         "bcg, mpole<m><n>, gnfw); results go to results/<cluster>_<model>/")
    ap.add_argument("--dry-run", action="store_true", help="only list the clusters")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(HERE, a.notebook)):
        raise SystemExit(f"no notebook {a.notebook} in {HERE} (run its make_notebook*.py first)")
    clusters = a.clusters or caminha_clusters()
    try:
        build.parse_model(a.model)
    except ValueError as err:
        raise SystemExit(str(err))
    unknown = [c for c in clusters if c not in build.available()]
    if unknown:
        raise SystemExit(f"unknown clusters {unknown}; available: {build.available()}")
    excluded = [c for c in clusters if build.is_excluded(c)]
    results = "results_smoke" if a.smoke else "results"
    logfile = os.path.join(HERE, results, "run_caminha.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    print(f"{a.notebook} for {len(clusters)} cluster(s): {', '.join(clusters)}"
          + (f" with mass model {a.model}" if a.model else "") + (" (SMOKE_TEST)" if a.smoke else ""))
    if excluded:
        print(f"note: {excluded} are marked 'exclude'; the notebook stops for them unless RUN_EXCLUDED = True")
    if a.dry_run:
        return

    log(f"start: {a.notebook} for {', '.join(clusters)}" + (f" with mass model {a.model}" if a.model else "")
        + (" (SMOKE_TEST)" if a.smoke else ""), logfile)
    status = {}
    for c in clusters:
        log(f"{c}: starting", logfile)
        try:
            status[c] = run_one(c, a.notebook, results, a.smoke, logfile, a.model)
        except Exception:                               # kernel died, etc.: log and go on
            log(f"{c}: ERROR outside the notebook:\n{traceback.format_exc()}", logfile)
            status[c] = (False, 0.0)
    log("summary: " + ", ".join(f"{c} {'ok' if ok else 'FAILED'} ({t / 60:.0f} min)"
                                for c, (ok, t) in status.items()), logfile)
    sys.exit(0 if all(ok for ok, _ in status.values()) else 1)


if __name__ == "__main__":
    main()
