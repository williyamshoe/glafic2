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
batch-job time limit.

Progress goes to results/run_caminha.log, in detail: the node, batch job and
GPUs; for each cluster, the start of every notebook cell (with its section
heading), every line the notebook prints as it appears (model search, MAP runs,
NUTS checkpoints and R-hat checks, ...), errors, and each cell's run time.
Each line is tagged with the cluster; the same lines also go to the cluster's
own results/<cluster>/run.log.

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
    """Timestamped message to stdout and to logfile (a path or a list of paths)."""
    stamp = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
    text = "\n".join(stamp + ln for ln in str(msg).splitlines() or [""])
    print(text, flush=True)
    for path in [logfile] if isinstance(logfile, str) else logfile:
        with open(path, "a") as f:
            f.write(text + "\n")


def log_environment(logfile):
    """Where the run happens: host, batch job, Python, visible GPUs."""
    import platform
    import shutil
    import subprocess
    lines = [f"host {platform.node()}; python {sys.version.split()[0]} ({sys.executable})"]
    job = {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURM_GPUS_ON_NODE",
                                      "CUDA_VISIBLE_DEVICES") if k in os.environ}
    if job:
        lines.append("job: " + ", ".join(f"{k}={v}" for k, v in job.items()))
    if shutil.which("nvidia-smi"):
        try:
            gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
            lines += [f"gpu: {g}" for g in gpus.splitlines()] or ["gpu: none listed by nvidia-smi"]
        except Exception as err:
            lines.append(f"gpu: nvidia-smi failed ({err})")
    else:
        lines.append("gpu: no nvidia-smi on this node")
    log("\n".join(lines), logfile)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class LoggingClient(NotebookClient):
    """NotebookClient that logs the notebook as it runs: the start of every cell (with its section heading),
    every line the cell prints (stdout and stderr, as it appears), errors, and each cell's run time.
    Progress bars are logged once, as their last state."""

    def __init__(self, nb, cluster, logfile, **kw):
        super().__init__(nb, on_cell_execute=self._cell_start, on_cell_executed=self._cell_done, **kw)
        self.cluster, self.logfile = cluster, logfile
        self._buf, self._t0 = {}, {}
        code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
        self._ncode, self._code_no, self._section = len(code), {i: k + 1 for k, i in enumerate(code)}, {}
        heading = ""
        for i, c in enumerate(nb.cells):
            if c.cell_type == "markdown":
                heads = [ln.lstrip("#").strip() for ln in c.source.splitlines() if ln.startswith("#")]
                heading = heads[0] if heads else heading
            self._section[i] = heading

    def _log(self, msg):
        log("\n".join(f"{self.cluster} | {ln}" for ln in str(msg).splitlines()), self.logfile)

    def _cell_start(self, cell, cell_index, **kw):
        self._t0[cell_index] = time.time()
        first = next((ln.strip() for ln in cell.source.splitlines() if ln.strip()), "")
        self._log(f"--- cell {self._code_no.get(cell_index, '?')}/{self._ncode}"
                  + (f" [{self._section.get(cell_index)}]" if self._section.get(cell_index) else "")
                  + f": {first[:100]}")

    def _cell_done(self, cell, cell_index, **kw):
        self._flush()
        dt = time.time() - self._t0.pop(cell_index, time.time())
        self._log(f"--- cell {self._code_no.get(cell_index, '?')}/{self._ncode} finished in "
                  + (f"{dt / 60:.1f} min" if dt >= 120 else f"{dt:.1f} s"))

    def _stream(self, name, text):
        buf = self._buf.get(name, "") + text
        *lines, rest = buf.split("\n")
        for ln in lines:
            ln = _ANSI.sub("", ln.rsplit("\r", 1)[-1]).rstrip()   # a progress bar: its last state only
            if ln:
                self._log(ln if name == "stdout" else f"[{name}] {ln}")
        self._buf[name] = rest[rest.rfind("\r"):] if "\r" in rest else rest

    def _flush(self):
        for name in list(self._buf):
            if self._buf[name].strip("\r"):
                self._stream(name, "\n")
        self._buf.clear()

    def process_message(self, msg, cell, cell_index):
        try:
            kind, content = msg["msg_type"], msg.get("content", {})
            if kind == "stream":
                self._stream(content.get("name", "stdout"), content.get("text", ""))
            elif kind == "error":
                self._flush()
                self._log(f"ERROR {content.get('ename')}: {_ANSI.sub('', str(content.get('evalue')))}")
        except Exception:                               # logging must never break the run
            pass
        return super().process_message(msg, cell, cell_index)


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
    logfile = [logfile, os.path.join(outdir, "run.log")]   # the cluster's own log, next to its results
    client = LoggingClient(nb, cluster, logfile, timeout=None, kernel_name="python3",
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
    log_environment(logfile)
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
