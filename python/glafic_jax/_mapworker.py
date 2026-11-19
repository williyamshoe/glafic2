"""Worker process for ``Pipeline.map(n_runs > 1)``.

    python -m glafic_jax._mapworker <task> <result>

Runs the MAP fits listed in the task file (written by ``Pipeline._map_processes``)
on the one device this process sees (the parent sets ``CUDA_VISIBLE_DEVICES``),
and writes ``[(run, (z, chi2) or None, device, seconds)]`` to the result file.
"""
import os
import sys
import time


def main(task, result):
    import glafic_jax  # noqa: F401  (sets jax_enable_x64 before any array is unpickled)
    import cloudpickle
    import jax
    import numpy as np

    with open(task, "rb") as f:
        t = cloudpickle.load(f)
    pipe = cloudpickle.loads(t["pipe"])
    dev = str(jax.devices()[0]) + (f" (GPU {os.environ['CUDA_VISIBLE_DEVICES']})"
                                   if os.environ.get("CUDA_VISIBLE_DEVICES") else "")
    out = []
    for r, seed, center in t["runs"]:
        t0 = time.time()
        best = pipe._map_run(*t["args"], np.random.default_rng(seed), center, t["lp_kw"], t["verbose"],
                             f" run {r} ({dev})")
        out.append((r, None if best is None else (np.asarray(best[0]), float(best[1])), dev, time.time() - t0))
        print(f"[MAP worker] run {r} on {dev}: chi2 = {np.inf if best is None else best[1]:.6g} "
              f"({out[-1][3]:.0f} s)", flush=True)
    with open(result + ".tmp", "wb") as f:
        cloudpickle.dump(out, f)
    os.replace(result + ".tmp", result)


if __name__ == "__main__":
    main(*sys.argv[1:3])
