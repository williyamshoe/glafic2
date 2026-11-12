#!/usr/bin/env python
"""Profile likelihood in Ω_m (flat ΛCDM) for a finished cluster run, on the real images and on mocks.

    python mock_cosmo.py as1063                       # real data + noiseless mock + 3 noisy mocks
    python mock_cosmo.py as1063 --model bcg           # the run in results/as1063_bcg/
    python mock_cosmo.py as1063 --grid 0.1,0.3 --n-starts 2 --seeds   # quick look (no noisy mocks)

Is a low Ω_m a property of the data or of the method?  The mock images are made with the stage-A
best fit of results/<cluster>[_<model>]/ (calib.json: the mass model fitted at the input file's
cosmology, Ω_m = 0.3): for each source, the predicted image nearest to each observed image, at the
observed systems, redshifts and errors.  Each data set is then refitted (MAP, source positions
profiled, same σ, same priors and parameterization as the notebook) with Ω_m held fixed at every
grid value.  χ²_min(Ω_m) is the profile likelihood.

* real data:          where its minimum sits, and Δχ² between that and Ω_m = 0.3.
* noiseless mock:     must have its minimum at the true Ω_m with χ² ≈ 0; a minimum elsewhere is a
                      bias of the method (source-plane χ², parameterization, image matching).
* noisy mocks (σ):    the scatter of the minimum expected from noise alone.

The truth model reproduces the data only to σ, so the mock tests the method, not the mass model;
a mass-model error (missing BCG, too rigid a halo) is not in the mocks by construction.

Outputs go to results/<cluster>[_<model>]/mock/: one folder per data set (glafic files, the mock
images, profile.json with the cluster and model tag stamped in), and profile.png / profile.txt.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "python")))
import build_cluster as build  # noqa: E402
import glafic_jax as gj  # noqa: E402
import jax.numpy as jnp  # noqa: E402

GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]


def read_obs(path):
    # obs file -> list of [header line, [image lines]] (comments kept in the header)
    out, head = [], []
    for ln in open(path):
        t = ln.split()
        if not t or ln.lstrip().startswith("#"):
            head.append(ln)
        elif len(t) >= 4 and len(t) < 8:
            out.append([head + [ln], []]); head = []
        else:
            out[-1][1].append(ln)
    return out


def write_obs(path, blocks, xy):
    # blocks with the image positions replaced by xy (list of (n, 2) arrays, one per system)
    with open(path, "w") as f:
        for (head, imgs), new in zip(blocks, xy):
            f.writelines(head)
            for ln, (x, y) in zip(imgs, new):
                t = ln.split()
                f.write(f"{x:11.5f} {y:11.5f} " + " ".join(t[2:]) + "\n")


def make_model(d, cluster, om, lens=None, poi=None):
    # model in folder d at fixed flat ΛCDM with Ω_m = om
    m = gj.read_input(os.path.join(d, f"{cluster}.input"))
    for k in ("ovary", "wvary", "wavary", "hvary"):
        m.set_secondary(f"{k} 0")
    m.set_cosmo(om, 1.0 - om, -1.0, float(m.settings.hubble), 0.0)
    if lens is not None:
        m.lens_params = np.array(lens); m.poi_params = np.array(poi)
    m.model_init()
    m.set_source_mode("profile")
    return m


def reparam_of(setup):
    # the notebook's halo_reparam, read back from the setup's tag
    halo = setup["model"].get("halo", "anfw")
    a = re.search(r"nfwa([0-9.]+)z([0-9.]+)", setup["settings"])
    return {f"lens1_{halo}_p2": ("nfw_alpha", float(a.group(1)), float(a.group(2))) if a else "log10",
            "lens2_gals_p7": "log10"}


def fit(m, reparam, n_starts, seed):
    pipe = gj.Pipeline(m, reparam=reparam)
    for k, jit in enumerate((0.3, 1.0)):
        try:
            return float(pipe.map(n_starts=n_starts, jitter=jit, seed=seed + k, verbose=False)["chi2"])
        except ValueError as err:
            print(f"      MAP failed ({err}); retrying with wider starts", flush=True)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cluster")
    ap.add_argument("--model", default="", help="the run's GLAFIC_MODEL ('' = the automatic choice)")
    ap.add_argument("--grid", default=",".join(map(str, GRID)), help="Ω_m values, comma-separated")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3], help="noisy mocks (none: --seeds)")
    ap.add_argument("--n-starts", type=int, default=8, help="MAP starts per grid point")
    ap.add_argument("--no-data", action="store_true", help="skip the real data")
    a = ap.parse_args()

    run = os.path.join(HERE, "results", build.results_name(a.cluster, a.model))
    setup = json.load(open(os.path.join(run, "setup.json")))
    calib = json.load(open(os.path.join(run, "calib.json")))
    if (calib["cluster"], calib["tag"]) != (a.cluster, setup["tag"]):
        raise SystemExit(f"{run}/calib.json belongs to {calib['cluster']}/{calib['tag']}, not {setup['tag']}")
    if calib.get("parity"):
        print(f"note: the run fixes parities {calib['parity']}; this script fits without the parity penalty")
    grid = [float(v) for v in a.grid.split(",")]
    reparam = reparam_of(setup)
    out = os.path.join(run, "mock")
    os.makedirs(out, exist_ok=True)

    # truth: the stage-A fit, at the input file's cosmology
    om_true = float(gj.read_input(os.path.join(run, f"{a.cluster}.input")).settings.omega)
    truth = make_model(run, a.cluster, om_true, calib["lens"], calib["poi"])
    P = truth.params(); sig = calib["sigma"]
    blocks = read_obs(os.path.join(run, f"{a.cluster}_obs.dat"))
    clean = []
    for i, o in enumerate(truth.obs_point):
        pred = np.asarray(truth.findimg(i + 1, P=P))[:, :2]
        obs = np.asarray(o["tab"])[:, :2]
        if len(pred) < len(obs):
            raise SystemExit(f"truth model forms {len(pred)} < {len(obs)} images of system {i + 1}")
        # assign predicted images to observed ones, closest pairs first, each used once
        d = np.hypot(*(obs[:, None, :] - pred[None, :, :]).transpose(2, 0, 1))
        pick = -np.ones(len(obs), int)
        for _ in range(len(obs)):
            j, k = np.unravel_index(np.argmin(d), d.shape)
            pick[j] = k; d[j, :] = np.inf; d[:, k] = np.inf
        clean.append(pred[pick])
    off = np.concatenate([np.hypot(*(c - np.asarray(o["tab"])[:, :2]).T) for c, o in zip(clean, truth.obs_point)])
    print(f"{a.cluster} ({setup['tag']}): truth = stage-A fit at Ω_m = {om_true}; mock - observed offsets: "
          f"rms {np.sqrt(np.mean(off ** 2)):.3f}\", max {off.max():.3f}\" (σ = {sig:.3f}\")")

    sets = ([] if a.no_data else [("data", None)]) + [("mock_noiseless", clean)]
    for s in a.seeds:
        rng = np.random.default_rng(s)
        sets.append((f"mock_noise{s}", [c + sig * rng.standard_normal(c.shape) for c in clean]))

    results = {}
    for name, xy in sets:
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for f in (f"{a.cluster}.input", f"{a.cluster}_galfile.dat", f"{a.cluster}_prior.dat", f"{a.cluster}_obs.dat"):
            shutil.copy(os.path.join(run, f), d)
        if xy is not None:
            write_obs(os.path.join(d, f"{a.cluster}_obs.dat"), blocks, xy)
        pf = os.path.join(d, "profile.json")
        prof = json.load(open(pf)) if os.path.exists(pf) else {}
        if prof and (prof.get("cluster"), prof.get("tag")) != (a.cluster, setup["tag"]):
            raise SystemExit(f"{pf} belongs to another run; delete {out}")
        chi2 = {float(k): v for k, v in prof.get("chi2", {}).items()}
        for om in grid:
            if om in chi2:
                continue
            t0 = time.time()
            m = make_model(d, a.cluster, om, calib["lens"], calib["poi"])
            chi2[om] = fit(m, reparam, a.n_starts, seed=int(1000 * om))
            print(f"   {name:15s} Ω_m = {om:5.2f}: χ² = {chi2[om]:9.2f}  ({time.time() - t0:.0f} s)", flush=True)
            json.dump(dict(cluster=a.cluster, tag=setup["tag"], sigma=sig, om_true=om_true,
                           chi2={str(k): v for k, v in sorted(chi2.items())}), open(pf, "w"), indent=1)
        results[name] = chi2

    # summary
    lines = [f"{a.cluster} ({setup['tag']}): profile χ²_min(Ω_m), flat ΛCDM, σ = {sig:.3f}\"; "
             f"mocks made at Ω_m = {om_true}",
             f"{'data set':16s}" + "".join(f"{om:>9.2f}" for om in grid) + "   best Ω_m   Δχ²(0.3)"]
    for name, c in results.items():
        v = np.array([c[om] for om in grid]); b = grid[int(np.nanargmin(v))]
        ref = c.get(0.3, np.nan)
        lines.append(f"{name:16s}" + "".join(f"{x:9.2f}" for x in v) + f"   {b:8.2f}   {ref - np.nanmin(v):8.2f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(out, "profile.txt"), "w").write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in results.items():
        v = np.array([c[om] for om in grid])
        style = dict(color="k", lw=2) if name == "data" else dict(color="tab:blue", lw=2) \
            if name == "mock_noiseless" else dict(color="tab:blue", lw=0.8, alpha=0.6)
        ax.plot(grid, v - np.nanmin(v), "o-", ms=3, label=name, **style)
    ax.axvline(om_true, color="0.5", ls=":")
    ax.set_xlabel(r"$\Omega_m$ (fixed)"); ax.set_ylabel(r"$\chi^2_{\min} - \min$")
    ax.set_ylim(0, None); ax.set_title(f"{a.cluster}: profile likelihood"); ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "profile.png"), dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, HERE)}/profile.png, profile.txt")


if __name__ == "__main__":
    main()
