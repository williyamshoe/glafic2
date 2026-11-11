"""Generate cluster_cosmo_models.ipynb: one cluster, three cosmologies (flat ΛCDM, wCDM, w0waCDM).

    python make_notebook_cosmo.py

A trimmed variant of make_notebook.py (no AIES validation, no sampler
diagnostics section) that samples Ω_m in ΛCDM, (Ω_m, w) in wCDM and
(Ω_m, w0, wa) in w0waCDM, and compares the three in one corner plot.

The notebook models any cluster in example_data/<name>/ (see build_cluster.py
for the file format).  Pick the cluster with CLUSTER in the configuration cell
or the GLAFIC_CLUSTER environment variable (handy for batch jobs).

It is designed to run on a GPU node (e.g. NERSC Perlmutter): every stage
caches its result in results/, and the samplers write checkpoints, so it can
be run piecewise across batch jobs with time limits.
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Strong-lensing cosmography of a galaxy cluster: ΛCDM, wCDM and w0waCDM

This notebook models one cluster from `example_data/` and samples the lens
mass model jointly with the cosmology in three flat models, using Hamiltonian
Monte Carlo (NUTS) with exact JAX gradients:

| model | free cosmological parameters |
|---|---|
| ΛCDM | Ω_m |
| wCDM | Ω_m, w |
| w0waCDM | Ω_m, w0, wa |

It is a trimmed version of `cluster_cosmo_hmc.ipynb`, which also has a
gradient-free (AIES) validation of NUTS and a sampler-diagnostics section.

**Choosing a cluster.** Set `CLUSTER` in the next cell to any folder name in
`example_data/`, or set the environment variable `GLAFIC_CLUSTER`. Each folder
holds catalogues in the glafic public-model format:

| file | columns |
|---|---|
| `glafic_<name>_vN_mulimage.txt` | ID (`system.image`), RA, Dec, positional error ["], z_s, z_s error (> 0: photometric, sampled) |
| `glafic_<name>_vN_memgal.txt` | RA, Dec, L/L*, e, PA of the cluster members (the brightest is the BCG) |
| `glafic_<name>_vN_readme.txt` | lens redshift (`# zl = ...`) and, optionally, the published fit (`# chi2 = .. , dof = ..`) |

With `SPEC_ONLY = True` (the default) systems with a photometric redshift are
left out entirely, so every source redshift is fixed. This also changes the
number of constraints, so the χ² no longer compares directly with the published
fit.

Clusters marked in the `exclude` column of
`example_data/venus_clusters_summary.csv` are judged too complex or not
worth modelling. The notebook refuses to run one unless `RUN_EXCLUDED = True`.
Rows are matched to folders by lens redshift, number of systems and published
χ², or by a `folder` column if you add one.

To add a cluster, drop a new folder in `example_data/`. `build_cluster.py`
lists what it finds (`python build_cluster.py`).

**Mass model.** The glafic recipe of Oguri et al. (VENUS, 2026, Sec. 2.3):

* an elliptical NFW halo, with all 6 parameters free;
* pseudo-Jaffe members with σ ∝ L^1/4 and r_trun ∝ L^η, where σ*, r_trun* and
  η are free;
* optional external shear (`shear`, off by default);
* optional multipoles centred on the BCG (`mpoles`) and an optional separate
  BCG (`bcg`).

Every cluster runs with no per-cluster settings; additions go in
`CLUSTER_MODEL`. The shear is off by default because it is often consistent
with zero. Its angle is then unconstrained, and the amplitude–angle pair
forms a funnel at γ = 0 that NUTS samples poorly.

**Image parities** (the sign of 1/μ). The source-plane χ² does not check
which side of a critical curve an image lies on. For an image pair merging
across a critical curve, this lets the posterior split into a physical mode,
with the curve between the two images, and an unphysical one. With
`PARITY = "auto"` (the default), stage A finds the best fit without a
parity constraint. Every **close pair** (images of one system closer than
`PAIR_SEP`) that has opposite parities there then has both parities fixed.
Close pairs with the same parity are reported but not constrained: if one of
them is really a merging pair, the best fit is in the wrong mode, and you set
its signs by hand (`parity={"3.3": -1, "3.4": +1}` in `CLUSTER_MODEL`).

The constraint is a steep χ² penalty on the true 1/μ at the observed
positions (`PARITY_TOL`), not glafic's own parity column. In the
source-plane χ², glafic checks the sign of a linearly corrected
magnification, which for |μ| ≳ 100 is dominated by the correction and has an
essentially random sign.

For RXJ2129, the base recipe reproduces the published fit but fails to form
the image pair 3.2/3.3, and an m = 3 multipole fixes that. For a new cluster,
start from the base recipe and let the image check in stage A tell you
whether anything is missing.

### Likelihood

We use glafic's **source-plane χ²** (`chi2_splane 1`), the choice of most
published cluster models:

* Each image's offset is mapped back to the source plane with the
  magnification tensor of the lens model.
* The source positions are **integrated out analytically**:
  χ² → χ²_min + ln det M, which removes two nuisance parameters per system.
* Positional errors are **rescaled** so that the fixed-cosmology best fit has
  χ²/dof = 1.

The μ² weighting in this χ² is what stops a model from placing images on
critical curves, where any image offset maps to a tiny source-plane residual.
Images that really do lie close to critical curves make the likelihood
sharply peaked, so NUTS needs long warmup and a high target acceptance. Do
**not** use `mag_cap`: capping the magnification removes that protection.

### Stages

Each stage caches its result in `results/<cluster>/<model tag>/`, and
re-running skips finished stages. Every cluster gets its own folder, and
changing the multipoles, the BCG or the halo-mass parameterization changes the
model tag, so stale chains are never reused. Each cached file also records its
cluster and model tag, and loading one into a different run raises an error.

| stage | what |
|---|---|
| A | best fit at fixed cosmology (Ω_m = 0.3, w = −1); error calibration; parities; image check |
| B | NUTS for each cosmological model in `MODELS` (ΛCDM, wCDM, w0waCDM) |
| C | cosmology corner plot of all models; lens-parameter table; posterior-predictive check |

`MODELS` in the next cell picks which cosmologies to sample, for example one
per batch job; with `MODELS = []` only stage A runs. The samplers write
a checkpoint every `chunk` samples, so a job that hits its time limit resumes
where it stopped.
""")

code(r"""
import os, sys, re, json, time, math
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.abspath(".")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
try:
    import glafic_jax as gj
except ImportError:                      # not pip-installed: use the repository copy
    sys.path.insert(0, os.path.join(REPO, "python"))
    import glafic_jax as gj
import jax, jax.numpy as jnp
import build_cluster as build

print("jax", jax.__version__, "| backend:", jax.default_backend(), "| devices:", jax.devices())
EXCLUDED = build.excluded_clusters()      # marked in example_data/venus_clusters_summary.csv
print("clusters to model:", build.available(excluded=False))
print("excluded:", EXCLUDED)

# ---------------- configuration ----------------
CLUSTER = os.environ.get("GLAFIC_CLUSTER", "rj2129")   # folder name in example_data/
RUN_EXCLUDED = False              # allow a cluster marked in the summary table's 'exclude' column
if CLUSTER in EXCLUDED and not RUN_EXCLUDED:
    raise ValueError(f"{CLUSTER} is excluded in venus_clusters_summary.csv; set RUN_EXCLUDED = True to run it")
CLUSTER_MODEL = {                 # per-cluster additions to the base recipe
    "rj2129": dict(mpoles=[(3, 2)]),       # m = 3 needed to form images 3.2/3.3
    # "<name>": dict(shear=True, bcg=True, mpoles=[(4, 2)], parity={"3.3": -1, "3.4": +1}),
}
MPOLES = CLUSTER_MODEL.get(CLUSTER, {}).get("mpoles", [])   # (m, n) multipoles centred on the BCG
BCG = CLUSTER_MODEL.get(CLUSTER, {}).get("bcg", False)      # model the BCG separately
SHEAR = CLUSTER_MODEL.get(CLUSTER, {}).get("shear", False)  # external shear (amplitude, angle)
PARITY = CLUSTER_MODEL.get(CLUSTER, {}).get("parity", "auto")
# "auto": fix the parities of close pairs that straddle a critical curve at the best fit;
# {image ID: ±1}: required signs of 1/μ, set by hand;  {}: no parity constraint
PAIR_SEP = 2.0                    # [arcsec] images of one system closer than this form a close pair
SPEC_ONLY = True                  # use only spectroscopically confirmed systems (drop photometric z_s)
MAG_CAP = 0.0                     # 0 = glafic's exact source-plane chi2 (do not cap; see above)
POS_ERR0 = 0.4                    # catalogue positional error [arcsec], before rescaling
CALIBRATE_ERRORS = True           # rescale errors so that the fixed-cosmology best fit has chi2/dof = 1
N_STARTS = 16                     # jittered L-BFGS starts for the MAP
USE_NFW_ALPHA = True              # halo mass: sample log10 of the NFW deflection at R (straightens M-c)
NFW_ALPHA_R = None                # [arcsec]; None = median image distance from the BCG
NFW_ALPHA_ZS = None               # source z for the deflection; None = median z_s
COSMO_REPARAM = {"wa": ("offset", "weos", -3.0, 0.0)}   # sample w0 + wa in [-3, 0)
NUTS_KW = dict(num_chains=4, num_warmup=1000, num_samples=1000, target_accept=0.95,
               max_tree_depth=10, init="map", init_jitter=0.01, use_approx_mass=False, chunk=250)
MODELS = ["lcdm", "wcdm", "w0wa"]   # cosmologies to sample in stage B (finished ones are loaded from disk)
SMOKE_TEST = False                # tiny runs, to check that the notebook works end to end

RESULTS = os.path.join(HERE, "results_smoke" if SMOKE_TEST else "results", CLUSTER)
if SMOKE_TEST:
    NUTS_KW.update(num_chains=2, num_warmup=20, num_samples=20, chunk=10, max_tree_depth=5)
""")

md("## 1. Data")

code(r"""
cl = build.load(CLUSTER, spec_only=SPEC_ONLY)
print(build.summary(cl))
if PARITY != "auto":
    bad = set(PARITY) - {iid for s in cl["systems"].values() for iid in s["ids"]}
    if bad:
        raise ValueError(f"PARITY refers to images not in the fit (photometric systems with SPEC_ONLY?): "
                         f"{sorted(bad)}")
if len({round(s["zs"], 3) for s in cl["systems"].values()}) < 2:
    print("WARNING: all systems are at one source redshift; the lens model can be fitted, "
          "but the cosmology is unconstrained (it needs distance ratios between source planes)")
systems = cl["systems"]
SIDS = list(systems)                                   # catalogue system IDs, in glafic point order
xy_all = np.concatenate([np.array(s["xy"]) for s in systems.values()])
r_img = np.hypot(*xy_all.T)
zs_all = np.array([s["zs"] for s in systems.values()])
LIM = 5.0 * math.ceil(1.25 * r_img.max() / 5.0)        # plot half-width [arcsec]

# halo-mass parameterization (data-driven defaults)
if NFW_ALPHA_R is None:
    NFW_ALPHA_R = float(np.round(2 * np.median(r_img)) / 2)
if NFW_ALPHA_ZS is None:
    NFW_ALPHA_ZS = float(np.round(np.median(zs_all), 2))
REPARAM = {"lens1_anfw_p2": ("nfw_alpha", NFW_ALPHA_R, NFW_ALPHA_ZS) if USE_NFW_ALPHA else "log10",
           "lens2_gals_p7": "log10"}          # member truncation radius: flat in log10

# results folder for this model setup
TAG = "_".join(["base"] + (["speconly"] if SPEC_ONLY else []) + ([] if SHEAR else ["noshear"]) + [f"mpole{m}{n}" for m, n in MPOLES]
               + (["bcg"] if BCG else [])
               + (["autopar"] if PARITY == "auto" else ["mupar" + "-".join(sorted(PARITY))] if PARITY else [])
               + [f"nfwa{NFW_ALPHA_R:g}z{NFW_ALPHA_ZS:g}" if USE_NFW_ALPHA else "log10M"])
RUN_DIR = os.path.join(RESULTS, TAG)
os.makedirs(RUN_DIR, exist_ok=True)
print(f"model: {TAG}  ->  {RUN_DIR}")


def write_files(outdir, pos_err):
    return build.write_all(CLUSTER, outdir, mpoles=MPOLES, bcg=BCG, shear=SHEAR, pos_err=pos_err,
                           spec_only=SPEC_ONLY)
    # parities are not written to the obs file (see add_parity_penalty)


write_files(RUN_DIR, POS_ERR0)
print(open(os.path.join(RUN_DIR, f"{CLUSTER}.input")).read().split("start_command")[0])

gals = cl["gals"][1:] if BCG else cl["gals"]
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(gals[:, 0], gals[:, 1], s=150 * gals[:, 2] ** 0.5, c="0.75", label="members (size ∝ √L)")
for k, (sid, s) in enumerate(systems.items()):
    xy = np.array(s["xy"])
    ax.scatter(xy[:, 0], xy[:, 1], s=35, color=cmap(k % 20), edgecolor="k", zorder=3,
               label=f"{sid}: z={s['zs']:.2f}" + (f"±{s['zerr']:.1f}" if s['zerr'] > 0 else ""))
ax.set_xlim(LIM, -LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
ax.set_xlabel("x [arcsec] (west →)"); ax.set_ylabel("y [arcsec]"); ax.legend(fontsize=7, ncol=2)
ax.set_title(CLUSTER)
plt.show()
""")

code(r"""
def capdir(cap):
    d = os.path.join(RUN_DIR, f"cap{cap:g}")
    os.makedirs(d, exist_ok=True)
    return d


# free cosmological parameters of each model (glafic flags), and its label
COSMO_MODELS = {"fixed": ((), "fixed cosmology"),
                "lcdm": (("ovary",), "ΛCDM"),
                "wcdm": (("ovary", "wvary"), "wCDM"),
                "w0wa": (("ovary", "wvary", "wavary"), "w0waCDM")}


def make_model(cap, cosmo="fixed", lens=None, poi=None):
    m = gj.read_input(os.path.join(capdir(cap), f"{CLUSTER}.input"))
    free = COSMO_MODELS[cosmo][0]
    for k in ("ovary", "wvary", "wavary"):
        m.set_secondary(f"{k} {1 if k in free else 0}")
    m.set_secondary(f"mag_cap {cap}")
    if lens is not None:
        m.lens_params = np.array(lens); m.poi_params = np.array(poi)
    m.model_init()
    return add_parity_penalty(m)


PARITY_TOL = 1e-4                 # softness of the parity wall, in 1/μ
# parities in force: set by hand, or (PARITY = "auto") found from the best fit in stage A
PARITY_FIX = {} if PARITY == "auto" else dict(PARITY)
IMG_IDS = [iid for s in systems.values() for iid in s["ids"]]   # same order as the obs file
CLOSE_PAIRS = [(s["ids"][a], s["ids"][b]) for s in systems.values()
               for a in range(len(s["ids"])) for b in range(a + 1, len(s["ids"]))
               if np.hypot(*np.subtract(s["xy"][a], s["xy"][b])) < PAIR_SEP]


def add_parity_penalty(m):
    # chi2 += sum ( min(0, parity * 1/μ) / PARITY_TOL )^2 at the observed positions of the PARITY_FIX images.
    # Zero on the required side of the critical curve, Δχ² = 100 at 1/μ = 1e-3 on the wrong side; smooth,
    # so the MAP can climb out of the wrong side.  Applies to the MAP (m.chi2) and the samplers (log_prob).
    terms = []
    for k, s in enumerate(systems.values()):           # glafic point k = catalogue system k
        js = [j for j, iid in enumerate(s["ids"]) if iid in PARITY_FIX]
        if js:
            tab = m.obs_point[k]["tab"]
            terms.append((k, jnp.asarray(tab[js, 0]), jnp.asarray(tab[js, 1]),
                          jnp.asarray([float(PARITY_FIX[s["ids"][j]]) for j in js]), int(m.nlp_point[k])))
    if not terms:
        return m
    ls, base = m.lens_system, m.chi2

    def chi2_parity(P):
        c = 0.0
        for k, x, y, sgn, nlp in terms:
            mi = ls.ray_shoot(x, y, P["lens"], P["point"][k, 0], P["cosmo"], nlp=nlp).muinv
            c = c + jnp.sum((jnp.minimum(sgn * mi, 0.0) / PARITY_TOL) ** 2)
        return c

    m.chi2_parity = chi2_parity
    m.chi2 = lambda P=None, **kw: base(P, **kw) + chi2_parity(m.params() if P is None else P)
    return m


def dof_of(m):
    nimg = sum(o["tab"].shape[0] for o in m.obs_point)
    nlens = sum(n.startswith("lens") for n in m.theta_names)
    return 2 * nimg - 2 * m.num_poi - nlens


def muinv_obs(m, P):
    # signed 1/μ at every observed image position, in IMG_IDS order
    return np.concatenate([np.asarray(m.lensmodel(jnp.asarray(o["tab"][:, 0]), jnp.asarray(o["tab"][:, 1]),
                                                  float(P["point"][i, 0]), P).muinv)
                           for i, o in enumerate(m.obs_point)])


def auto_parity(m, P):
    # parities of the close pairs that straddle a critical curve (opposite signs of 1/μ) in model P
    mi = muinv_obs(m, P)
    out = {}
    for a, b in CLOSE_PAIRS:
        ia, ib = IMG_IDS.index(a), IMG_IDS.index(b)
        sa, sb = int(np.sign(mi[ia])), int(np.sign(mi[ib]))
        txt = f"   close pair {a}/{b}: 1/μ = {mi[ia]:+.4f} / {mi[ib]:+.4f}"
        if sa * sb < 0:
            out[a], out[b] = sa, sb
            print(txt + "  ->  straddles a critical curve; parities fixed")
        else:
            print(txt + "  ->  same parity, not constrained (if this is a merging pair, the best fit is "
                  "in the wrong mode: set its parity in CLUSTER_MODEL)")
    return out


def check_owner(d, path):
    # every cached result records the cluster and model it belongs to; refuse to mix them up
    got = (str(d.get("cluster", CLUSTER)), str(d.get("tag", TAG)))
    if got != (CLUSTER, TAG):
        raise ValueError(f"{path} belongs to cluster {got[0]!r} / model {got[1]!r}, "
                         f"not {CLUSTER!r} / {TAG!r}; move or delete it")
    return d


def save_npz(path, **kw):
    np.savez(path, cluster=CLUSTER, tag=TAG, **kw)


def load_npz(path):
    return check_owner(dict(np.load(path, allow_pickle=True)), path) if os.path.exists(path) else None


# readable parameter names
PLAB = {("anfw", 2): "M_vir [M_sun/h]", ("anfw", 3): "x_halo", ("anfw", 4): "y_halo",
        ("anfw", 5): "e_halo", ("anfw", 6): "θ_halo", ("anfw", 7): "c_halo",
        ("gals", 2): "σ* [km/s]", ("gals", 7): "r_trun* [arcsec]", ("gals", 8): "η",
        ("pert", 5): "γ_ext", ("pert", 6): "θ_γ", ("jaffe", 2): "σ_BCG [km/s]", ("jaffe", 7): "r_trun,BCG"}
COSMO_LAB = {"omega": "Ω_m", "weos": "w0", "wa": "wa", "hubble": "h", "lambda": "Ω_Λ"}
N_BASE = 2 + int(SHEAR) + int(BCG)                     # lenses before the multipoles


def label(n):
    n = str(n)
    m = re.fullmatch(r"lens(\d+)_([a-z0-9]+)_p(\d+)", n)
    if m:
        i, mod, p = int(m.group(1)), m.group(2), int(m.group(3))
        if mod == "mpole" and p in (5, 6):
            mm = MPOLES[i - N_BASE - 1][0]
            return f"ε_{mm}" if p == 5 else f"θ_{mm}"
        return PLAB.get((mod, p), n)
    m = re.fullmatch(r"point(\d+)_zs", n)
    if m:
        return f"z_s (system {SIDS[int(m.group(1)) - 1]})"
    return COSMO_LAB.get(n, n)
""")

md(r"""
## 2. Stage A: best fit at fixed cosmology and error calibration

1. **Best fit.** Minimise the source-plane χ² over the lens parameters and
   photometric redshifts, with source positions profiled, from `N_STARTS`
   jittered starts.
2. **Error calibration.** All images share one positional error σ, and the
   χ² scales as 1/σ². Setting σ = σ₀ √(χ²/dof) makes χ²/dof = 1 exactly,
   without changing the best fit.
3. **Parities** (`PARITY = "auto"`). The close pairs that straddle a critical
   curve at the best fit get their parities fixed for all later stages. The
   penalty is zero at the best fit, so the fit and the calibration do not
   change. The signs are saved in `calib.json`.

If the readme gives a published fit, its χ² (at the catalogue errors) is
printed for comparison. The two match only when the model and the number of
free parameters match the paper's.
""")

code(r"""
def stage_a(cap, start=None, n_starts=N_STARTS):
    f = os.path.join(capdir(cap), "calib.json")
    if os.path.exists(f):
        return check_owner(json.load(open(f)), f)
    write_files(capdir(cap), POS_ERR0)
    m = make_model(cap, **(start or {}))        # with PARITY = "auto", no parity constraint yet
    m.set_source_mode("profile")
    dof = dof_of(m)
    if dof <= 0:
        raise ValueError(f"{CLUSTER}: {dof} degrees of freedom ({sum(len(s['xy']) for s in systems.values())} "
                         f"images in {len(systems)} systems): too few constraints for this model")
    pipe = gj.Pipeline(m, reparam=REPARAM)
    for k, (ns, jit) in enumerate([(n_starts, 0.5), (4 * n_starts, 1.0), (8 * n_starts, 1.5)]):
        try:
            r = pipe.map(n_starts=ns, jitter=jit, seed=k, verbose=False)
            break
        except ValueError as err:
            print(f"MAP attempt {k + 1}: {err}; retrying with more, wider starts")
    else:
        raise RuntimeError("MAP failed from every start")
    P = m.best_sources(m.unflatten(jnp.asarray(r["theta"])))
    if PARITY == "auto":                       # read the parities off the best fit (zero penalty there)
        print("automatic parities from the best fit:")
        parity = auto_parity(m, P)
    else:
        parity = dict(PARITY)
    sig = POS_ERR0 * np.sqrt(r["chi2"] / dof) if CALIBRATE_ERRORS else POS_ERR0
    write_files(capdir(cap), sig)
    out = dict(cluster=CLUSTER, tag=TAG, cap=cap, sigma=float(sig), chi2_raw=float(r["chi2"]), dof=int(dof),
               lens=np.asarray(P["lens"]).tolist(), poi=np.asarray(P["point"]).tolist(), parity=parity)
    json.dump(out, open(f, "w"))
    return out


t0 = time.time()
calib = {MAG_CAP: stage_a(MAG_CAP)}        # cached after the first run; every later stage needs it
print(f"stage A: {time.time() - t0:.0f} s")
PARITY_FIX = {k: int(v) for k, v in calib[MAG_CAP]["parity"].items()}   # in force from here on
print(f"parity constraints{' (automatic)' if PARITY == 'auto' else ''}: "
      + (", ".join(f"{k}: {v:+d}" for k, v in PARITY_FIX.items()) or "none"))
for cap, c in calib.items():
    m = make_model(cap, lens=c["lens"], poi=c["poi"]); m.set_source_mode("profile")
    c2 = float(m.chi2(m.params()))
    print(f"cap {cap:g}: chi2 = {c['chi2_raw']:.2f} with σ = {POS_ERR0}\"  ->  σ = {c['sigma']:.3f}\", "
          f"chi2 = {c2:.2f} for {c['dof']} dof")
    if PARITY_FIX:
        print(f"   parity penalty at the MAP: {float(m.chi2_parity(m.params())):.3g} (should be 0)")
ref = cl["ref"]
if "chi2" in ref:
    print(f"published fit ({ref.get('model_ref', 'readme')}): chi2 = {ref['chi2']:.2f} for {ref['dof']:.0f} dof"
          + (f", rms = {ref['rms']:.2f}\"" if "rms" in ref else ""))
""")

md(r"""
### Check with glafic's exact image finder

The source-plane χ² does not check which images the model actually forms.
For the best fit, we solve the lens equation with glafic's adaptive-mesh
image finder. We then compare the predicted and observed images, and
overlay the critical curves for the lowest, median and highest
spectroscopic source redshifts.
""")

code(r"""
c = calib[MAG_CAP]
m = make_model(MAG_CAP, lens=c["lens"], poi=c["poi"])
P = m.params()
zspec = sorted({s["zs"] for s in systems.values() if s["zerr"] == 0}) or sorted(set(zs_all))
ZCRIT = sorted({zspec[0], zspec[len(zspec) // 2], zspec[-1]})
fig, ax = plt.subplots(figsize=(6.5, 6.5))
g = np.linspace(-LIM, LIM, 501); X, Y = np.meshgrid(g, g)
for zs, col in zip(ZCRIT, ("tab:blue", "tab:green", "tab:red")):
    mu = jax.jit(lambda x, y: m.lensmodel(x, y, zs, P).muinv)(jnp.asarray(X), jnp.asarray(Y))
    ax.contour(X, Y, np.asarray(mu), levels=[0.0], colors=col, linewidths=0.8)
    ax.plot([], [], color=col, label=f"critical curve z_s={zs:.2f}")
offs, mus, missing = [], [], []
for i in range(m.num_poi):
    obs = m.obs_point[i]["tab"]; pred = m.findimg(i + 1)
    o = [np.sqrt(((pred[:, :2] - p[:2]) ** 2).sum(1).min()) if len(pred) else np.inf for p in obs]; offs += o
    mu = np.abs(1 / np.asarray(m.lensmodel(jnp.asarray(obs[:, 0]), jnp.asarray(obs[:, 1]),
                                           float(P["point"][i, 0]), P).muinv)); mus += list(mu)
    if pred.shape[0] < obs.shape[0]:
        missing.append(SIDS[i])
    print(f"system {str(SIDS[i]):>4s} (z={P['point'][i,0]:.3f}): {obs.shape[0]} observed / {pred.shape[0]} predicted, "
          f"offsets {np.round(o, 2)}, |μ| {np.round(mu, 1)}")
    ax.scatter(obs[:, 0], obs[:, 1], marker="o", facecolor="none", edgecolor="k", s=45)
    ax.scatter(pred[:, 0], pred[:, 1], marker="x", color=cmap(i % 20), s=30)
ax.set_xlim(LIM, -LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal"); ax.legend(fontsize=8)
ax.set_title(f"{CLUSTER}: best fit (Ω_m = 0.3, w = −1)")
plt.show()
rms = np.sqrt(np.mean(np.square(offs)))
print(f"image-plane rms = {rms:.3f} arcsec (expect ~{c['sigma']:.2f}); median |μ| = {np.median(mus):.1f}; "
      f"systems with missing images: {missing or 'none'}")
if rms > 1.0 or missing:
    print("WARNING: the MAP does not reproduce the observed images. Do not run the samplers from it; "
          "try more N_STARTS, a different seed, or a richer model in CLUSTER_MODEL (multipoles, bcg).")
""")

md(r"""
## 3. Stage B: NUTS for each cosmology

For each model in `MODELS`, NUTS samples the free lens parameters, the
photometric redshifts (if any) and that model's cosmological parameters, with
the source positions marginalised. The priors are flat:

* Ω_m ∈ [0.05, 1] (all models);
* w ∈ [−2.5, 0] (wCDM), w0 ∈ [−2.5, 0] (w0waCDM);
* **w0 + wa ∈ [−3, 0)** (w0waCDM), sampled directly, which builds the
  condition w0 + wa < 0 (no dark-energy domination at early times) into the
  parameter space instead of a hard wall in the posterior.

Each orientation angle is restricted to one period around the best fit
(`gj.angle_bounds`). Chains start near the stage-A best fit, at Ω_m = 0.3,
w = −1, wa = 0, and the dense mass matrix is adapted during warmup.

Healthy output: few divergences, R-hat < 1.01–1.02, ESS of a few hundred or
more. If the median number of gradient evaluations per sample is
2^max_tree_depth − 1, NUTS is running out of tree depth.
""")

code(r"""
NUTS_FILE = {"lcdm": "nuts_lcdm_om.npz", "wcdm": "nuts_wcdm.npz", "w0wa": "nuts_w0wa.npz"}
REPARAM_OF = {"lcdm": REPARAM, "wcdm": REPARAM, "w0wa": dict(REPARAM, **COSMO_REPARAM)}


def nuts_stage(path, model, theta0, reparam, **over):
    out = load_npz(path)
    if out is not None:
        return out
    pipe = gj.Pipeline(model, reparam=reparam, bounds=gj.angle_bounds(model, theta0))
    pipe.set_map(theta0)
    h = pipe.hmc(checkpoint=path.replace(".npz", ".ckpt"), **dict(NUTS_KW, **over))
    d = gj.diagnostics(h)
    save_npz(path, theta=h["theta"], log_prob=h["log_prob"], diverging=h["diverging"],
             num_steps=h["num_steps"], names=np.array(h["names"]), rhat=d["rhat"], ess=d["ess"],
             time=h["time"])
    return load_npz(path)


def is_source(n):
    return n.startswith("point") and not n.endswith("zs")


def show_diag(res):
    names = list(res["names"]); keep = [k for k, n in enumerate(names) if not is_source(n)]
    th = res["theta"].reshape(-1, len(names))
    print(f"{float(res['time'])/60:.0f} min; divergences {int(res['diverging'].sum())} / {res['diverging'].size}; "
          f"max R-hat {np.nanmax(res['rhat'][keep]):.3f}; min ESS {np.nanmin(res['ess'][keep]):.0f}; "
          f"median {np.median(res['num_steps']):.0f} gradient evaluations per sample")
    print(f"{'parameter':22s} {'mean':>11s} {'std':>10s} {'R-hat':>7s} {'ESS':>7s}")
    for k in keep:
        n = names[k]
        print(f"{label(n):22s} {th[:, k].mean():11.4g} {th[:, k].std():10.3g} "
              f"{res['rhat'][k]:7.3f} {res['ess'][k]:7.0f}")


def cosmo_model(cap, cosmo):
    c = calib[cap]
    mm = make_model(cap, cosmo=cosmo, lens=c["lens"], poi=c["poi"])
    mm.set_source_mode("marginalize")
    return mm


nuts = {}                          # cosmology -> finished NUTS result
for cosmo in NUTS_FILE:
    path = os.path.join(capdir(MAG_CAP), NUTS_FILE[cosmo])
    res = load_npz(path)
    if res is None and cosmo in MODELS:
        mm = cosmo_model(MAG_CAP, cosmo)
        print(f"--- {COSMO_MODELS[cosmo][1]}: {mm.ndim} sampled parameters")
        res = nuts_stage(path, mm, mm.theta0(), REPARAM_OF[cosmo])
    if res is not None:
        nuts[cosmo] = res
        print(f"=== {COSMO_MODELS[cosmo][1]}")
        show_diag(res)
        print()
""")

md(r"""
## 4. Stage C: cosmology

One corner plot for all models. Columns are Ω_m, w0 (w for wCDM) and wa; a
model is drawn only in the panels of its own free parameters, so ΛCDM appears
only in the Ω_m histogram and wCDM is absent from every wa panel. The grey
dashed curves are the priors: where a posterior follows its prior, the data
do not constrain that parameter. Shading marks the 68% (darker) and 95%
(lighter) regions. Dotted lines mark (Ω_m, w0, wa) = (0.3, −1, 0).
""")

code(r"""
import corner
import matplotlib.colors

COS = [("omega", r"$\Omega_m$", "Ω_m", (0.05, 1.0)),
       ("weos", r"$w_0$ ($w$ in wCDM)", "w0 (w)", (-2.5, 0.0)),
       ("wa", r"$w_a$", "wa", (-3.0, 3.0))]
REF = {"omega": 0.3, "weos": -1.0, "wa": 0.0}
COLORS = {"lcdm": "tab:red", "wcdm": "tab:green", "w0wa": "tab:blue"}


def cosmo_columns(res):
    names = list(res["names"]); th = res["theta"].reshape(-1, len(names))
    return {n: th[:, names.index(n)] for n, *_ in COS if n in names}


def prior_draws(n=400_000, seed=0):
    # draws from the flat priors actually used: ranges from the prior file, w0 + wa from COSMO_REPARAM
    rng = np.random.default_rng(seed)
    box = {}
    for ln in open(os.path.join(capdir(MAG_CAP), f"{CLUSTER}_prior.dat")):
        t = ln.split()
        if len(t) == 4 and t[0] == "range" and t[1] in ("omega", "weos", "wa"):
            box[t[1]] = (float(t[2]), float(t[3]))
    out = {"omega": rng.uniform(*box["omega"], n), "weos": rng.uniform(*box["weos"], n)}
    spec = COSMO_REPARAM.get("wa")
    out["wa"] = (rng.uniform(spec[2], spec[3], n) - out["weos"]) if spec and spec[0] == "offset" \
        else rng.uniform(*box["wa"], n)
    return out


def shade(color):
    # translucent fills for corner's three bands: outside 95% (clear), the 95% ring, the 68% region
    r, g, b, _ = matplotlib.colors.to_rgba(color)
    return [(r, g, b, 0.0), (r, g, b, 0.18), (r, g, b, 0.45)]


def interval(v):
    q = np.percentile(v, [16, 50, 84])
    return f"{q[1]:6.2f} [{q[0]:5.2f}, {q[2]:5.2f}]"


if nuts:
    cols = {k: cosmo_columns(r) for k, r in nuts.items()}
    prior = prior_draws()
    K = len(COS)
    fig, axs = plt.subplots(K, K, figsize=(3.2 * K, 3.2 * K))
    for i, (ni, li, _, ri) in enumerate(COS):
        for j, (nj, lj, _, rj) in enumerate(COS):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(prior[ni], bins=40, range=ri, density=True, histtype="step", color="0.5", ls="--")
            for k, cc in cols.items():
                if i == j and ni in cc:
                    ax.hist(cc[ni], bins=40, range=ri, density=True, histtype="stepfilled", color=COLORS[k],
                            alpha=0.25)
                    ax.hist(cc[ni], bins=40, range=ri, density=True, histtype="step", color=COLORS[k], lw=1.5)
                elif i != j and ni in cc and nj in cc:
                    corner.hist2d(cc[nj], cc[ni], ax=ax, range=[rj, ri], bins=40, smooth=1.0, color=COLORS[k],
                                  levels=(0.68, 0.95), plot_datapoints=False, plot_density=False,
                                  no_fill_contours=True, fill_contours=True,
                                  contourf_kwargs=dict(colors=shade(COLORS[k])))
            ax.axvline(REF[nj], color="0.5", ls=":", lw=0.8)
            ax.set_xlim(rj)
            if i == j:
                ax.set_yticks([])
            else:
                ax.axhline(REF[ni], color="0.5", ls=":", lw=0.8)
                ax.set_ylim(ri)
            if i == K - 1:
                ax.set_xlabel(lj, fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(li, fontsize=12)
            elif i != j:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], color=COLORS[k], lw=1.5) for k in cols] + [plt.Line2D([], [], color="0.5", ls="--")]
    fig.legend(handles, [COSMO_MODELS[k][1] for k in cols] + ["prior"], loc="upper right",
               bbox_to_anchor=(0.95, 0.95), fontsize=12, title=CLUSTER)
    plt.tight_layout(); plt.show()

    print(f"median [68%]   {'':12s}" + "".join(f"{lab:>24s}" for _, _, lab, _ in COS))
    for k, cc in cols.items():
        print(f"{COSMO_MODELS[k][1]:12s} posterior " + "".join(
            f"{interval(cc[n]) if n in cc else '':>24s}" for n, *_ in COS))
    print(f"{'':12s} prior     " + "".join(f"{interval(prior[n]):>24s}" for n, *_ in COS))
    print("\nIntervals close to the prior's are prior-dominated, not constraints.")
""")

code(r"""
# lens parameters in each cosmology: freeing the cosmology widens the ones degenerate with distances
if nuts:
    base = list(next(iter(nuts.values()))["names"])
    lens_names = [n for n in base if not is_source(n) and n not in ("omega", "weos", "wa")]
    print(f"{'parameter':22s}" + "".join(f"{COSMO_MODELS[k][1]:>26s}" for k in nuts))
    for n in lens_names:
        row = ""
        for res in nuts.values():
            names = list(res["names"]); v = res["theta"].reshape(-1, len(names))[:, names.index(n)]
            row += f"{v.mean():14.4g} ± {v.std():9.3g}"
        print(f"{label(n):22s}{row}")
""")

md(r"""
### Posterior-predictive check with exact image finding

For random posterior draws of the most general model sampled, set the
sources to their best positions and solve the lens equation with glafic's
exact image finder. Count how many draws reproduce every observed image, and
compute the image-plane rms; systems that lose images are listed.
""")

code(r"""
if nuts:
    pp = next(k for k in ("w0wa", "wcdm", "lcdm") if k in nuts)
    mw = cosmo_model(MAG_CAP, pp)            # marginalised sources: matches the sampled theta
    names = list(nuts[pp]["names"]); thw = nuts[pp]["theta"].reshape(-1, len(names))
    rng = np.random.default_rng(1)
    n_ok, rms, lost = 0, [], {}
    nd = 5 if SMOKE_TEST else 20
    for t in thw[rng.choice(len(thw), nd, replace=False)]:
        Pd = mw.best_sources(mw.unflatten(jnp.asarray(t)))
        good, d2 = True, []
        for i in range(mw.num_poi):
            obs = mw.obs_point[i]["tab"]; im = mw.findimg(i + 1, P=Pd)
            if im.shape[0] < obs.shape[0]:
                good = False; lost[SIDS[i]] = lost.get(SIDS[i], 0) + 1
            if len(im):
                d2 += [((im[:, :2] - p[:2]) ** 2).sum(1).min() for p in obs]
        n_ok += good; rms.append(np.sqrt(np.mean(d2)))
    print(f"{COSMO_MODELS[pp][1]}: {n_ok}/{nd} draws form at least the observed number of images in every system; "
          f"image-plane rms = {np.mean(rms):.3f} ± {np.std(rms):.3f} arcsec")
    if lost:
        print("draws losing images, by system:", ", ".join(f"{s}: {c}" for s, c in lost.items()))
""")

md(r"""
## Notes

* **Running several clusters.** One kernel per cluster: set `GLAFIC_CLUSTER`
  in each batch job. Results go to separate folders, and the w0waCDM chain is
  shared with `cluster_cosmo_hmc.ipynb` when the model setup is the same.
* **Interpreting the cosmology.** A single cluster constrains ratios of
  distances between source planes, which depend weakly on (Ω_m, w0, wa) and
  are degenerate with the mass model. Compare each posterior with its prior
  in the corner plot; clusters with many spectroscopic systems over a wide
  redshift range constrain it best.
* **Validation.** This notebook has no gradient-free cross-check and no
  sampler-diagnostics section; use `cluster_cosmo_hmc.ipynb` for those.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(HERE, "cluster_cosmo_models.ipynb")
nbf.write(nb, out)
print("wrote", out)
