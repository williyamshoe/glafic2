"""Generate cluster_cosmo_hmc.ipynb.

    python make_notebook.py

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
# Strong-lensing cosmography of a galaxy cluster with HMC and `glafic_jax`

This notebook models one cluster from `example_data/` and constrains flat
**w0waCDM** (Ω_m, w0, wa) jointly with the lens mass model, using
Hamiltonian Monte Carlo (NUTS) with exact JAX gradients.

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
essentially random sign. Section 7 checks that the samples respect the
constraints.

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
* Positional errors are **rescaled** so that the ΛCDM best fit has
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

| stage | what | cost (laptop CPU) |
|---|---|---|
| A | ΛCDM MAP; error calibration; image check with glafic's exact image finder | ~5 min |
| B | ΛCDM NUTS | hours |
| C | ΛCDM AIES (gradient-free) to validate NUTS | ~30 min |
| D | w0waCDM NUTS | hours |
| E | cosmology results and posterior-predictive check | minutes |

The `RUN_*` flags in the next cell run only some stages, for example one per
batch job; `RUN_A` alone checks the MAP in a few minutes. The samplers write
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
CALIBRATE_ERRORS = True           # rescale errors so that the LCDM best fit has chi2/dof = 1
N_STARTS = 16                     # jittered L-BFGS starts for the MAP
USE_NFW_ALPHA = True              # halo mass: sample log10 of the NFW deflection at R (straightens M-c)
NFW_ALPHA_R = None                # [arcsec]; None = median image distance from the BCG
NFW_ALPHA_ZS = None               # source z for the deflection; None = median z_s
COSMO_REPARAM = {"wa": ("offset", "weos", -3.0, 0.0)}   # sample w0 + wa in [-3, 0)
NUTS_KW = dict(num_chains=4, num_warmup=1000, num_samples=1000, target_accept=0.95,
               max_tree_depth=10, init="map", init_jitter=0.01, use_approx_mass=False, chunk=250)
AIES_KW = dict(num_warmup=5000, num_samples=5000, thinning=5, init_jitter=0.005, chunk=1000)
RUN_A, RUN_B, RUN_C, RUN_D = True, True, True, True
SMOKE_TEST = False                # tiny runs, to check that the notebook works end to end

RESULTS = os.path.join(HERE, "results_smoke" if SMOKE_TEST else "results", CLUSTER)
if SMOKE_TEST:
    NUTS_KW.update(num_chains=2, num_warmup=20, num_samples=20, chunk=10, max_tree_depth=5)
    AIES_KW.update(num_warmup=100, num_samples=100, thinning=1, chunk=50)
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


def make_model(cap, cosmology_free=False, lens=None, poi=None):
    m = gj.read_input(os.path.join(capdir(cap), f"{CLUSTER}.input"))
    for k in ("ovary", "wvary", "wavary"):
        m.set_secondary(f"{k} {1 if cosmology_free else 0}")
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
## 2. Stage A: ΛCDM best fit and error calibration

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
ax.set_title(f"{CLUSTER}: ΛCDM best fit")
plt.show()
rms = np.sqrt(np.mean(np.square(offs)))
print(f"image-plane rms = {rms:.3f} arcsec (expect ~{c['sigma']:.2f}); median |μ| = {np.median(mus):.1f}; "
      f"systems with missing images: {missing or 'none'}")
if rms > 1.0 or missing:
    print("WARNING: the MAP does not reproduce the observed images. Do not run the samplers from it; "
          "try more N_STARTS, a different seed, or a richer model in CLUSTER_MODEL (multipoles, bcg).")
""")

md(r"""
## 3. Stage B: ΛCDM NUTS

The sampled parameters are the free lens parameters and the photometric
redshifts; source positions are marginalised. Each orientation angle is
restricted to one period around the best fit (`gj.angle_bounds`), which
removes the exact θ ↔ θ + 180° copies of the posterior. Chains start near the
best fit, and the dense mass matrix is adapted during warmup.

Healthy output: no or very few divergences, R-hat < 1.01–1.02, ESS of a few
hundred or more. If the median number of gradient evaluations per sample is
2^max_tree_depth − 1, NUTS is running out of tree depth: raise
`max_tree_depth`, or check the parameterization.
""")

code(r"""
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


def lcdm_model(cap):
    c = calib[cap]
    mm = make_model(cap, lens=c["lens"], poi=c["poi"])
    mm.set_source_mode("marginalize")
    return mm


p_nuts = os.path.join(capdir(MAG_CAP), "nuts_lcdm.npz")
nuts_lcdm = load_npz(p_nuts)
if nuts_lcdm is None and RUN_B:
    ml = lcdm_model(MAG_CAP)
    nl = sum(n.startswith("lens") for n in ml.theta_names)
    print(f"{ml.ndim} sampled parameters: {nl} lens, {ml.ndim - nl} photometric redshifts")
    nuts_lcdm = nuts_stage(p_nuts, ml, ml.theta0(), REPARAM)
if nuts_lcdm is not None:
    show_diag(nuts_lcdm)
""")

md(r"""
## 4. Stage C: validating NUTS with a gradient-free sampler

AIES (the affine-invariant ensemble sampler behind `emcee`) moves its walkers
using the positions of the others, with no gradients. It is far less
efficient than NUTS in high dimensions, but it cannot be misled by a
non-smooth posterior. It samples the *same* posterior: same
parameterization, priors and angle windows. If the marginals agree within
sampling noise, NUTS is trustworthy for this likelihood.

The comparison is only meaningful when both samplers have a usable ESS: with
an ESS of a few, the tolerance is so wide that anything passes.
""")

code(r"""
p_aies = os.path.join(capdir(MAG_CAP), "aies_lcdm.npz")
aies = load_npz(p_aies)
if aies is None and RUN_C:
    ml = lcdm_model(MAG_CAP); th0 = ml.theta0()
    pipe = gj.Pipeline(ml, reparam=REPARAM, bounds=gj.angle_bounds(ml, th0))
    pipe.set_map(th0)
    e = pipe.ensemble(checkpoint=p_aies.replace(".npz", ".ckpt"), **AIES_KW)
    save_npz(p_aies, theta=e["theta"], names=np.array(e["names"]))
    aies = load_npz(p_aies)

if aies is not None and nuts_lcdm is not None:
    from numpyro.diagnostics import effective_sample_size
    a3 = aies["theta"]; a = a3.reshape(-1, a3.shape[-1]); h = nuts_lcdm["theta"].reshape(-1, a.shape[-1])
    ess_a = np.asarray(effective_sample_size(a3))
    print(f"{'parameter':22s} {'NUTS mean':>11s} {'AIES mean':>11s} {'NUTS sd':>9s} {'AIES sd':>9s} "
          f"{'Δmean/sd':>9s} {'sd ratio':>8s}")
    flag, weak = [], []
    for k, n in enumerate(aies["names"]):
        if is_source(n):
            continue
        dm = (h[:, k].mean() - a[:, k].mean()) / h[:, k].std()
        sr = a[:, k].std() / h[:, k].std()
        tol_m = 3 * np.sqrt(1 / ess_a[k] + 1 / nuts_lcdm["ess"][k])
        tol_s = 3 * np.sqrt(0.5 / ess_a[k] + 0.5 / nuts_lcdm["ess"][k])
        if abs(dm) > tol_m or abs(sr - 1) > tol_s:
            flag.append(label(n))
        if min(ess_a[k], nuts_lcdm["ess"][k]) < 50:
            weak.append(label(n))
        print(f"{label(n):22s} {h[:,k].mean():11.4g} {a[:,k].mean():11.4g} {h[:,k].std():9.3g} "
              f"{a[:,k].std():9.3g} {dm:9.2f} {sr:8.2f}")
    print(f"\nAIES ESS {ess_a.min():.0f}-{ess_a.max():.0f}.  "
          f"Parameters disagreeing beyond 3σ sampling noise: {flag or 'none'}")
    if weak:
        print(f"Not a real test (ESS < 50 in one sampler): {weak}")
""")

md(r"""
## 5. Stage D: flat w0waCDM

Ω_m, w0 and wa are sampled jointly with the mass model, with flat priors on:

* Ω_m ∈ [0.05, 1];
* w0 ∈ [−2.5, 0];
* **w0 + wa ∈ [−3, 0)**.

Sampling w0 + wa directly builds the condition w0 + wa < 0 (dark energy
subdominant at early times) into the parameter space, instead of a hard
wall in the posterior. The chains start at the ΛCDM best fit, with
(Ω_m, w0, wa) = (0.3, −1, 0).
""")

code(r"""
def w0wa_stage(cap, run):
    path = os.path.join(capdir(cap), "nuts_w0wa.npz")
    if not run or os.path.exists(path):
        return load_npz(path)
    c = calib[cap]
    mw = make_model(cap, cosmology_free=True, lens=c["lens"], poi=c["poi"])
    mw.set_source_mode("marginalize")
    return nuts_stage(path, mw, mw.theta0(), dict(REPARAM, **COSMO_REPARAM))


nuts_w = w0wa_stage(MAG_CAP, RUN_D)
if nuts_w is not None:
    show_diag(nuts_w)
""")

md("## 6. Stage E: cosmology")

code(r"""
import corner

COS = ("omega", "weos", "wa")
COS_LAB = [r"$\Omega_m$", r"$w_0$", r"$w_a$", r"$w_0+w_a$"]
COS_TXT = ["Ω_m", "w0", "wa", "w0+wa"]
RANGE = [(0.05, 1.0), (-2.5, 0.0), (-3.0, 3.0), (-3.0, 0.0)]


def cosmo_samples(res):
    names = list(res["names"]); th = res["theta"].reshape(-1, len(names))
    om, w0, wa = (th[:, names.index(n)] for n in COS)
    return np.column_stack([om, w0, wa, w0 + wa])


runs = [(f"{CLUSTER} w0waCDM", nuts_w, "tab:blue")]   # add more (label, result, colour) to compare runs
runs = [r for r in runs if r[1] is not None]
if runs:
    fig = None
    for lab, res, col in runs:
        fig = corner.corner(cosmo_samples(res), fig=fig, labels=COS_LAB, range=RANGE, color=col,
                            levels=(0.68, 0.95), plot_datapoints=False, plot_density=False,
                            hist_kwargs=dict(density=True))
    corner.overplot_lines(fig, [0.3, -1.0, 0.0, -1.0], color="0.5", ls=":")
    fig.legend(handles=[plt.Line2D([], [], color=c) for _, _, c in runs], labels=[l for l, _, _ in runs],
               loc="upper right")
    plt.show()
    for lab, res, _ in runs:
        print(lab)
        for cl_, v in zip(COS_TXT, cosmo_samples(res).T):
            q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
            print(f"   {cl_:6s} median {q[2]: .3f}   68%: [{q[1]: .3f}, {q[3]: .3f}]   95%: [{q[0]: .3f}, {q[4]: .3f}]")
    print("\nPrior box: Ω_m ∈ [0.05, 1], w0 ∈ [-2.5, 0], w0+wa ∈ [-3, 0).  Intervals that fill the box "
          "are prior-dominated, not constraints.")
""")

code(r"""
if nuts_w is not None and nuts_lcdm is not None:
    nl, nw = list(nuts_lcdm["names"]), list(nuts_w["names"])
    hl = nuts_lcdm["theta"].reshape(-1, len(nl)); hw = nuts_w["theta"].reshape(-1, len(nw))
    print(f"{'parameter':22s} {'ΛCDM':>24s} {'w0waCDM':>24s}")
    for n in nl:
        if is_source(n):
            continue
        a, b = hl[:, nl.index(n)], hw[:, nw.index(n)]
        print(f"{label(n):22s} {a.mean():13.4g} ± {a.std():8.3g} {b.mean():13.4g} ± {b.std():8.3g}")
""")

md(r"""
### ΛCDM corner plot: lens parameters and photometric redshifts

ΛCDM has no free cosmological parameters, so this shows the lens model: the
NFW halo, the member scaling relation and the photometric redshifts. The
w0waCDM posterior of the same parameters is overplotted. Where the two agree,
freeing the cosmology does not change the lens model. M_vir is in units of
10¹⁴ M_sun/h, and r_trun* is shown as log10, the way it is sampled.
""")

code(r"""
import corner


def lens_samples(res, names_ref):
    # columns of names_ref (lens + photometric z) in plot units, from a NUTS result
    names = list(res["names"]); th = res["theta"].reshape(-1, len(names))
    cols, labs = [], []
    for n in names_ref:
        v, lab = th[:, names.index(n)], label(n)
        if n.endswith("anfw_p2"):
            v, lab = v / 1e14, "M_vir [1e14 M_sun/h]"
        elif n.endswith("gals_p7"):
            v, lab = np.log10(v), "log10 r_trun* [arcsec]"
        cols.append(v); labs.append(lab)
    return np.column_stack(cols), labs


if nuts_lcdm is not None:
    LENS_NAMES = [n for n in nuts_lcdm["names"] if not is_source(n)]
    xl, labs = lens_samples(nuts_lcdm, LENS_NAMES)
    runs = [(f"{CLUSTER} ΛCDM", xl, "tab:red")]
    if nuts_w is not None:
        runs.append((f"{CLUSTER} w0waCDM", lens_samples(nuts_w, LENS_NAMES)[0], "tab:blue"))
    rng_ = [tuple(np.percentile(np.concatenate([x[:, k] for _, x, _ in runs]), [0.5, 99.5]))
            for k in range(xl.shape[1])]
    fig = None
    for lab, x, col in runs:
        fig = corner.corner(x, fig=fig, labels=labs, range=rng_, color=col, levels=(0.68, 0.95),
                            plot_datapoints=False, plot_density=False, hist_kwargs=dict(density=True),
                            label_kwargs=dict(fontsize=9))
    fig.legend(handles=[plt.Line2D([], [], color=c) for _, _, c in runs], labels=[l for l, _, _ in runs],
               loc="upper right", fontsize=12)
    plt.show()
    print(f"{CLUSTER} ΛCDM")
    for lab, v in zip(labs, xl.T):
        q = np.percentile(v, [16, 50, 84])
        print(f"   {lab:24s} {q[1]:10.4g}  +{q[2] - q[1]:.3g} / -{q[1] - q[0]:.3g}")
""")

md(r"""
### Posterior-predictive check with exact image finding

For random w0waCDM posterior draws, set the sources to their best positions
and solve the lens equation with glafic's exact image finder. Count how many
observed images each draw reproduces, and compute the image-plane rms.
""")

code(r"""
if nuts_w is not None:
    c = calib[MAG_CAP]
    mw = make_model(MAG_CAP, cosmology_free=True, lens=c["lens"], poi=c["poi"])
    mw.set_source_mode("marginalize")     # match the sampled theta (no source positions)
    names = list(nuts_w["names"]); thw = nuts_w["theta"].reshape(-1, len(names))
    rng = np.random.default_rng(1)
    n_ok, rms = 0, []
    nd = 5 if SMOKE_TEST else 20
    for t in thw[rng.choice(len(thw), nd, replace=False)]:
        Pd = mw.best_sources(mw.unflatten(jnp.asarray(t)))
        good, d2 = True, []
        for i in range(mw.num_poi):
            obs = mw.obs_point[i]["tab"]; im = mw.findimg(i + 1, P=Pd)
            good &= im.shape[0] >= obs.shape[0]
            if len(im):
                d2 += [((im[:, :2] - p[:2]) ** 2).sum(1).min() for p in obs]
        n_ok += good; rms.append(np.sqrt(np.mean(d2)))
    print(f"{n_ok}/{nd} draws form at least the observed number of images in every system;  "
          f"image-plane rms = {np.mean(rms):.3f} ± {np.std(rms):.3f} arcsec")
""")

md(r"""
## 7. Sampler diagnostics: divergences and near-critical images

Reads the saved chains; nothing is resampled. For every posterior sample, it
computes the magnification of each *observed* image position. An image whose
|μ| reaches the hundreds sits on a critical curve, where the source-plane χ²
is extremely sharp. If the divergences pile up where that image's |μ| blows
up, that image is what limits NUTS.

* **Table:** images whose |μ| exceeds `MU_FLAG` in more than 5% of samples.
  "parity flips" is the fraction of samples in which the image is on the other
  side of the critical curve from the majority.
* **Divergence test:** the divergence rate among the 10% of samples with the
  largest |μ| of the worst image, compared with the rest.
* **Close pairs:** every pair of images in the same system closer than
  `PAIR_SEP`. A pair merging across a critical curve must have opposite
  parities; samples where they share a sign are an unphysical mode. If one
  image of the pair keeps its parity, the suggested `parity` setting for
  `CLUSTER_MODEL` is printed. Images whose parity is fixed (by hand or
  automatically) are checked too: their violation fraction should be 0%.
* **Plots:** M_vir vs e_halo, and e_halo vs the signed 1/μ of the worst image
  (the critical curve is 1/μ = 0), with divergent samples in red; then the
  e_halo trace for each chain.
""")

code(r"""
MU_FLAG = 100.0                   # |μ| above which an observed image counts as near-critical
# IMG_IDS, CLOSE_PAIRS (pairs closer than PAIR_SEP) and PARITY_FIX are defined in sections 1-2


def mu_scan(res, model, batch=500):
    # signed 1/μ at every observed image position for every saved sample -> [n_samples, n_images]
    names = list(res["names"])
    if names != list(model.theta_names):
        raise ValueError("chain parameters do not match the model's")
    th = np.asarray(res["theta"]).reshape(-1, len(names))
    obs = [np.asarray(model.obs_point[i]["tab"]) for i in range(model.num_poi)]

    ls = model.lens_system

    def one(t):
        P = model.unflatten(t)
        # nlp (lens planes in front of the source) must be concrete under vmap: use the precomputed one
        return jnp.concatenate([ls.ray_shoot(jnp.asarray(o[:, 0]), jnp.asarray(o[:, 1]), P["lens"],
                                             P["point"][i, 0], P["cosmo"], nlp=int(model.nlp_point[i])).muinv
                                for i, o in enumerate(obs)])
    f = jax.jit(jax.vmap(one))
    return np.concatenate([np.asarray(f(jnp.asarray(th[k:k + batch]))) for k in range(0, len(th), batch)])


def sampler_diagnostics(title, res, model):
    names = list(res["names"]); nd = len(names)
    th3 = np.asarray(res["theta"]).reshape(-1, *np.shape(res["theta"])[-2:]) if np.ndim(res["theta"]) == 3 \
        else np.asarray(res["theta"])[None]
    th = th3.reshape(-1, nd)
    div = np.asarray(res["diverging"]).reshape(-1).astype(bool)
    t0 = time.time()
    muinv = mu_scan(res, model)
    amu = 1.0 / np.maximum(np.abs(muinv), 1e-12)
    print(f"=== {title}: {len(th)} samples, {div.sum()} divergent; |μ| scan {time.time() - t0:.0f} s")

    # near-critical images
    print(f"{'image':>6s} {'median |μ|':>11s} {'95% |μ|':>9s} {'f(|μ|>' + f'{MU_FLAG:g}' + ')':>11s} "
          f"{'parity flips':>13s} {'median |μ| (div)':>17s}")
    near = []
    for j, iid in enumerate(IMG_IDS):
        frac = np.mean(amu[:, j] > MU_FLAG)
        if frac < 0.05:
            continue
        near.append(iid)
        sgn = np.sign(muinv[:, j]); flips = min(np.mean(sgn > 0), np.mean(sgn < 0))
        mdiv = f"{np.median(amu[div, j]):17.1f}" if div.any() else f"{'-':>17s}"
        print(f"{iid:>6s} {np.median(amu[:, j]):11.1f} {np.percentile(amu[:, j], 95):9.1f} {frac:11.2f} "
              f"{flips:13.3f} {mdiv}")
    if not near:
        print(f"   no image has |μ| > {MU_FLAG:g} in more than 5% of samples")

    # close pairs: a pair merging across a critical curve must have opposite parity
    pos = lambda iid: np.mean(muinv[:, IMG_IDS.index(iid)] > 0)      # fraction with 1/μ > 0
    for a, b in CLOSE_PAIRS:
        same = np.mean(np.sign(muinv[:, IMG_IDS.index(a)]) == np.sign(muinv[:, IMG_IDS.index(b)]))
        msg = f"close pair {a}/{b}: 1/μ > 0 in {pos(a):.0%} / {pos(b):.0%} of samples; same parity in {same:.0%}"
        if same > 0.01:
            stable = [i for i in (a, b) if min(pos(i), 1 - pos(i)) < 0.01]
            if stable:
                sa = 1 if pos(stable[0]) > 0.5 else -1
                other = b if stable[0] == a else a
                msg += (f"  ->  unphysical mode; suggested: parity={{\"{stable[0]}\": {sa:+d}, "
                        f"\"{other}\": {-sa:+d}}}")
            else:
                msg += "  ->  both images flip; fix the signs from the best fit, or drop the pair"
        print(msg)
    for iid, p in PARITY_FIX.items():
        bad = np.mean(np.sign(muinv[:, IMG_IDS.index(iid)]) != p)
        print(f"parity constraint {iid} = {p:+d}: violated in {bad:.1%} of samples"
              + ("" if bad < 0.005 else "  <- constraint not respected"))

    # divergences vs the worst image's magnification
    jw = int(np.argmax(np.median(amu, axis=0)))
    if div.any():
        top = amu[:, jw] >= np.percentile(amu[:, jw], 90)
        print(f"divergence rate: {100 * div[top].mean():.1f}% in the 10% of samples with the largest |μ| of "
              f"image {IMG_IDS[jw]}, {100 * div[~top].mean():.1f}% in the rest")

    # chains: per-chain means of the worst-mixing parameters
    keep = [k for k, n in enumerate(names) if not is_source(n)]
    worst = sorted(keep, key=lambda k: -np.nan_to_num(res["rhat"][k]))[:4]
    print("per-chain means (worst R-hat):")
    for k in worst:
        print(f"   {label(names[k]):22s} R-hat {res['rhat'][k]:.3f}: "
              + "  ".join(f"{v:.4g}" for v in th3[:, :, k].mean(axis=1)))

    kM, ke = names.index("lens1_anfw_p2"), names.index("lens1_anfw_p5")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    ax.scatter(th[~div, kM], th[~div, ke], s=2, c="0.6", alpha=0.4, label="samples")
    ax.scatter(th[div, kM], th[div, ke], s=12, c="tab:red", label="divergent")
    ax.set_xlabel(label(names[kM])); ax.set_ylabel(label(names[ke])); ax.legend(fontsize=8)
    ax = axs[1]
    ax.scatter(muinv[~div, jw], th[~div, ke], s=2, c="0.6", alpha=0.4)
    ax.scatter(muinv[div, jw], th[div, ke], s=12, c="tab:red")
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(f"1/μ of image {IMG_IDS[jw]} (0 = critical curve)"); ax.set_ylabel(label(names[ke]))
    ax = axs[2]
    for ci in range(th3.shape[0]):
        ax.plot(th3[ci, :, ke], lw=0.5, label=f"chain {ci}")
    ax.set_xlabel("sample"); ax.set_ylabel(label(names[ke])); ax.legend(fontsize=7)
    fig.suptitle(f"{CLUSTER}: {title}")
    plt.tight_layout(); plt.show()
    return dict(muinv=muinv, near=near, worst_image=IMG_IDS[jw])


diag = {}
if nuts_lcdm is not None:
    diag["ΛCDM"] = sampler_diagnostics("ΛCDM NUTS", nuts_lcdm, lcdm_model(MAG_CAP))
if nuts_w is not None:
    c = calib[MAG_CAP]
    mw = make_model(MAG_CAP, cosmology_free=True, lens=c["lens"], poi=c["poi"])
    mw.set_source_mode("marginalize")
    diag["w0waCDM"] = sampler_diagnostics("w0waCDM NUTS", nuts_w, mw)
""")

md(r"""
## Notes

* **Running several clusters.** One kernel per cluster: set `GLAFIC_CLUSTER`
  in each batch job (for example `GLAFIC_CLUSTER=mj0429 jupyter nbconvert
  --execute ...`). Results go to separate folders.
* **Interpreting the cosmology.** A single cluster constrains ratios of
  distances, D_ls/D_s, between source redshifts. These ratios depend only
  weakly on (Ω_m, w0, wa) and are degenerate with the mass model. Compare
  the intervals with the prior box: where a posterior fills the box, the data
  have not constrained that parameter. Clusters with many spectroscopic
  systems over a wide redshift range constrain it best.
* **Assumptions worth testing:**
  * the mass-model family, for example adding multipoles or a separate BCG,
    or adding the external shear (`CLUSTER_MODEL`);
  * one common positional error for all images;
  * the photometric-redshift ranges (±4σ, flat).
* **Validation.** Trust stage D only if stage C agrees and its own
  diagnostics are clean.
* **Performance.** NUTS cost is roughly (gradient evaluations per sample) ×
  (time per gradient), and `show_diag` prints the first. On a GPU, more
  chains (`num_chains`) cost little extra, because chains are vectorized.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(HERE, "cluster_cosmo_hmc.ipynb")
nbf.write(nb, out)
print("wrote", out)
