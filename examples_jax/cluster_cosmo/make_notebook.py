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
`python build_cluster.py` lists which clusters are excluded.
Rows are matched to folders by the table's `glafic_name` column (the folder
name); a new cluster needs a row there to be excluded.

To add a cluster, drop a new folder in `example_data/`. `build_cluster.py`
lists what it finds (`python build_cluster.py`).

**Mass model.** The glafic recipe of Oguri et al. (VENUS, 2026, Sec. 2.3):

* an elliptical NFW halo, with all 6 parameters free;
* pseudo-Jaffe members with σ ∝ L^1/4 and r_trun ∝ L^η, where σ*, r_trun* and
  η are free;
* optional external shear (`shear`, off by default);
* optional multipoles centred on the BCG (`mpoles`) and an optional separate
  BCG (`bcg`).

Every cluster runs with no per-cluster settings. The shear is off by
default because it is often consistent with zero. Its angle is then
unconstrained, and the amplitude–angle pair forms a funnel at γ = 0 that
NUTS samples poorly.

**Automatic model selection** (`AUTO_MODEL`, stage A). The base model is fitted
first and checked with glafic's exact image finder. It fails the check if a
system has fewer predicted than observed images, if an observed image lies more
than `IMG_OFFSET_MAX` σ from every predicted image, or if an observed image has
|μ| > `MU_CRIT` at the best fit (the source-plane χ² rewards putting an image on
a critical curve, where any offset maps to a tiny source-plane residual). If
the base model fails, each component in `MODEL_CANDIDATES` (m = 3 and m = 4
multipoles centred on the BCG, a separate BCG, external shear) is added in turn
and refitted. Among the candidates that pass, the one with the lowest BIC is
kept; if none passes, a second component is added to the closest one (at most
`MAX_ADDITIONS`). A cluster that still fails stops with an error and needs a
model by hand. With `BIC_SELECT = True`, a component is also added to a
passing model when it lowers the BIC by more than `BIC_MIN_DROP`. The χ² alone
is not enough: for RXJ2129 the base model's χ² is only 2.7 worse than with an
m = 3 multipole, but it puts image 6.2 on a critical curve and loses two images
of system 6. Setting `mpoles`, `bcg` or `shear` for a cluster in
`CLUSTER_MODEL` fixes its model by hand and skips the search.

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

Each candidate's fit is cached in `results/<cluster>/model_search/`, and the
decision in `results/<cluster>/model_choice.json`, which later runs with the
same settings reuse.

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

Everything for a cluster goes in one folder, `results/<cluster>/`: the glafic
input files, the cached result of each stage (re-running skips finished
stages) and every figure as a PNG. The folder holds one model setup at a time.
`setup.json` records its model tag, which changes with the multipoles, the
BCG, the shear, the parities, `SPEC_ONLY` and the halo-mass parameterization.
Running a different setup into the same folder stops with an error instead
of reusing stale results; move or rename the folder to keep both. Each cached
file also records its cluster and model tag.

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
import os, sys, re, json, time, math, shutil
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

# ---------------- configuration ----------------
CLUSTER = os.environ.get("GLAFIC_CLUSTER", "rj2129")   # folder name in example_data/
RUN_EXCLUDED = False              # allow a cluster marked in the summary table's 'exclude' column
if build.is_excluded(CLUSTER) and not RUN_EXCLUDED:   # marked in example_data/venus_clusters_summary.csv
    raise ValueError(f"{CLUSTER} is excluded in venus_clusters_summary.csv; set RUN_EXCLUDED = True to run it")
CLUSTER_MODEL = {                 # optional per-cluster settings; "mpoles", "bcg" or "shear" fix the model by hand
    # "<name>": dict(mpoles=[(3, 2)], bcg=True, shear=True, parity={"3.3": -1, "3.4": +1}),
}
MANUAL_MODEL = {k: v for k, v in CLUSTER_MODEL.get(CLUSTER, {}).items() if k in ("mpoles", "bcg", "shear")}
AUTO_MODEL = True                 # without a manual model: add components when the base model fails the image check
MODEL_CANDIDATES = [dict(mpoles=[(3, 2)]), dict(mpoles=[(4, 2)]), dict(bcg=True), dict(shear=True)]
MAX_ADDITIONS = 2                 # at most this many components are added to the base model
BIC_SELECT = False                # also add a component to a passing model if it lowers the BIC by > BIC_MIN_DROP
BIC_MIN_DROP = 10.0
IMG_OFFSET_MAX = 5.0              # image check fails if an observed image is > this many σ from every predicted one
MU_CRIT = 1000.0                  # ... or has |μ| above this at the best fit (it sits on a critical curve)
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
COSMO_REPARAM = {"wa": ("sum_below", "weos", -3.0, 2.0, 0.0)}   # DESI: wa in [-3, 2], w0 + wa < 0
W_PRIOR = "desi"                  # tag of the w0/wa prior in the chain file names (w0 in [-3, 1]: prior file)
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

# results: one folder per cluster, holding one model setup at a time
_PARITY_TAG = ["autopar"] if PARITY == "auto" else ["mupar" + "-".join(sorted(PARITY))] if PARITY else []
_NFW_TAG = [f"nfwa{NFW_ALPHA_R:g}z{NFW_ALPHA_ZS:g}" if USE_NFW_ALPHA else "log10M"]
SETTINGS_TAG = "_".join((["speconly"] if SPEC_ONLY else []) + _PARITY_TAG
                        + ([f"cap{MAG_CAP:g}"] if MAG_CAP else []) + _NFW_TAG)   # everything but the mass model


def model_tag(cfg):
    # name of a model setup: the mass model cfg (mpoles, bcg, shear) plus the settings above
    return "_".join(["base"] + (["speconly"] if SPEC_ONLY else []) + ([] if cfg["shear"] else ["noshear"])
                    + [f"mpole{m}{n}" for m, n in cfg["mpoles"]] + (["bcg"] if cfg["bcg"] else [])
                    + _PARITY_TAG + ([f"cap{MAG_CAP:g}"] if MAG_CAP else []) + _NFW_TAG)


RUN_DIR = RESULTS                 # every file of this cluster: results/<cluster>/
os.makedirs(RUN_DIR, exist_ok=True)
SETUP_FILE = os.path.join(RUN_DIR, "setup.json")
_old_setup = json.load(open(SETUP_FILE)) if os.path.exists(SETUP_FILE) else {}
if _old_setup.get("settings", SETTINGS_TAG) != SETTINGS_TAG:   # checked again with the model in stage A
    raise ValueError(f"{RUN_DIR} holds results of model {_old_setup.get('tag')!r}, not of settings "
                     f"{SETTINGS_TAG!r}; move or rename it (e.g. to {RESULTS}_{_old_setup.get('tag')}) first")


def show(name):
    # save the current figure as results/<cluster>/<name>.png, then display it
    plt.savefig(os.path.join(RUN_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def write_files(outdir, pos_err, cfg=None):
    # glafic files of mass model cfg (default: the selected MODEL_CFG); parities are not written to the obs
    # file (see add_parity_penalty)
    cfg = cfg or MODEL_CFG
    return build.write_all(CLUSTER, outdir, mpoles=cfg["mpoles"], bcg=cfg["bcg"], shear=cfg["shear"],
                           pos_err=pos_err, spec_only=SPEC_ONLY)


gals = cl["gals"]
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
show("images")
""")

code(r"""
def capdir(cap):
    return RUN_DIR                # one folder per cluster (a non-zero MAG_CAP is part of the model tag)


def make_model(cap, cosmology_free=False, lens=None, poi=None, d=None):
    m = gj.read_input(os.path.join(d or capdir(cap), f"{CLUSTER}.input"))
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
    # parities of the close pairs that straddle a critical curve (opposite signs of 1/μ) in model P,
    # and one line of explanation per close pair
    mi = muinv_obs(m, P)
    out, notes = {}, []
    for a, b in CLOSE_PAIRS:
        ia, ib = IMG_IDS.index(a), IMG_IDS.index(b)
        sa, sb = int(np.sign(mi[ia])), int(np.sign(mi[ib]))
        txt = f"   close pair {a}/{b}: 1/μ = {mi[ia]:+.4f} / {mi[ib]:+.4f}"
        if sa * sb < 0:
            out[a], out[b] = sa, sb
            notes.append(txt + "  ->  straddles a critical curve; parities fixed")
        else:
            notes.append(txt + "  ->  same parity, not constrained (if this is a merging pair, the best fit is "
                         "in the wrong mode: set its parity in CLUSTER_MODEL)")
    return out, notes


def image_check(m, P, sigma):
    # glafic's exact image finder on model P.  Per system: the predicted images, the offset of each observed
    # image from the nearest predicted one and |μ| at the observed positions.  'fails' lists what is wrong.
    rows, missing = [], []
    for i in range(m.num_poi):
        obs = m.obs_point[i]["tab"]; pred = m.findimg(i + 1, P=P)
        off = np.array([np.sqrt(((pred[:, :2] - p[:2]) ** 2).sum(1).min()) if len(pred) else np.inf for p in obs])
        mu = np.abs(1 / np.asarray(m.lensmodel(jnp.asarray(obs[:, 0]), jnp.asarray(obs[:, 1]),
                                               float(P["point"][i, 0]), P).muinv))
        rows.append(dict(sid=SIDS[i], zs=float(P["point"][i, 0]), obs=obs, pred=pred, off=off, mu=mu))
        if pred.shape[0] < obs.shape[0]:
            missing.append(SIDS[i])
    off = np.concatenate([r["off"] for r in rows]); mu = np.concatenate([r["mu"] for r in rows])
    jo, jm = int(np.argmax(off)), int(np.argmax(mu))
    fails = []
    if missing:
        fails.append(f"missing images in system(s) {', '.join(map(str, missing))}")
    if off[jo] > IMG_OFFSET_MAX * sigma:
        fails.append(f"image {IMG_IDS[jo]} is {off[jo]:.2f}\" from every predicted image (> {IMG_OFFSET_MAX:g}σ)")
    if mu[jm] > MU_CRIT:
        fails.append(f"image {IMG_IDS[jm]} has |μ| = {mu[jm]:.0f} (on a critical curve)")
    return dict(rows=rows, missing=missing, rms=float(np.sqrt(np.mean(off ** 2))), max_off=float(off[jo] / sigma),
                median_mu=float(np.median(mu)), fails=fails, ok=not fails)


def check_owner(d, path, tag=None):
    # every cached result records the cluster and model it belongs to; refuse to mix them up
    tag = TAG if tag is None else tag
    got = (str(d.get("cluster", CLUSTER)), str(d.get("tag", tag)))
    if got != (CLUSTER, tag):
        raise ValueError(f"{path} belongs to cluster {got[0]!r} / model {got[1]!r}, "
                         f"not {CLUSTER!r} / {tag!r}; move or delete it")
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


def label(n):
    n = str(n)
    m = re.fullmatch(r"lens(\d+)_([a-z0-9]+)_p(\d+)", n)
    if m:
        i, mod, p = int(m.group(1)), m.group(2), int(m.group(3))
        if mod == "mpole" and p in (5, 6):
            mm = MPOLES[i - (2 + int(SHEAR) + int(BCG)) - 1][0]     # lenses: anfw, gals, [pert], [jaffe], mpoles
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
4. **Model selection** (`AUTO_MODEL`; see the introduction). Steps 1–3 run for
   the base model and, if it fails the image check, for each candidate
   component; the selected model's fit becomes the stage-A result.

If the readme gives a published fit, its χ² (at the catalogue errors) is
printed for comparison. The two match only when the model and the number of
free parameters match the paper's.
""")

code(r"""
def fit_setup(d, cfg, tag, n_starts=N_STARTS):
    # stage A for mass model cfg in folder d (cached in d/calib.json): best fit at fixed cosmology,
    # error calibration, parities
    f = os.path.join(d, "calib.json")
    if os.path.exists(f):
        return check_owner(json.load(open(f)), f, tag)
    os.makedirs(d, exist_ok=True)
    write_files(d, POS_ERR0, cfg)
    m = make_model(MAG_CAP, d=d)                # with PARITY = "auto", no parity constraint yet
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
    # parities: read off the best fit (zero penalty there) or set by hand
    parity, notes = auto_parity(m, P) if PARITY == "auto" else (dict(PARITY), [])
    sig = POS_ERR0 * np.sqrt(r["chi2"] / dof) if CALIBRATE_ERRORS else POS_ERR0
    write_files(d, sig, cfg)
    nobs = 2 * sum(o["tab"].shape[0] for o in m.obs_point)
    npar = sum(n.startswith("lens") or n.endswith("_zs") for n in m.theta_names)
    out = dict(cluster=CLUSTER, tag=tag, model=cfg, cap=MAG_CAP, sigma=float(sig), chi2_raw=float(r["chi2"]),
               dof=int(dof), nobs=int(nobs), npar=int(npar), lens=np.asarray(P["lens"]).tolist(),
               poi=np.asarray(P["point"]).tolist(), parity=parity, parity_notes=notes)
    json.dump(out, open(f, "w"))
    return out


def merged(cfg, add):
    # mass model cfg with the components of add
    out = dict(mpoles=[tuple(p) for p in cfg.get("mpoles", [])], bcg=bool(cfg.get("bcg", False)),
               shear=bool(cfg.get("shear", False)))
    out["mpoles"] += [tuple(p) for p in add.get("mpoles", []) if tuple(p) not in out["mpoles"]]
    out["bcg"] |= bool(add.get("bcg", False)); out["shear"] |= bool(add.get("shear", False))
    return out


def evaluate(cfg):
    # fit candidate cfg in results/<cluster>/model_search/<tag>/ (cached) and run the image check
    tag = model_tag(cfg); d = os.path.join(RUN_DIR, "model_search", tag)
    t0 = time.time()
    c = fit_setup(d, cfg, tag)
    m = make_model(MAG_CAP, lens=c["lens"], poi=c["poi"], d=d)
    chk = image_check(m, m.params(), c["sigma"])
    e = dict(cfg=cfg, tag=tag, dir=d, calib=c, ok=chk["ok"], fails=chk["fails"], missing=len(chk["missing"]),
             max_off=chk["max_off"])
    print(f"   {tag:55s} chi2 {c['chi2_raw']:8.2f}  {c['npar']:3d} par  BIC {bic(e):8.2f}  "
          + ("passes" if e["ok"] else "FAILS: " + "; ".join(e["fails"])) + f"  ({time.time() - t0:.0f} s)")
    return e


def bic(e):
    # BIC with the chi2 at a common positional error (the base model's calibrated sigma)
    c = e["calib"]
    return c["chi2_raw"] * (POS_ERR0 / SIGMA_REF) ** 2 + c["npar"] * np.log(c["nobs"])


BASE_MODEL = dict(mpoles=[], bcg=False, shear=False)
_search_key = dict(settings=SETTINGS_TAG, candidates=[merged(BASE_MODEL, a) for a in MODEL_CANDIDATES],
                   max_additions=MAX_ADDITIONS, bic_select=BIC_SELECT, bic_min_drop=BIC_MIN_DROP,
                   img_offset_max=IMG_OFFSET_MAX, mu_crit=MU_CRIT)
CHOICE_FILE = os.path.join(RUN_DIR, "model_choice.json")
t0 = time.time()
if MANUAL_MODEL or not AUTO_MODEL:
    MODEL_CFG = merged(BASE_MODEL, MANUAL_MODEL)
    print("mass model " + ("set by hand in CLUSTER_MODEL" if MANUAL_MODEL else "= base model (AUTO_MODEL off)"))
elif os.path.exists(CHOICE_FILE) and json.load(open(CHOICE_FILE)).get("key") == json.loads(json.dumps(_search_key)):
    MODEL_CFG = merged(BASE_MODEL, json.load(open(CHOICE_FILE))["model"])
    print(f"mass model: {model_tag(MODEL_CFG)} (selected earlier; see {CHOICE_FILE})")
else:
    print("model selection (image check, then BIC):")
    _base = model_tag(BASE_MODEL)       # its calibrated σ is the common error for the BIC comparison
    SIGMA_REF = fit_setup(os.path.join(RUN_DIR, "model_search", _base), BASE_MODEL, _base)["sigma"]
    cur = evaluate(BASE_MODEL); tried = [cur]
    for _ in range(MAX_ADDITIONS):
        if cur["ok"] and not BIC_SELECT:
            break
        trials = [evaluate(merged(cur["cfg"], a)) for a in MODEL_CANDIDATES if merged(cur["cfg"], a) != cur["cfg"]]
        tried += trials
        passing = [t for t in trials if t["ok"]]
        if not cur["ok"]:            # repair: the best passing candidate, else the closest one gets another component
            cur = min(passing, key=bic) if passing else min(trials, key=lambda t: (t["missing"], t["max_off"]))
        else:                        # BIC_SELECT: keep a component only for a clear BIC gain
            best = min(passing, key=bic, default=None)
            if best is None or bic(best) > bic(cur) - BIC_MIN_DROP:
                break
            cur = best
    table = [dict(tag=t["tag"], ok=t["ok"], fails=t["fails"], chi2=t["calib"]["chi2_raw"], npar=t["calib"]["npar"],
                  bic=bic(t)) for t in tried]
    json.dump(dict(key=_search_key, model=cur["cfg"] if cur["ok"] else None, tag=cur["tag"], ok=cur["ok"],
                   candidates=table), open(CHOICE_FILE, "w"), indent=1)
    if not cur["ok"]:
        raise RuntimeError(f"{CLUSTER}: no model with up to {MAX_ADDITIONS} added components passes the image check "
                           f"(closest: {cur['tag']}: {'; '.join(cur['fails'])}). Set the model by hand in "
                           f"CLUSTER_MODEL, or exclude the cluster.")
    MODEL_CFG = cur["cfg"]
    if cur["dir"] != RUN_DIR and not os.path.exists(os.path.join(RUN_DIR, "calib.json")):
        shutil.copy(os.path.join(cur["dir"], "calib.json"), os.path.join(RUN_DIR, "calib.json"))
    print(f"selected: {cur['tag']}  (model selection {time.time() - t0:.0f} s)")

MPOLES, BCG, SHEAR = MODEL_CFG["mpoles"], MODEL_CFG["bcg"], MODEL_CFG["shear"]
TAG = model_tag(MODEL_CFG)
if _old_setup.get("tag", TAG) != TAG:      # one model setup per folder: never mix results of different setups
    raise ValueError(f"{RUN_DIR} holds results of model {_old_setup['tag']!r}, not {TAG!r}; move or rename it "
                     f"(e.g. to {RESULTS}_{_old_setup['tag']}) before running this setup")
json.dump(dict(cluster=CLUSTER, tag=TAG, settings=SETTINGS_TAG, model=MODEL_CFG), open(SETUP_FILE, "w"))
print(f"model: {TAG}  ->  {RUN_DIR}")
write_files(RUN_DIR, POS_ERR0)
print(open(os.path.join(RUN_DIR, f"{CLUSTER}.input")).read().split("start_command")[0])
""")

code(r"""
t0 = time.time()
calib = {MAG_CAP: fit_setup(RUN_DIR, MODEL_CFG, TAG)}   # cached (or copied from the model search)
print(f"stage A: {time.time() - t0:.0f} s")
PARITY_FIX = {k: int(v) for k, v in calib[MAG_CAP]["parity"].items()}   # in force from here on
write_files(capdir(MAG_CAP), calib[MAG_CAP]["sigma"])   # current priors/settings, also for a cached stage A
for ln in calib[MAG_CAP].get("parity_notes", []):
    print(ln)
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
chk = image_check(m, P, c["sigma"])
for i, r in enumerate(chk["rows"]):
    print(f"system {str(r['sid']):>4s} (z={r['zs']:.3f}): {r['obs'].shape[0]} observed / {r['pred'].shape[0]} "
          f"predicted, offsets {np.round(r['off'], 2)}, |μ| {np.round(r['mu'], 1)}")
    ax.scatter(r["obs"][:, 0], r["obs"][:, 1], marker="o", facecolor="none", edgecolor="k", s=45)
    ax.scatter(r["pred"][:, 0], r["pred"][:, 1], marker="x", color=cmap(i % 20), s=30)
ax.set_xlim(LIM, -LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal"); ax.legend(fontsize=8)
ax.set_title(f"{CLUSTER}: ΛCDM best fit")
show("best_fit")
print(f"image-plane rms = {chk['rms']:.3f} arcsec (expect ~{c['sigma']:.2f}); median |μ| = {chk['median_mu']:.1f}; "
      f"systems with missing images: {chk['missing'] or 'none'}")
if not chk["ok"]:
    print("WARNING: the best fit fails the image check (" + "; ".join(chk["fails"]) + "). Do not run the "
          "samplers from it; try more N_STARTS, or another model in CLUSTER_MODEL (multipoles, bcg, shear).")
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


def save_txt(fname, txt):
    # print a text summary and save it as results/<cluster>/<fname>
    print(txt)
    open(os.path.join(RUN_DIR, fname), "w").write(txt + "\n")


def show_diag(res, title, fname):
    # sampler summary (time, divergences, R-hat, ESS, posterior mean and std); also saved to results/<cluster>/<fname>
    names = list(res["names"]); keep = [k for k, n in enumerate(names) if not is_source(n)]
    th = res["theta"].reshape(-1, len(names))
    lines = [f"=== {CLUSTER}: {title}   (model {TAG})",
             f"{float(res['time'])/60:.0f} min; divergences {int(res['diverging'].sum())} / {res['diverging'].size}; "
             f"max R-hat {np.nanmax(res['rhat'][keep]):.3f}; min ESS {np.nanmin(res['ess'][keep]):.0f}; "
             f"median {np.median(res['num_steps']):.0f} gradient evaluations per sample",
             f"{'parameter':22s} {'mean':>11s} {'std':>10s} {'R-hat':>7s} {'ESS':>7s}"]
    for k in keep:
        n = names[k]
        lines.append(f"{label(n):22s} {th[:, k].mean():11.4g} {th[:, k].std():10.3g} "
                     f"{res['rhat'][k]:7.3f} {res['ess'][k]:7.0f}")
    save_txt(fname, "\n".join(lines))


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
    show_diag(nuts_lcdm, "ΛCDM NUTS (fixed cosmology)", "nuts_lcdm.txt")
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
* w0 ∈ [−3, 1] and wa ∈ [−3, 2], with **w0 + wa < 0** (DESI's convention).

The condition w0 + wa < 0 (dark energy subdominant at early times) cuts the
(w0, wa) box diagonally. wa is sampled as the fraction of its allowed range
at the current w0 (`sum_below` in `COSMO_REPARAM`), with the Jacobian
included, so the prior is exactly flat in (w0, wa) and there is no wall in
the posterior. The chains start at the ΛCDM best fit, with
(Ω_m, w0, wa) = (0.3, −1, 0).
""")

code(r"""
def w0wa_stage(cap, run):
    path = os.path.join(capdir(cap), f"nuts_w0wa_{W_PRIOR}.npz")
    if not run or os.path.exists(path):
        return load_npz(path)
    c = calib[cap]
    mw = make_model(cap, cosmology_free=True, lens=c["lens"], poi=c["poi"])
    mw.set_source_mode("marginalize")
    return nuts_stage(path, mw, mw.theta0(), dict(REPARAM, **COSMO_REPARAM))


nuts_w = w0wa_stage(MAG_CAP, RUN_D)
if nuts_w is not None:
    show_diag(nuts_w, "w0waCDM NUTS", f"nuts_w0wa_{W_PRIOR}.txt")
""")

md("## 6. Stage E: cosmology")

code(r"""
import corner

COS = ("omega", "weos", "wa")
COS_LAB = [r"$\Omega_m$", r"$w_0$", r"$w_a$", r"$w_0+w_a$"]
COS_TXT = ["Ω_m", "w0", "wa", "w0+wa"]
RANGE = [(0.05, 1.0), (-3.0, 1.0), (-3.0, 2.0), (-6.0, 0.0)]


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
    show("w0wa_corner")
    lines = [f"=== {CLUSTER}: w0waCDM cosmology   (model {TAG})"]
    for lab, res, _ in runs:
        lines.append(lab)
        for cl_, v in zip(COS_TXT, cosmo_samples(res).T):
            q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
            lines.append(f"   {cl_:6s} median {q[2]: .3f}   68%: [{q[1]: .3f}, {q[3]: .3f}]   "
                         f"95%: [{q[0]: .3f}, {q[4]: .3f}]")
    lines.append("\nPrior (DESI): Ω_m ∈ [0.05, 1], w0 ∈ [-3, 1], wa ∈ [-3, 2], w0+wa < 0.  Intervals that fill the "
                 "box are prior-dominated, not constraints.")
    save_txt("w0wa_cosmology.txt", "\n".join(lines))
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
    show("lens_corner")
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


def sampler_diagnostics(title, res, model, fname):
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
    plt.tight_layout(); show(fname)
    return dict(muinv=muinv, near=near, worst_image=IMG_IDS[jw])


diag = {}
if nuts_lcdm is not None:
    diag["ΛCDM"] = sampler_diagnostics("ΛCDM NUTS", nuts_lcdm, lcdm_model(MAG_CAP), "diagnostics_lcdm")
if nuts_w is not None:
    c = calib[MAG_CAP]
    mw = make_model(MAG_CAP, cosmology_free=True, lens=c["lens"], poi=c["poi"])
    mw.set_source_mode("marginalize")
    diag["w0waCDM"] = sampler_diagnostics("w0waCDM NUTS", nuts_w, mw, "diagnostics_w0wa")
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
