"""Generate rxj2129_cosmo_hmc.ipynb.

    python make_notebook.py

The notebook is designed to run on a GPU node (e.g. NERSC Perlmutter): every
stage caches its result in results/ and the samplers write checkpoints, so it
can be run piecewise across batch jobs with time limits.
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
# RXJ2129: strong-lensing cosmography with HMC and `glafic_jax`

This notebook models the cluster **RXC J2129.7+0005** (z = 0.234). It constrains
flat **w0waCDM** (Ω_m, w0, wa) jointly with the lens mass model, using
Hamiltonian Monte Carlo (NUTS) with exact JAX gradients.

**Data** (`example_data/`): 38 multiple images of 14 systems with
0.68 < z_s < 9.5. Three systems have photometric redshifts, which are sampled.
There are also 144 cluster members.

**Mass model.** The glafic recipe of Oguri et al. (VENUS, 2026, Sec. 2.3):

* an elliptical NFW halo, with all 6 parameters free;
* pseudo-Jaffe members with σ ∝ L^1/4 and r_trun ∝ L^η, where σ*, r_trun* and
  η are free;
* external shear;
* an m = 3 multipole.

Without the multipole, this recipe reproduces the published fit (χ² 21.96 for
37 degrees of freedom, against the published 21.99/37), but it fails to form
the image pair 3.2/3.3. The m = 3 term fixes that: all 38 images are
reproduced.

### Likelihood

We use glafic's **source-plane χ²** (`chi2_splane 1`), the choice of most
published cluster models, including the paper:

* Each image's offset is mapped back to the source plane with the
  magnification tensor of the lens model.
* The source positions are **integrated out analytically**:
  χ² → χ²_min + ln det M. This removes 28 nuisance parameters.
* A **magnification cap** (`mag_cap`) bounds the weight of strongly magnified
  images. The weight 1/μ⁻¹ is replaced by μ⁻¹/(μ⁻¹² + ε²) with ε = 1/(2 μ_cap),
  so |μ| ≤ μ_cap. Several images (systems 3, 6, 11) lie almost on critical
  curves, where the uncapped χ² has spikes that HMC cannot traverse. Without
  the cap, test chains did not mix (R-hat 1.8). With μ_cap = 10 they are clean.
* Positional errors are **rescaled for each cap**, so the ΛCDM best fit has
  χ²/dof = 1, the convention of the paper.

The cap changes the likelihood, so the cosmology is re-run with a second
value (stage E) to check that the result does not depend on the choice.

### Stages

Each stage caches its result in `results/`, and re-running skips finished
stages.

| stage | what | cost (laptop CPU) |
|---|---|---|
| A | ΛCDM best fit for each cap; error calibration; image check with glafic's exact image finder | ~5 min per cap |
| B | ΛCDM NUTS | ~1–2 h |
| C | ΛCDM AIES (gradient-free) to validate NUTS | ~10 min |
| D | w0waCDM NUTS | ~2–3 h |
| E | w0waCDM NUTS at a second magnification cap | ~2–3 h |
| F | cosmology results, cap comparison, posterior-predictive check | minutes |

The `RUN_*` flags in the next cell run only some stages, for example one per
batch job. The samplers write a checkpoint every `chunk` samples, so a job
that hits its time limit resumes where it stopped.
""")

code(r"""
import os, sys, json, time
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
import build_rxj2129 as build

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
print("jax", jax.__version__, "| backend:", jax.default_backend(), "| devices:", jax.devices())

# ---------------- configuration ----------------
MPOLES = [(3, 2)]                 # (m, n) multipoles centred on the BCG
MAG_CAP = 10.0                    # main analysis
CAP_CHECK = 30.0                  # sensitivity check (stage E); None to skip
POS_ERR0 = 0.4                    # catalogue positional error [arcsec], before rescaling
CALIBRATE_ERRORS = True           # rescale errors so that the LCDM best fit has chi2/dof = 1
REPARAM = {"lens1_anfw_p2": "log10",          # halo mass: flat in log10 M
           "lens2_gals_p7": "log10"}          # member truncation radius: flat in log10
COSMO_REPARAM = {"wa": ("offset", "weos", -3.0, 0.0)}   # sample w0 + wa in [-3, 0)
NUTS_KW = dict(num_chains=4, num_warmup=1000, num_samples=1000, target_accept=0.9,
               max_tree_depth=8, init="map", init_jitter=0.005, use_approx_mass=False, chunk=250)
AIES_KW = dict(num_warmup=4000, num_samples=4000, thinning=4, init_jitter=0.005, chunk=1000)
RUN_A, RUN_B, RUN_C, RUN_D, RUN_E = True, True, True, True, True
SMOKE_TEST = False                # tiny runs, to check that the notebook works end to end

if SMOKE_TEST:
    RESULTS = os.path.join(HERE, "results_smoke"); os.makedirs(RESULTS, exist_ok=True)
    NUTS_KW.update(num_chains=2, num_warmup=20, num_samples=20, chunk=10, max_tree_depth=5)
    AIES_KW.update(num_warmup=100, num_samples=100, thinning=1, chunk=50)

LABELS = {"lens1_anfw_p2": "M_vir [M_sun/h]", "lens1_anfw_p3": "x_halo", "lens1_anfw_p4": "y_halo",
          "lens1_anfw_p5": "e_halo", "lens1_anfw_p6": "θ_halo", "lens1_anfw_p7": "c_halo",
          "lens2_gals_p2": "σ* [km/s]", "lens2_gals_p7": "r_trun* [arcsec]", "lens2_gals_p8": "η",
          "lens3_pert_p5": "γ_ext", "lens3_pert_p6": "θ_γ", "lens4_mpole_p5": "ε_3", "lens4_mpole_p6": "θ_3",
          "omega": "Ω_m", "weos": "w0", "wa": "wa"}
""")

md("## 1. Data")

code(r"""
systems = build.write_all(RESULTS, mpoles=MPOLES, pos_err=POS_ERR0)
print(open(os.path.join(RESULTS, "rxj2129.input")).read().split("start_command")[0])

gals = np.loadtxt(os.path.join(RESULTS, "rxj2129_galfile.dat"))
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(gals[:, 0], gals[:, 1], s=150 * gals[:, 2] ** 0.5, c="0.75", label="members (size ∝ √L)")
for k, (sid, s) in enumerate(systems.items()):
    xy = np.array(s["xy"])
    ax.scatter(xy[:, 0], xy[:, 1], s=35, color=cmap(k % 20), edgecolor="k", zorder=3,
               label=f"{sid}: z={s['zs']:.2f}" + (f"±{s['zerr']:.1f}" if s['zerr'] > 0 else ""))
ax.set_xlim(35, -35); ax.set_ylim(-35, 35); ax.set_aspect("equal")
ax.set_xlabel("x [arcsec] (west →)"); ax.set_ylabel("y [arcsec]"); ax.legend(fontsize=7, ncol=2)
plt.show()
""")

code(r"""
def capdir(cap):
    d = os.path.join(RESULTS, f"cap{cap:g}")
    os.makedirs(d, exist_ok=True)
    return d


def make_model(cap, cosmology_free=False, lens=None, poi=None):
    m = gj.read_input(os.path.join(capdir(cap), "rxj2129.input"))
    for k in ("ovary", "wvary", "wavary"):
        m.set_secondary(f"{k} {1 if cosmology_free else 0}")
    m.set_secondary(f"mag_cap {cap}")
    if lens is not None:
        m.lens_params = np.array(lens); m.poi_params = np.array(poi)
    m.model_init()
    return m


def dof_of(m):
    nimg = sum(o["tab"].shape[0] for o in m.obs_point)
    nlens = sum(n.startswith("lens") for n in m.theta_names)
    return 2 * nimg - 2 * m.num_poi - nlens


def save_npz(path, **kw):
    np.savez(path, **kw)


def load_npz(path):
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None
""")

md(r"""
## 2. Stage A: ΛCDM best fit and error calibration

For each magnification cap:

1. **Best fit.** Minimise the source-plane χ² over the lens parameters and
   photometric redshifts, with source positions profiled, from 16 jittered
   starts.
2. **Error calibration.** All images share one positional error σ, and the
   χ² scales as 1/σ². Setting σ = σ₀ √(χ²/dof) makes χ²/dof = 1 exactly,
   without changing the best fit.
""")

code(r"""
def stage_a(cap, start=None, n_starts=16):
    f = os.path.join(capdir(cap), "calib.json")
    if os.path.exists(f):
        return json.load(open(f))
    build.write_all(capdir(cap), mpoles=MPOLES, pos_err=POS_ERR0)
    m = make_model(cap, **(start or {}))
    m.set_source_mode("profile")
    r = gj.Pipeline(m, reparam=REPARAM).map(n_starts=n_starts, jitter=0.5, seed=0, verbose=False)
    P = m.best_sources(m.unflatten(jnp.asarray(r["theta"])))
    dof = dof_of(m)
    sig = POS_ERR0 * np.sqrt(r["chi2"] / dof) if CALIBRATE_ERRORS else POS_ERR0
    build.write_all(capdir(cap), mpoles=MPOLES, pos_err=sig)
    out = dict(cap=cap, sigma=float(sig), chi2_raw=float(r["chi2"]), dof=int(dof),
               lens=np.asarray(P["lens"]).tolist(), poi=np.asarray(P["point"]).tolist())
    json.dump(out, open(f, "w"))
    return out


calib = {}
if RUN_A:
    calib[MAG_CAP] = stage_a(MAG_CAP)
    if CAP_CHECK:
        c0 = calib[MAG_CAP]
        calib[CAP_CHECK] = stage_a(CAP_CHECK, start=dict(lens=c0["lens"], poi=c0["poi"]), n_starts=4)
for cap, c in calib.items():
    m = make_model(cap, lens=c["lens"], poi=c["poi"]); m.set_source_mode("profile")
    c2 = float(m.chi2(m.params()))
    print(f"cap {cap:g}: chi2 = {c['chi2_raw']:.2f} with σ = {POS_ERR0}\"  ->  σ = {c['sigma']:.3f}\", "
          f"chi2 = {c2:.2f} for {c['dof']} dof")
""")

md(r"""
### Check with glafic's exact image finder

The source-plane χ² does not check which images the model actually forms.
For the main best fit, we solve the lens equation with glafic's adaptive-mesh
image finder. We then compare the predicted and observed images, and
overlay the critical curves.
""")

code(r"""
c = calib[MAG_CAP]
m = make_model(MAG_CAP, lens=c["lens"], poi=c["poi"])
P = m.params()
fig, ax = plt.subplots(figsize=(6.5, 6.5))
g = np.linspace(-35, 35, 501); X, Y = np.meshgrid(g, g)
for zs, col in ((1.52, "tab:blue"), (3.08, "tab:green"), (9.51, "tab:red")):
    mu = jax.jit(lambda x, y: m.lensmodel(x, y, zs, P).muinv)(jnp.asarray(X), jnp.asarray(Y))
    ax.contour(X, Y, np.asarray(mu), levels=[0.0], colors=col, linewidths=0.8)
    ax.plot([], [], color=col, label=f"critical curve z_s={zs}")
offs = []
for i in range(m.num_poi):
    obs = m.obs_point[i]["tab"]; pred = m.findimg(i + 1)
    o = [np.sqrt(((pred[:, :2] - p[:2]) ** 2).sum(1).min()) for p in obs]; offs += o
    print(f"system {i+1:2d} (z={P['point'][i,0]:.3f}): {obs.shape[0]} observed / {pred.shape[0]} predicted, "
          f"offsets {np.round(o, 2)}")
    ax.scatter(obs[:, 0], obs[:, 1], marker="o", facecolor="none", edgecolor="k", s=45)
    ax.scatter(pred[:, 0], pred[:, 1], marker="x", color=cmap(i % 20), s=30)
ax.set_xlim(35, -35); ax.set_ylim(-35, 35); ax.set_aspect("equal"); ax.legend(fontsize=8)
plt.show()
print(f"image-plane rms = {np.sqrt(np.mean(np.square(offs))):.3f} arcsec")
""")

md(r"""
## 3. Stage B: ΛCDM NUTS

The sampled parameters are the 13 lens parameters and 3 photometric
redshifts; source positions are marginalised. Each orientation angle is
restricted to one period around the best fit (`gj.angle_bounds`), which
removes the exact θ ↔ θ + 180° copies of the posterior. Chains start near the
best fit, and the dense mass matrix is adapted during warmup.

Healthy output: no or very few divergences, R-hat < 1.01–1.02, ESS of a few
hundred or more.
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
    print(f"{'parameter':18s} {'mean':>11s} {'std':>10s} {'R-hat':>7s} {'ESS':>7s}")
    for k in keep:
        n = names[k]
        print(f"{LABELS.get(n, n):18s} {th[:, k].mean():11.4g} {th[:, k].std():10.3g} "
              f"{res['rhat'][k]:7.3f} {res['ess'][k]:7.0f}")


def lcdm_model(cap):
    c = calib[cap]
    mm = make_model(cap, lens=c["lens"], poi=c["poi"])
    mm.set_source_mode("marginalize")
    return mm


nuts_lcdm = None
if RUN_B:
    ml = lcdm_model(MAG_CAP)
    nuts_lcdm = nuts_stage(os.path.join(capdir(MAG_CAP), "nuts_lcdm.npz"), ml, ml.theta0(), REPARAM)
else:
    nuts_lcdm = load_npz(os.path.join(capdir(MAG_CAP), "nuts_lcdm.npz"))
if nuts_lcdm is not None:
    show_diag(nuts_lcdm)
""")

md(r"""
## 4. Stage C: validating NUTS with a gradient-free sampler

AIES (the affine-invariant ensemble sampler behind `emcee`) moves 64 walkers
using the positions of the others, with no gradients. It is far less
efficient than NUTS in high dimensions, but it cannot be misled by a
non-smooth posterior. It samples the *same* posterior: same
parameterization, priors and angle windows. If the marginals agree within
sampling noise, NUTS is trustworthy for this likelihood.
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
    print(f"{'parameter':18s} {'NUTS mean':>11s} {'AIES mean':>11s} {'NUTS sd':>9s} {'AIES sd':>9s} "
          f"{'Δmean/sd':>9s} {'sd ratio':>8s}")
    flag = []
    for k, n in enumerate(aies["names"]):
        if is_source(n):
            continue
        dm = (h[:, k].mean() - a[:, k].mean()) / h[:, k].std()
        sr = a[:, k].std() / h[:, k].std()
        # sampling noise is dominated by AIES's small ESS
        tol_m = 3 * np.sqrt(1 / ess_a[k] + 1 / nuts_lcdm["ess"][k])
        tol_s = 3 * np.sqrt(0.5 / ess_a[k] + 0.5 / nuts_lcdm["ess"][k])
        if abs(dm) > tol_m or abs(sr - 1) > tol_s:
            flag.append(n)
        print(f"{LABELS.get(n, n):18s} {h[:,k].mean():11.4g} {a[:,k].mean():11.4g} {h[:,k].std():9.3g} "
              f"{a[:,k].std():9.3g} {dm:9.2f} {sr:8.2f}")
    print(f"\nAIES ESS {ess_a.min():.0f}-{ess_a.max():.0f}.  "
          f"Parameters disagreeing beyond 3σ sampling noise: {flag or 'none'}")
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

md(r"""
## 6. Stage E: sensitivity to the magnification cap

Repeat stage D with a different cap (`CAP_CHECK`), with errors recalibrated
for that cap. A larger cap is closer to glafic's uncapped χ², but brings
back some of the spikes near critical curves, so check its diagnostics
carefully.
""")

code(r"""
nuts_w2 = w0wa_stage(CAP_CHECK, RUN_E) if CAP_CHECK else None
if nuts_w2 is not None:
    show_diag(nuts_w2)
""")

md("## 7. Stage F: cosmology")

code(r"""
import corner

COS = ("omega", "weos", "wa")
COS_LAB = [r"$\Omega_m$", r"$w_0$", r"$w_a$", r"$w_0+w_a$"]
RANGE = [(0.05, 1.0), (-2.5, 0.0), (-3.0, 3.0), (-3.0, 0.0)]


def cosmo_samples(res):
    names = list(res["names"]); th = res["theta"].reshape(-1, len(names))
    om, w0, wa = (th[:, names.index(n)] for n in COS)
    return np.column_stack([om, w0, wa, w0 + wa])


runs = [(f"μ_cap = {MAG_CAP:g}", nuts_w, "tab:blue"), (f"μ_cap = {CAP_CHECK}", nuts_w2, "tab:orange")]
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
        for cl, v in zip(COS_LAB, cosmo_samples(res).T):
            q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
            print(f"   {cl:12s} median {q[2]: .3f}   68%: [{q[1]: .3f}, {q[3]: .3f}]   95%: [{q[0]: .3f}, {q[4]: .3f}]")
    print("\nPrior box: Ω_m ∈ [0.05, 1], w0 ∈ [-2.5, 0], w0+wa ∈ [-3, 0).  Intervals that fill the box "
          "are prior-dominated, not constraints.")
""")

code(r"""
if nuts_w is not None and nuts_lcdm is not None:
    nl, nw = list(nuts_lcdm["names"]), list(nuts_w["names"])
    hl = nuts_lcdm["theta"].reshape(-1, len(nl)); hw = nuts_w["theta"].reshape(-1, len(nw))
    print(f"{'parameter':18s} {'ΛCDM':>24s} {'w0waCDM':>24s}")
    for n in nl:
        if is_source(n):
            continue
        a, b = hl[:, nl.index(n)], hw[:, nw.index(n)]
        print(f"{LABELS.get(n, n):18s} {a.mean():13.4g} ± {a.std():8.3g} {b.mean():13.4g} ± {b.std():8.3g}")
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
## Notes

* **Interpreting the cosmology.** A single cluster constrains ratios of
  distances, D_ls/D_s, between source redshifts. These ratios depend only
  weakly on (Ω_m, w0, wa) and are degenerate with the mass model. Compare
  the intervals with the prior box: where a posterior fills the box, the data
  have not constrained that parameter.
* **Assumptions worth testing:**
  * the mass-model family, for example adding an m = 4 multipole or a
    separate BCG (`build.write_all(mpoles=..., bcg=True)`);
  * the magnification cap (stage E);
  * one common positional error for all images;
  * the photometric-redshift ranges (±4σ, flat).
* **Validation.** Trust stages D and E only if stage C agrees and their own
  diagnostics are clean.
* **Performance.** NUTS cost is roughly (gradient evaluations per sample) ×
  (time per gradient), and `show_diag` prints the first. On a GPU, more
  chains (`num_chains`) cost little extra, because chains are vectorized.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(HERE, "rxj2129_cosmo_hmc.ipynb")
nbf.write(nb, out)
print("wrote", out)
