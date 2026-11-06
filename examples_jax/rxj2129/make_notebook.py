"""Generate rxj2129_hmc.ipynb (run once; then execute with jupyter nbconvert --execute)."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# RXJ2129 with `glafic_jax`: MAP → SVI → HMC, including w0waCDM cosmology

This notebook models the strong-lensing cluster **RXJ2129** (z = 0.234) from the
catalogues in `example_data/`:

* 38 multiple images of 14 source systems, 0.68 < z_s < 9.5. Three systems
  have photometric redshifts and are sampled with Gaussian priors.
* 144 cluster members with luminosity, ellipticity and position angle.

**Lens model** (glafic models):

| component | glafic model | free parameters |
|---|---|---|
| dark-matter halo | `anfw` (elliptical NFW, CSE approximation) | M, x, y, e, θ, c |
| cluster members | `gals` (pseudo-Jaffe, σ ∝ L^1/4, r_t ∝ L^1/2) | σ*, r_t* |
| external shear | `pert` | γ, θ_γ |
| cosmology | flat **w0waCDM**: w(a) = w0 + wa(1−a), Ω_Λ = 1 − Ω_m | Ω_m, w0, wa |
| source redshifts | systems 12–14 | z_s |

Strong lensing depends on cosmology only through distance ratios
D_ls/D_s. Having many sources at different redshifts behind the same lens is
what makes cluster *cosmography* possible.

**Likelihood.** We use glafic's source-plane χ² (`chi2_splane 1`), which
weights positions by the magnification tensor, so it approximates the
image-plane χ² to first order. The 14 source positions enter quadratically and
are **marginalised analytically** (`source_mode="marginalize"`). This leaves
16 sampled parameters. The full image-plane χ², with glafic's image finder,
is then used as a check of the final model.

**Inference pipeline** (`glafic_jax.Pipeline`):

1. **MAP**: multi-start L-BFGS with exact JAX gradients. We first fix the
   cosmology to ΛCDM to find the lens configuration, then free (Ω_m, w0, wa).
2. **SVI**: a full-rank Gaussian variational posterior, initialised at the MAP.
3. **HMC**: NUTS initialised from SVI draws, using the SVI covariance as the
   starting mass matrix.
""")

code(r"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.abspath(".")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
try:
    import glafic_jax as gj
except ImportError:                       # not pip-installed: use the repository copy
    sys.path.insert(0, os.path.join(REPO, "python"))
    import glafic_jax as gj
import jax, jax.numpy as jnp
import build_rxj2129 as build

print("jax", jax.__version__, "| devices:", jax.devices())
""")

md(r"""
## 1. Data → glafic input files

`build_rxj2129.py` converts RA/Dec to glafic's arcsec coordinates. The origin is
the BCG and x is positive to the west, as in glafic's `coord2xy`. It writes a
normal glafic input file, an observation file, a galaxy file and a prior file,
so the same model can also be run with C glafic.
""")

code(r"""
systems = build.write_all(HERE)
print(open("rxj2129.input").read())
print(open("rxj2129_prior.dat").read())
""")

code(r"""
gals = np.loadtxt("rxj2129_galfile.dat")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(gals[:, 0], gals[:, 1], s=200 * gals[:, 2] ** 0.5, c="0.7", label="cluster members (size ∝ √L)")
cmap = plt.get_cmap("tab20")
for k, (sid, s) in enumerate(systems.items()):
    xy = np.array(s["xy"])
    ax.scatter(xy[:, 0], xy[:, 1], s=40, color=cmap(k % 20), edgecolor="k", zorder=3,
               label=f"{sid}: z={s['zs']:.2f}" + (f"±{s['zerr']:.1f}" if s['zerr'] > 0 else ""))
ax.set_xlim(40, -40); ax.set_ylim(-40, 40); ax.set_aspect("equal")
ax.set_xlabel("x [arcsec] (west →)"); ax.set_ylabel("y [arcsec]")
ax.legend(fontsize=7, ncol=2, loc="lower left"); ax.set_title("RXJ2129: multiple images and members")
plt.show()
""")

md(r"""
## 2. Build the model

Constraints and choices:

* **Cosmology priors** are flat inside the ranges in `rxj2129_prior.dat`:
  Ω_m ∈ [0.05, 1], w0 ∈ [−2.5, 0], wa ∈ [−3, 2]. Flatness is imposed with
  `flatfix 1`.
* **w0 + wa < 0** is added through `extra_log_prior`, so that dark energy is
  subdominant at high redshift. Without it, the z = 9.5 source would sit in a
  universe dominated by dark energy.
* **h** does not affect image positions and is fixed (`hvary 0`).
""")

code(r"""
model = gj.read_input("rxj2129.input")
model.extra_log_prior = lambda P: jnp.where(P["cosmo"][2] + P["cosmo"][4] < 0.0, 0.0, -jnp.inf)
model.set_source_mode("marginalize")
print(model.ndim, "sampled parameters:")
print(model.theta_names)
""")

md(r"""
## 3. MAP

**Stage 1** fixes ΛCDM (Ω_m = 0.3, w0 = −1, wa = 0) and fits the lens from 12
jittered starting points. The source positions are *profiled* here (set to
their χ²-minimising values), which is the MAP-equivalent of marginalising.
The χ² surface of a cluster model is multimodal, so several starts are needed;
compiled gradients make each start only a few seconds.

**Stage 2** frees (Ω_m, w0, wa) from the stage-1 solution.
""")

code(r"""
for k in ("ovary", "wvary", "wavary"):
    model.set_secondary(f"{k} 0")
model.set_source_mode("profile")
t0 = time.time()
map1 = gj.Pipeline(model).map(n_starts=12, jitter=0.5, seed=0)
model.update_from(model.unflatten(jnp.asarray(map1["theta"])))
chi2_lcdm = map1["chi2"]

for k in ("ovary", "wvary", "wavary"):
    model.set_secondary(f"{k} 1")
model.set_source_mode("profile")
map2 = gj.Pipeline(model).map(n_starts=1)
model.update_from(model.best_sources(model.unflatten(jnp.asarray(map2["theta"]))))
model.model_init()
print(f"\nMAP: chi2(LCDM) = {chi2_lcdm:.3f},  chi2(w0waCDM) = {map2['chi2']:.3f}   [{time.time()-t0:.0f} s total]")
nimg = sum(o["tab"].shape[0] for o in model.obs_point)
print(f"{nimg} images -> {2*nimg} constraints; "
      f"free: {model.ndim} lens/cosmology/z + {2*model.num_poi} source coordinates")
for n, v in zip(model.theta_names, np.asarray(model.theta0())):
    print(f"   {n:16s} {v: .5g}")
""")

md(r"""
### Check in the image plane

We solve the lens equation with glafic's adaptive-mesh image finder for every
system at the MAP, and compare the predicted images with the observed ones.
""")

code(r"""
P_map = model.params()
rms2, nmatch = [], 0
fig, ax = plt.subplots(figsize=(7, 7))
# critical curves for z_s = 1.5 and 9.5 (|1/mu| -> 0)
g = np.linspace(-45, 45, 451)
X, Y = np.meshgrid(g, g)
for zs, col in ((1.5, "tab:blue"), (9.51, "tab:red")):
    mu = jax.jit(lambda x, y: model.lensmodel(x, y, zs, P_map).muinv)(jnp.asarray(X), jnp.asarray(Y))
    ax.contour(X, Y, np.asarray(mu), levels=[0.0], colors=col, linewidths=1)
    ax.plot([], [], color=col, label=f"critical curve z_s={zs}")
for i in range(model.num_poi):
    obs = model.obs_point[i]["tab"]
    pred = model.findimg(i + 1)
    ax.scatter(obs[:, 0], obs[:, 1], marker="o", facecolor="none", edgecolor="k", s=60)
    ax.scatter(pred[:, 0], pred[:, 1], marker="x", color=cmap(i % 20), s=40)
    for o in obs:   # nearest predicted image
        d2 = ((pred[:, :2] - o[:2]) ** 2).sum(1)
        rms2.append(d2.min()); nmatch += 1
    print(f"system {i+1:2d}: {obs.shape[0]} observed, {pred.shape[0]} predicted images; "
          f"offsets [arcsec] = {np.sqrt([((pred[:, :2]-o[:2])**2).sum(1).min() for o in obs]).round(2)}")
ax.scatter([], [], marker="o", facecolor="none", edgecolor="k", label="observed")
ax.scatter([], [], marker="x", color="k", label="predicted (MAP)")
ax.set_xlim(45, -45); ax.set_ylim(-45, 45); ax.set_aspect("equal"); ax.legend(fontsize=8)
ax.set_xlabel("x [arcsec]"); ax.set_ylabel("y [arcsec]")
plt.show()
print(f"image-plane rms (nearest predicted image) = {np.sqrt(np.mean(rms2)):.2f} arcsec")
""")

md(r"""
For almost all systems the model predicts exactly the observed number of
images, with small image-plane offsets. Where it predicts an extra image,
the extra one is the faint central or counter image that usually goes
unobserved. The one clear failure is image 3.4, which this three-component
model does not produce. A model with a separate BCG component, or
multipoles, would be the natural next step. Everything below uses the
source-plane likelihood.

## 4. SVI

We fit a full-rank Gaussian guide in NumPyro's unconstrained space (every
parameter is mapped from its prior interval to ℝ), initialised at the MAP.
""")

code(r"""
model.set_source_mode("marginalize")
pipe = gj.Pipeline(model)
pipe.results["map"] = dict(theta=np.asarray(model.theta0()), chi2=map2["chi2"], names=model.theta_names)
svi = pipe.svi(num_steps=4000, guide="mvn", lr=5e-3, num_particles=4, num_draws=4000)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(svi["losses"]); ax.set_yscale("symlog"); ax.set_xlabel("step"); ax.set_ylabel("−ELBO")
plt.show()
""")

md(r"""
## 5. HMC (NUTS)

Two chains start from independent SVI draws. The initial dense mass matrix is
the SVI covariance, and is still adapted during warm-up.
""")

code(r"""
hmc = pipe.hmc(num_warmup=500, num_samples=1500, num_chains=2, target_accept=0.85, progress_bar=False)
d = gj.diagnostics(hmc)
print(f"{'parameter':16s} {'mean':>11s} {'std':>10s} {'R-hat':>7s} {'ESS':>7s}")
th = hmc["theta"].reshape(-1, model.ndim)
for k, n in enumerate(d["names"]):
    print(f"{n:16s} {th[:, k].mean():11.4g} {th[:, k].std():10.3g} {d['rhat'][k]:7.3f} {d['ess'][k]:7.0f}")
""")

md(r"""
## 6. Cosmology posterior

The corner plot compares HMC (black) with the SVI approximation (orange).
Dashed lines mark ΛCDM.
""")

code(r"""
import corner
names = model.theta_names
sel = [names.index(n) for n in ("omega", "weos", "wa")]
labels = [r"$\Omega_m$", r"$w_0$", r"$w_a$"]
fig = corner.corner(svi["theta"][:, sel], labels=labels, color="tab:orange", plot_datapoints=False,
                    plot_density=False, levels=(0.68, 0.95), hist_kwargs=dict(density=True))
corner.corner(th[:, sel], fig=fig, labels=labels, color="k", plot_datapoints=False, levels=(0.68, 0.95),
              truths=[0.3, -1.0, 0.0], truth_color="tab:blue", hist_kwargs=dict(density=True))
plt.show()
for k, n in zip(sel, labels):
    q = np.percentile(th[:, k], [2.5, 16, 50, 84, 97.5])
    print(f"{n:12s} median {q[2]: .3f}   68%: [{q[1]: .3f}, {q[3]: .3f}]   95%: [{q[0]: .3f}, {q[4]: .3f}]")
""")

code(r"""
lens_sel = [names.index(n) for n in ("lens1_anfw_p2", "lens1_anfw_p7", "lens2_gals_p2", "lens2_gals_p7", "omega", "weos")]
lab = [r"$M_{\rm halo}$", r"$c$", r"$\sigma_*$", r"$r_{t,*}$", r"$\Omega_m$", r"$w_0$"]
fig = corner.corner(th[:, lens_sel], labels=lab, color="k", plot_datapoints=False, levels=(0.68, 0.95))
plt.show()
""")

md(r"""
## 7. Discussion

* **Pipeline.** The MAP stage finds the lens configuration, SVI gives a
  calibrated Gaussian approximation in seconds, and NUTS refines it. The SVI
  mass matrix means NUTS needs only a short warm-up.
* **Cosmology.** With a single cluster and free (Ω_m, w0, wa), the posterior
  is broad and strongly degenerate with the halo concentration and the
  member-galaxy scaling. The data constrain a combination of distance ratios
  rather than the individual parameters. Compare the HMC contours with the
  prior box: where they reach its edges, the result is prior-dominated.
  Tighter constraints need more spectroscopic systems, a more flexible mass
  model (which inflates the uncertainty further), or several clusters
  combined.
* **Likelihood.** For the final analysis of a system that the model
  reproduces in the image plane, switch to the image-plane χ²:
  `model.set_secondary("chi2_splane 0")` with
  `gj.Pipeline(model, point_mode="tracked")` (pure JAX) or `"glafic"` (exact
  image finding, slower). Use `source_mode="sample"` there.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(HERE, "rxj2129_hmc.ipynb")
nbf.write(nb, out)
print("wrote", out)
