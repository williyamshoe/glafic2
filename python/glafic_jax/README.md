# glafic_jax — a differentiable JAX port of glafic

`glafic_jax` re-implements the modelling core of glafic 2.1.15 in Python/JAX.
Every quantity that enters the χ² is differentiable with respect to all model
parameters, including lens, source, PSF, sky, source redshifts and cosmology.
That makes gradient-based optimisation and Hamiltonian Monte Carlo (NUTS)
possible. Results agree with the C code; see *Validation* below.

```
pip install ./python            # from the repository root; needs jax, numpy, scipy
pip install "./python[fits,hmc]"  # + astropy (FITS files), numpyro, blackjax
```

Double precision is switched on when the package is imported.

## Quick start

```python
import glafic_jax as gj

model = gj.read_input("samples/point.input")   # the same input files as glafic
model.set_secondary("hvary 0")
model.model_init()

theta, chi2, _ = gj.optimize(model)            # L-BFGS with exact gradients
res = gj.run_nuts(model, theta, num_warmup=500, num_samples=1000)
print(gj.summary(res))
```

You can also build a model with the same calls as the C python interface:

```python
m = gj.Model(gj.default_settings(xmin=-5, ymin=-5, xmax=5, ymax=5, pix_ext=0.02, pix_poi=0.5))
m.startup_setnum(2, 0, 1)
m.set_lens(1, "sie", 0.5, 300.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0)
m.set_lens(2, "pert", 0.5, 2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)
m.set_point(1, 2.0, -0.15, 0.05)
m.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
m.setopt_point(1, 0, 1, 1)
m.model_init()                 # call again after changing models, flags or redshifts
m.readobs_point("samples/obs_point.dat")
m.parprior("samples/prior_point.dat")
m.model_init()

m.calcimage(2.0, 0.5, 0.3)     # (ax, ay, td, kappa, gamma1, gamma2, 1/mu, rot)
m.point_solve(2.0, -0.15, 0.05)  # glafic findimg: rows of [x, y, mag, td]
```

### MAP → SVI → HMC pipeline

```python
model.set_source_mode("marginalize")         # analytic marginalisation of source positions
pipe = gj.Pipeline(model)                    # flat priors on the parprior ranges
pipe.map(n_starts=8, jitter=0.5)             # multi-start L-BFGS in unconstrained space
pipe.svi(num_steps=4000, guide="mvn")        # Gaussian variational posterior from the MAP
res = pipe.hmc(num_warmup=500, num_samples=1500, num_chains=2)   # NUTS, SVI mass matrix
gj.diagnostics(res)                          # R-hat, ESS
```

Each free parameter becomes a NumPyro site with a flat prior on its glafic
range. Ranges may be half-open or unbounded. All three stages work in
NumPyro's unconstrained space, so none of them hits a hard boundary.

* **MAP** minimises χ² (the posterior is flat apart from glafic's Gaussian
  priors) from several jittered starts.
* **SVI** fits an `AutoMultivariateNormal`, `AutoLowRankMultivariateNormal` or
  `AutoNormal` guide, initialised at the MAP.
* **NUTS** starts its chains from SVI draws and uses the SVI covariance as
  the initial dense mass matrix.

Stage options (`point_mode`, `ext_mode`, …) can be set for the whole
pipeline or per stage. You can also change the model between stages, for
example by fixing the cosmology in a first MAP.

**Source positions.** `model.set_source_mode(...)` selects how they enter:

* `"sample"` (default): they are free parameters.
* `"profile"`: they are replaced by the χ²-minimising position. This is exact
  in the source plane for position-only data.
* `"marginalize"`: they are integrated out analytically (source plane), as
  χ² → χ²_min + ln det M with M = Σ_k A_k/σ_k².

With position-only constraints, profiling and marginalising are exact. For
cluster models with many systems they cut the dimension dramatically and
make the MAP far more robust.

**Batched point χ².** The source-plane and tracked image-plane χ² are
evaluated with `vmap` over sources, so compile time does not grow with the
number of systems.

**Custom priors.** `model.extra_log_prior = lambda P: ...` adds an arbitrary
log-prior on the parameter pytree, for example
`jnp.where(P["cosmo"][2] + P["cosmo"][4] < 0, 0., -jnp.inf)` to require
w0 + wa < 0.

### Cosmology: w0waCDM

`P["cosmo"] = (Ω_m, Ω_Λ, w0, h, wa)` with w(a) = w0 + wa(1 − a). The C
code's parameters keep their meaning, `wa` is a new secondary parameter (it
defaults to 0, which is identical to glafic), and `wavary 1` samples it. Use
`flatfix 1` with `ovary 1` for flat models (Ω_Λ = 1 − Ω_m). In prior files,
`range wa lo hi` and `gauss wa med sig` work like the other cosmological
parameters. Distances agree with astropy's `w0waCDM` to better than 1e-14,
up to z ≈ 10.

A worked example on the cluster RXJ2129, with (Ω_m, w0, wa) free, is in
`examples_jax/rxj2129/rxj2129_hmc.ipynb`.

### Lower-level access

| what | function |
|---|---|
| parameter pytree `P` (`lens`, `cosmo`, `ext`, `point`, `psf`, `sky`) | `m.params()` |
| free-parameter vector ↔ `P` | `m.flatten(P)`, `m.unflatten(theta)`, `m.theta_names` |
| log posterior (−χ²/2, −∞ where glafic penalises) | `m.log_prob(theta)`, `m.make_log_prob()` |
| χ² and its pieces | `m.chi2(P)`, `m.chi2_components(P)` |
| lens equation at any positions (vectorised) | `m.lensmodel(x, y, zs, P)` |
| model image (`writeimage`) | `m.model_image(P)` |
| distances | `gj.cosmology.dis_angulard(z1, z2, cosmo)` |

All of these can be wrapped in `jax.jit`, `jax.grad` and `jax.vmap`.

## What is ported

* **Cosmology:** distances (any Ω, Λ, w, h) and halo overdensities.
* **All 26 lens models:** `gals`, `nfwpot`, `sie`, `jaffe`, `point`, `pert`,
  `clus3`, `mpole`, `hernpot`, `nfw`, `hern`, `powpot`, `pow` (TM15 or direct),
  `gnfwpot`, `gnfw`, `serspot`, `sers`, `tnfwpot`, `tnfw`, `einpot`, `ein`,
  `anfw`, `ahern`, `crline`, `gaupot` and `acnfw`. This includes the CSE
  tables and the gNFW/Einasto lookup tables, with the same bilinear
  interpolation.
* **Multi-plane lensing:** deflections, magnification matrix, rotation and
  time delays.
* **Point sources:** glafic's adaptive-mesh image finder, ported exactly
  (including float32 mesh storage, gap checks, Newton refinement and
  duplicate removal). Also the image-plane and source-plane χ² with flux,
  magnitude or luminosity-distance constraints, time delays, parities and the
  image-number check.
* **Extended sources:** `gauss`, `sersic`, `tophat`, `moffat`, `jaffe` and
  `point`, with glafic's sub-pixel integration, masks and noise estimation
  (`calc_obsnoise`). Also a double-Moffat or FITS PSF, `seeing_sub` bicubic
  sub-sampling, FFT convolution, and the sky level.
* **Priors and constraints:** `parprior` (`range`, `gauss` including the
  log-normal form, `match`, `extlens`, `poilens`, `relrange`), plus
  `flatfix`, `ovary`/`lvary`/`wvary`/`hvary` and `setopt` flags.
* **Input files:** primary and secondary parameters, `startup`, `setopt`,
  and the data commands (`readobs_point`, `readobs_extend`,
  `readnoise_extend`, `readpsf`, `parprior`, `reset_obs_point`). Any other
  command is stored in `model.commands`.

## Not ported

* Extended source `srcs`.
* `mapprior`.
* The output and analysis commands: `writecrit`, `writelens`, `calcein`,
  `kapparad`, `mock*`, `vary*`, the FITS writers, and glafic's own `mcmc`.
  Use `run_nuts` instead of glafic's `mcmc`, and `optimize` instead of the
  amoeba-based optimiser.

## How differentiability is achieved

* **Branch-heavy C code:** all of it is written with NaN-safe `jnp.where`.
  The fixed-tolerance adaptive integrals (Romberg, cquad) become fixed-order
  Gauss–Legendre rules. These converge to about 1e-12, so the port is more
  accurate than glafic's 5e-4 Romberg tolerance.
* **Image positions (image-plane χ²):** the images are found on the host with
  the exact glafic algorithm, through `jax.pure_callback`. One Newton step of
  the lens equation is then applied in JAX. This leaves the value unchanged
  but gives exact derivatives through the implicit-function theorem
  (`point_mode="glafic"`, the default).
  `point_mode="tracked"` is the pure-JAX alternative. It solves the lens
  equation by Newton iteration starting from the observed images: there is
  no callback, so it works on GPUs and with `vmap`, but it does not check the
  number of images.
* **Source positions:** these are *sampled* when flagged in `setopt`. glafic
  instead optimises them inside every χ² evaluation.
* **Analytic marginalisation:** the flux normalisation and the time-delay
  zero point are profiled analytically, as in glafic. The time-delay term
  uses a cancellation-free formula, because the usual "reference image with
  1e-6 day error" convention otherwise destroys gradient precision.

## Differences from glafic worth knowing

* **Static structure.** The grouping of lenses into planes, and the number of
  planes in front of each source, are fixed at `model_init()`. This means
  redshifts should not cross each other while sampling.
* **Discontinuities.** These are faithful to glafic, but they matter for HMC:
  * glafic switches the integration limit of elliptical-density models at
    ξ² = 10, which causes jumps of about 1e-3 in the deflection.
    `set_secondary("ell_accurate 1")` removes them.
  * glafic's sub-pixel refinement of extended sources switches on near the
    source centre. `m.log_prob(theta, ext_mode="uniform", n_sub=3)` uses a
    smooth fixed supersampling instead.
  * The number of point-source images changes across caustics. NUTS cannot
    cross these regions, so start it from a good fit, for example the output
    of `optimize`.
* **Source-plane χ².** The derivative of the magnification uses autodiff
  instead of finite differences, so values differ from C by about 1e-9.

## Validation

`test_jax/` contains 63 tests. To also compare against the compiled C code,
build it with `make python` and run:

```
GLAFIC_C_PATH=/path/to/dir/containing/glafic/package pytest test_jax
```

Results:

| quantity | agreement with C glafic |
|---|---|
| analytic lens models (`sie`, `pert`, `nfwpot`, `anfw`, `acnfw`, `hernpot`, …) | ~1e-15 |
| numerically integrated models (`nfw`, `hern`, `sers`, `gnfw`, `ein`, `tnfw`) | ~1e-4 (the C Romberg tolerance) |
| images, magnifications and time delays from `findimg` (`point.input`) | ~1e-15 |
| point χ², image plane / source plane | 7e-16 / 4e-10 |
| model images for `quadhost`, `multilens_extend` and `extend` | 5e-8, 6e-8, 3e-6 (C stores float32) |
| extended χ² for the same samples | 3e-8, 3e-9, 3e-7 |
| JAX gradients vs finite differences | ~1e-6 (point), ~1e-9 (extended) |

## Performance

These are CPU timings on one core after compilation:

| case | time per call |
|---|---|
| point-source `log_prob` + gradient, source plane | ~1 ms |
| point-source `log_prob` + gradient, image plane (glafic mode) | ~15 ms |
| `quadhost` (100×100 pixels, PSF, three sources), value only | ~0.2 s |
| `quadhost`, value + gradient | ~0.7 s (`ext_mode="uniform"`), ~1 s (glafic mode) |

Most of the extended-source cost is glafic's (4 × `num_pixint`)² = 400-fold
sub-sampling near the source centre. This is evaluated for every pixel
because the refined region is not known in advance. A GPU, or
`ext_mode="uniform"` with a small `n_sub`, makes it much faster.

Compilation takes 10–60 s the first time a model is used.
