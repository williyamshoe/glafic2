"""MAP -> SVI -> HMC modelling pipeline (NumPyro).

Every free parameter of a :class:`~glafic_jax.model.Model` becomes a NumPyro
sample site with a *flat* prior on its glafic ``parprior`` range (half-open
or unbounded ranges are allowed).  NumPyro maps each site to an unconstrained
space, so the three stages never hit hard boundaries.  The likelihood and
glafic's Gaussian priors enter through ``numpyro.factor``.

1. **MAP**: maximise the posterior (= minimise chi^2, since the priors are
   flat) with L-BFGS in the unconstrained space, from several starting
   points if requested.
2. **SVI**: fit a (low-rank) multivariate-normal guide, initialised at the
   MAP, to get a fast approximation of the posterior including correlations.
3. **HMC**: run NUTS initialised from the SVI solution, with the SVI
   covariance as the initial (dense) inverse mass matrix.  This shortens
   warm-up considerably.

Example::

    pipe = gj.Pipeline(model, point_mode="tracked")
    m = pipe.map(n_starts=4)
    s = pipe.svi(num_steps=3000)
    h = pipe.hmc(num_warmup=500, num_samples=1000, num_chains=2)

Between stages you may change the model's settings (e.g. switch from
source-plane to image-plane chi^2 after a first MAP).  Each stage uses the
model state at the time it is called.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_value
from numpyro.infer import autoguide

BIG = 1.0e29


def _range_of(model, name, k):
    """glafic prior range (min, max) of free parameter number k."""
    nl = len(model._tm_lens)
    nc = len(model._tm_cosmo)
    ne = len(model._tm_ext)
    nsky = 1 if model._tm_sky else 0
    nz = len(model._tm_poi_z)
    nxy = len(model._tm_poi_xy)
    if k < nl:
        i, j = model._tm_lens[k]
        return model.prior_lens["min"][i, j], model.prior_lens["max"][i, j]
    k -= nl
    if k < nc:
        c = model._tm_cosmo[k]
        return model.prior_cosmo["min"][c], model.prior_cosmo["max"][c]
    k -= nc
    if k < ne:
        i, j = model._tm_ext[k]
        return model.prior_ext["min"][i, j], model.prior_ext["max"][i, j]
    k -= ne
    if k < nsky:
        return -np.inf, np.inf
    k -= nsky
    if k < nz:
        i = model._tm_poi_z[k]
        return model.prior_point["min"][i, 0], model.prior_point["max"][i, 0]
    k -= nz
    if k < nxy:
        i, j = model._tm_poi_xy[k]
        return model.prior_point["min"][i, j + 1], model.prior_point["max"][i, j + 1]
    k -= nxy
    j = model._tm_psf[k]
    return model.prior_psf["min"][j], model.prior_psf["max"][j]


def _constraint(lo, hi):
    lo = None if (lo is None or not np.isfinite(lo) or lo <= -BIG) else float(lo)
    hi = None if (hi is None or not np.isfinite(hi) or hi >= BIG) else float(hi)
    if lo is not None and hi is not None:
        return constraints.interval(lo, hi)
    if lo is not None:
        return constraints.greater_than(lo)
    if hi is not None:
        return constraints.less_than(hi)
    return constraints.real


class Pipeline:
    """MAP -> SVI -> HMC for a glafic_jax :class:`Model`.

    Parameters
    ----------
    model : Model
    bounds : dict, optional
        ``{theta_name: (lo, hi)}`` overriding the glafic ranges (use
        ``None``/``inf`` for an open end).
    **lp_kw
        Forwarded to :meth:`Model.log_prob` (``point_mode``, ``ext_mode``,
        ``n_sub``).  They can be overridden per stage.
    """

    def __init__(self, model, bounds=None, **lp_kw):
        self.model = model
        self.lp_kw = lp_kw
        self.bounds = dict(bounds or {})
        self.results = {}
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self):
        m = self.model
        self.names = list(m.theta_names)
        self.site = [f"{k:03d}_{n}" for k, n in enumerate(self.names)]   # sorted == theta order
        cons = []
        for k, n in enumerate(self.names):
            lo, hi = self.bounds.get(n, _range_of(m, n, k))
            cons.append(_constraint(lo, hi))
        self.constraints = cons
        self.transforms = [biject_to(c) for c in cons]

    def to_theta(self, z):
        return jnp.stack([t(z[k]) for k, t in enumerate(self.transforms)])

    def to_z(self, theta):
        return jnp.stack([t.inv(theta[k]) for k, t in enumerate(self.transforms)])

    def clip_inside(self, theta, frac=1e-6):
        """Move a theta strictly inside the (finite) bounds."""
        th = np.array(theta, dtype=float)
        for k, c in enumerate(self.constraints):
            lo = getattr(c, "lower_bound", None)
            hi = getattr(c, "upper_bound", None)
            if lo is not None and hi is not None:
                eps = frac * (float(hi) - float(lo))
                th[k] = np.clip(th[k], float(lo) + eps, float(hi) - eps)
            elif lo is not None:
                th[k] = max(th[k], float(lo) + frac * max(1.0, abs(float(lo))))
            elif hi is not None:
                th[k] = min(th[k], float(hi) - frac * max(1.0, abs(float(hi))))
        return jnp.asarray(th)

    def log_prob(self, theta, **kw):
        return self.model.log_prob(theta, **{**self.lp_kw, **kw})

    def numpyro_model(self, floor=None, **kw):
        """NumPyro model: flat priors on the parameter ranges + glafic likelihood.

        ``floor`` replaces -inf (penalised regions) by a finite value, which
        keeps stochastic optimisation (SVI) stable."""
        def f():
            vals = [numpyro.sample(s, dist.ImproperUniform(c, (), ())) for s, c in zip(self.site, self.constraints)]
            theta = jnp.stack(vals)
            lp = self.log_prob(theta, **kw)
            if floor is not None:
                lp = jnp.where(jnp.isfinite(lp), lp, floor)
            numpyro.factor("glafic_loglike", lp)
        return f

    def _sites(self, theta):
        return {s: jnp.asarray(theta[k]) for k, s in enumerate(self.site)}

    def _theta_from_sites(self, d):
        return jnp.stack([jnp.asarray(d[s]) for s in self.site], axis=-1)

    # ------------------------------------------------------------------
    def map(self, theta0=None, n_starts=1, jitter=0.1, maxiter=3000, restarts=5, seed=0, verbose=True, **kw):
        """Maximum a posteriori fit (minimum chi^2) in unconstrained space.

        ``n_starts > 1`` adds starting points jittered by ``jitter`` (in
        unconstrained units) around ``theta0``; the best result is kept.
        Returns a dict with ``theta``, ``chi2``, ``names``.
        """
        from scipy.optimize import minimize
        self._refresh()
        m = self.model
        theta0 = self.clip_inside(m.theta0() if theta0 is None else theta0)
        z0 = np.asarray(self.to_z(theta0))
        pen = 0.5 * m.settings.chi2pen_range
        lp_kw = {**self.lp_kw, **kw}

        def chi2_z(z):
            return m.chi2(m.unflatten(self.to_theta(z)), **lp_kw)

        vg = jax.jit(jax.value_and_grad(chi2_z))
        rng = np.random.default_rng(seed)
        best = None
        t0 = time.time()
        for s in range(n_starts):
            z = z0 if s == 0 else z0 + jitter * rng.standard_normal(z0.shape)
            c2 = float(vg(jnp.asarray(z))[0])
            if not (np.isfinite(c2) and c2 < pen):
                if verbose:
                    print(f"[MAP] start {s}: invalid starting point, skipped")
                continue
            for it in range(restarts):
                norm = max(c2, 1.0)

                def f(x):
                    v, g = vg(jnp.asarray(x))
                    v = float(v)
                    g = np.asarray(g, dtype=float)
                    if not np.isfinite(v) or v >= pen or not np.all(np.isfinite(g)):
                        return 1e6, np.zeros_like(g)
                    return v / norm, g / norm

                res = minimize(f, z, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
                c2n = float(vg(jnp.asarray(res.x))[0])
                if not (c2n < c2):
                    break
                done = c2 - c2n < 1e-8 * max(1.0, c2)
                z, c2 = res.x, c2n
                if done:
                    break
            if verbose:
                print(f"[MAP] start {s}: chi2 = {c2:.6g}")
            if best is None or c2 < best[1]:
                best = (z, c2)
        if best is None:
            raise ValueError("no valid starting point for MAP")
        theta = self.to_theta(jnp.asarray(best[0]))
        out = dict(theta=np.asarray(theta), chi2=best[1], names=self.names, time=time.time() - t0)
        if verbose:
            print(f"[MAP] best chi2 = {best[1]:.6g}  ({out['time']:.0f} s)")
        self.results["map"] = out
        return out

    # ------------------------------------------------------------------
    def svi(self, theta0=None, num_steps=3000, guide="mvn", rank=None, lr=5e-3, num_particles=4,
            init_scale=0.01, num_draws=2000, seed=0, progress_bar=False, verbose=True, **kw):
        """Stochastic variational inference initialised at the MAP.

        guide : 'mvn' (AutoMultivariateNormal), 'lowrank'
            (AutoLowRankMultivariateNormal) or 'diag' (AutoNormal).
        Returns a dict with draws from the approximate posterior (``theta``),
        the guide median (``theta_median``), the unconstrained covariance
        (``cov_z``), the ELBO loss history, and the guide objects.
        """
        self._refresh()
        if theta0 is None:
            theta0 = self.results["map"]["theta"] if "map" in self.results else self.model.theta0()
        theta0 = self.clip_inside(theta0)
        lp0 = float(jax.jit(lambda t: self.log_prob(t, **kw))(theta0))
        if not np.isfinite(lp0):
            raise ValueError("log-probability is -inf at the SVI starting point")
        model_fn = self.numpyro_model(floor=lp0 - 1.0e6, **kw)
        init = init_to_value(values=self._sites(theta0))
        if guide == "mvn":
            g = autoguide.AutoMultivariateNormal(model_fn, init_loc_fn=init, init_scale=init_scale)
        elif guide == "lowrank":
            g = autoguide.AutoLowRankMultivariateNormal(model_fn, init_loc_fn=init, init_scale=init_scale, rank=rank)
        elif guide == "diag":
            g = autoguide.AutoNormal(model_fn, init_loc_fn=init, init_scale=init_scale)
        else:
            raise ValueError("guide must be 'mvn', 'lowrank' or 'diag'")
        optim = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
        svi = SVI(model_fn, g, optim, Trace_ELBO(num_particles=num_particles))
        t0 = time.time()
        res = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=progress_bar)
        params = res.params
        key = jax.random.PRNGKey(seed + 1)
        draws = g.sample_posterior(key, params, sample_shape=(num_draws,))
        theta = np.asarray(self._theta_from_sites(draws))
        med = np.asarray(self._theta_from_sites(g.median(params)))
        cov = None
        if guide in ("mvn", "lowrank"):
            post = g.get_posterior(params)
            cov = np.asarray(post.covariance_matrix)
        else:
            sc = g.get_posterior(params) if hasattr(g, "get_posterior") else None
            zs = np.asarray(jax.vmap(self.to_z)(jnp.asarray(theta)))
            cov = np.diag(zs.var(axis=0))
        out = dict(theta=theta, theta_median=med, cov_z=cov, losses=np.asarray(res.losses), names=self.names,
                   guide=g, params=params, time=time.time() - t0)
        if verbose:
            print(f"[SVI] final loss {float(np.mean(res.losses[-100:])):.6g}, median chi2 = "
                  f"{-2 * float(jax.jit(lambda t: self.log_prob(t, **kw))(jnp.asarray(med))):.6g} ({out['time']:.0f} s)")
        self.results["svi"] = out
        return out

    # ------------------------------------------------------------------
    def hmc(self, theta0=None, num_warmup=500, num_samples=1000, num_chains=1, seed=0, target_accept=0.8,
            dense_mass=True, use_svi_mass=True, adapt_mass_matrix=True, max_tree_depth=10,
            chain_method=None, init_spread=1.0, progress_bar=True, verbose=True, **kw):
        """NUTS sampling, warm-started from SVI (or MAP).

        Chains start from independent draws of the SVI approximation (scaled
        by ``init_spread``) when available.  Returns a dict with ``theta``
        (chains, samples, ndim), ``log_prob``, ``diverging``, ``names`` and the
        NumPyro ``MCMC`` object.
        """
        self._refresh()
        lp_kw = {**self.lp_kw, **kw}
        model_fn = self.numpyro_model(**kw)
        svi = self.results.get("svi")
        if theta0 is None:
            theta0 = svi["theta_median"] if svi is not None else (
                self.results["map"]["theta"] if "map" in self.results else self.model.theta0())
        theta0 = self.clip_inside(theta0)
        lpf = jax.jit(lambda t: self.log_prob(t, **kw))
        if not np.isfinite(float(lpf(theta0))):
            raise ValueError("log-probability is -inf at the HMC starting point")
        # starting points for the chains
        inits = [np.asarray(theta0)]
        if num_chains > 1 and svi is not None:
            rng = np.random.default_rng(seed)
            cand = svi["theta"][rng.permutation(len(svi["theta"]))]
            zc = np.asarray(self.to_z(jnp.asarray(theta0)))
            for th in cand:
                if len(inits) == num_chains:
                    break
                z = zc + init_spread * (np.asarray(self.to_z(jnp.asarray(th))) - zc)
                th2 = np.asarray(self.to_theta(jnp.asarray(z)))
                if np.isfinite(float(lpf(jnp.asarray(th2)))):
                    inits.append(th2)
        while len(inits) < num_chains:
            inits.append(np.asarray(theta0))
        inv_mass = None
        if use_svi_mass and svi is not None and svi["cov_z"] is not None:
            inv_mass = svi["cov_z"] if dense_mass else np.diag(svi["cov_z"])
        kernel = NUTS(model_fn, target_accept_prob=target_accept, dense_mass=dense_mass,
                      inverse_mass_matrix=None if inv_mass is None else jnp.asarray(inv_mass),
                      adapt_mass_matrix=adapt_mass_matrix, max_tree_depth=max_tree_depth,
                      init_strategy=init_to_value(values=self._sites(inits[0])))
        if chain_method is None:
            pure = (self.model.settings.chi2_splane == 1 or lp_kw.get("point_mode") == "tracked") and \
                lp_kw.get("point_mode") != "glafic" or not any(o is not None for o in self.model.obs_point)
            chain_method = "vectorized" if (pure and num_chains > 1) else "sequential"
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method=chain_method, progress_bar=progress_bar)
        init_params = None
        if num_chains > 1:
            init_params = {s: jnp.asarray(np.array([biject_to(c).inv(th[k]) for th in inits]))
                           for k, (s, c) in enumerate(zip(self.site, self.constraints))}
        t0 = time.time()
        mcmc.run(jax.random.PRNGKey(seed), init_params=init_params, extra_fields=("potential_energy", "diverging"))
        smp = mcmc.get_samples(group_by_chain=True)
        theta = np.asarray(self._theta_from_sites(smp))
        ex = mcmc.get_extra_fields(group_by_chain=True)
        lp = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda t: self.log_prob(t, **kw))))(jnp.asarray(theta)))
        out = dict(theta=theta, log_prob=lp, diverging=np.asarray(ex["diverging"]), names=self.names,
                   mcmc=mcmc, time=time.time() - t0)
        if verbose:
            print(f"[HMC] {num_chains} chain(s) x {num_samples} samples, {int(out['diverging'].sum())} divergences, "
                  f"({out['time']:.0f} s)")
        self.results["hmc"] = out
        return out

    # ------------------------------------------------------------------
    def run(self, theta0=None, map_kw=None, svi_kw=None, hmc_kw=None):
        """MAP -> SVI -> HMC with default settings; returns ``self.results``."""
        self.map(theta0, **(map_kw or {}))
        self.svi(**(svi_kw or {}))
        self.hmc(**(hmc_kw or {}))
        return self.results

    def to_arviz(self, stage="hmc"):
        """ArviZ InferenceData of the HMC samples (requires arviz)."""
        import arviz as az
        r = self.results[stage]
        th = r["theta"] if r["theta"].ndim == 3 else r["theta"][None]
        post = {n: th[..., k] for k, n in enumerate(self.names)}
        return az.from_dict(posterior=post, sample_stats={"lp": r["log_prob"]} if stage == "hmc" else None)


def diagnostics(result):
    """R-hat and effective sample size per parameter (numpyro diagnostics)."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    th = result["theta"]
    if th.ndim == 2:
        th = th[None]
    ess = np.asarray(effective_sample_size(th))
    rhat = np.asarray(split_gelman_rubin(th)) if th.shape[0] > 1 else np.full(th.shape[-1], np.nan)
    return dict(names=result["names"], ess=ess, rhat=rhat)
