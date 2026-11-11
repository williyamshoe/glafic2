"""MAP -> (Laplace) -> SVI -> HMC modelling pipeline (NumPyro).

Every free parameter of a :class:`~glafic_jax.model.Model` becomes a NumPyro
sample site with a *flat* prior on a range (by default its glafic
``parprior`` range; half-open or unbounded ranges are allowed).  NumPyro maps
each site to an unconstrained space ``z``, so no stage meets a hard boundary.
The likelihood and glafic's Gaussian priors enter through ``numpyro.factor``.

Re-parameterisation
-------------------
``reparam`` chooses what is actually sampled (and hence which prior is flat):

* ``"log10"``: sample log10 of the parameter (flat in log).  Recommended for
  masses, scale radii, etc.
* ``("offset", other, lo, hi)``: sample ``param + other`` in ``[lo, hi]``,
  e.g. ``{"wa": ("offset", "weos", -3.0, 0.0)}`` samples ``w0 + wa`` so the
  condition ``w0 + wa < 0`` is part of the parameter space instead of a hard
  wall in the posterior (walls cause NUTS divergences).
* ``("sum_below", other, lo, hi, cap)``: a prior flat in (param, other) on
  ``lo <= param <= hi`` and ``param + other < cap``.  Sampled as the fraction
  u in [0, 1) of the allowed range, ``param = lo + u (min(hi, cap - other) - lo)``,
  with the Jacobian added so the prior stays flat; no walls.  DESI's w0waCDM
  prior (w0 in [-3, 1], wa in [-3, 2], w0 + wa < 0) is
  ``{"wa": ("sum_below", "weos", -3.0, 2.0, 0.0)}`` with the w0 range [-3, 1].
* ``("nfw_alpha", R, zs_ref)`` for the mass of an NFW-family halo
  (``nfw``, ``nfwpot``, ``anfw``, ``tnfw``...; glafic ``nfw_users 0``):
  sample log10 of the *spherical NFW deflection at radius R* [arcsec] for a
  source at ``zs_ref`` instead of the virial mass.  This is proportional to
  the projected mass inside R, the combination the multiple images measure,
  so the mass-concentration degeneracy becomes a straight, axis-aligned
  direction.  The mass is recovered by Newton inversion at the current
  concentration and cosmology.  The prior stays flat in log10 M (the
  Jacobian is included).

Stages
------
1. **MAP**: minimise chi^2 with L-BFGS in ``z`` from several jittered starts.
2. **Laplace**: Gaussian approximation from the Hessian of the potential at
   the MAP (in ``z``).  Cheap, and a robust initialiser.
3. **SVI**: (low-rank) multivariate-normal guide initialised *from the
   Laplace approximation* (location and Cholesky factor).
4. **HMC**: NUTS.  The chains start from draws of the best available
   approximation (SVI if it is consistent with the MAP, otherwise Laplace),
   whose covariance is the initial dense inverse mass matrix.

Example::

    pipe = gj.Pipeline(model, reparam={"lens1_anfw_p2": "log10"})
    pipe.map(n_starts=8)
    pipe.laplace()
    pipe.svi(num_steps=3000)
    res = pipe.hmc(num_warmup=1000, num_samples=1000, num_chains=4)

The model's settings may be changed between stages (e.g. fixing cosmology
in a first MAP); each stage uses the model state at the time it is called.
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


def _range_of(model, k):
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


def _finite(v, sign):
    return v is not None and np.isfinite(v) and (v > -BIG if sign < 0 else v < BIG)


def _constraint(lo, hi):
    lo = float(lo) if _finite(lo, -1) else None
    hi = float(hi) if _finite(hi, +1) else None
    if lo is not None and hi is not None:
        return constraints.interval(lo, hi)
    if lo is not None:
        return constraints.greater_than(lo)
    if hi is not None:
        return constraints.less_than(hi)
    return constraints.real


class Pipeline:
    """MAP -> Laplace -> SVI -> HMC for a glafic_jax :class:`Model`.

    Parameters
    ----------
    model : Model
    bounds : dict, optional
        ``{theta_name: (lo, hi)}`` overriding the glafic ranges (in parameter
        units, also for log10-sampled parameters).
    reparam : dict, optional
        ``{theta_name: "log10" | ("offset", other_name, lo, hi) |
        ("sum_below", other_name, lo, hi, cap) | ("nfw_alpha", R, zs_ref)}``.
    **lp_kw
        Forwarded to :meth:`Model.log_prob` (``point_mode``, ``ext_mode``,
        ``n_sub``).  Can be overridden per stage.
    """

    def __init__(self, model, bounds=None, reparam=None, **lp_kw):
        self.model = model
        self.lp_kw = lp_kw
        self.bounds = dict(bounds or {})
        self.reparam = dict(reparam or {})
        self.results = {}
        self._refresh()

    # ------------------------------------------------------------------
    # parameter maps:  z (unconstrained) <-> v (sampled sites) <-> theta
    def _refresh(self):
        m = self.model
        self.names = list(m.theta_names)
        unknown = set(self.reparam) - set(self.names)
        if unknown:
            raise ValueError(f"reparam refers to unknown parameters {sorted(unknown)}")
        self.kind, self.ref, self.site, cons = [], [], [], []
        self._nfw = {}
        self._capped = {}
        for k, n in enumerate(self.names):
            spec = self.reparam.get(n, "linear")
            lo, hi = self.bounds.get(n, _range_of(m, k))
            if spec == "log10":
                if not (_finite(lo, -1) and lo > 0):
                    raise ValueError(f"log10 parameter {n} needs a positive lower bound")
                lo = np.log10(lo)
                hi = np.log10(hi) if _finite(hi, +1) else None
                self.kind.append("log10")
                self.ref.append(-1)
                self.site.append(f"{k:03d}_log10_{n}")
            elif isinstance(spec, tuple) and spec[0] == "offset":
                _, other, lo, hi = spec
                r = self.names.index(other)
                if self.reparam.get(other, "linear") != "linear":
                    raise ValueError("offset reference must be a linearly sampled parameter")
                self.kind.append("offset")
                self.ref.append(r)
                self.site.append(f"{k:03d}_{n}_plus_{other}")
            elif isinstance(spec, tuple) and spec[0] == "sum_below":
                _, other, plo, phi, cap = spec
                r = self.names.index(other)
                if self.reparam.get(other, "linear") != "linear":
                    raise ValueError("sum_below reference must be a linearly sampled parameter")
                rlo, rhi = self.bounds.get(other, _range_of(m, r))
                if not _finite(rhi, +1) or min(phi, cap - rhi) <= plo:
                    raise ValueError(f"sum_below for {n}: empty range when {other} = {rhi}")
                self._capped[k] = (float(plo), float(phi), float(cap))
                self.kind.append("sum_below")
                self.ref.append(r)
                self.site.append(f"{k:03d}_{n}_frac")
                lo, hi = 0.0, 1.0
            elif isinstance(spec, tuple) and spec[0] == "nfw_alpha":
                _, R, zref = spec
                i, rest = n[4:].split("_", 1)
                if not (n.startswith("lens") and rest.endswith("_p2")):
                    raise ValueError("nfw_alpha applies to the mass (p2) of a lens")
                if m.settings.nfw_users != 0:
                    raise ValueError("nfw_alpha needs nfw_users 0 (mass, concentration)")
                il = int(i) - 1
                cname = f"lens{i}_{rest[:-3]}_p7"
                kc = self.names.index(cname) if cname in self.names else -1
                self._nfw[k] = (il, float(R), float(zref), kc)
                mlo, mhi = self.bounds.get(n, _range_of(m, k))
                lo, hi = self.bounds.get(n + ":alpha", (-3.0, 4.0))
                self.kind.append("nfw_alpha")
                self.ref.append(-1)
                self.site.append(f"{k:03d}_log10alpha{R:g}_{n}")
            elif spec == "linear":
                self.kind.append("linear")
                self.ref.append(-1)
                self.site.append(f"{k:03d}_{n}")
            else:
                raise ValueError(f"unknown reparam {spec!r} for {n}")
            cons.append(_constraint(lo, hi))
        self.constraints = cons
        self.transforms = [biject_to(c) for c in cons]

    def v_to_theta(self, v):
        v = jnp.asarray(v)
        if self._nfw and v.ndim > 1:                 # batched (sites first): vmap the inversion
            sh = v.shape[1:]
            out = jax.vmap(self.v_to_theta, in_axes=1)(v.reshape(v.shape[0], -1))
            return out.reshape(sh + (v.shape[0],))
        out = []
        for k, kind in enumerate(self.kind):
            if kind == "log10":
                out.append(10.0 ** v[k])
            elif kind == "offset":
                out.append(v[k] - v[self.ref[k]])
            elif kind == "sum_below":
                out.append(self._capped[k][0] + v[k] * self._capped_width(k, v[self.ref[k]]))
            elif kind == "nfw_alpha":
                out.append(jnp.asarray(1.0e14, dtype=v.dtype))       # placeholder, solved below
            else:
                out.append(v[k])
        theta = jnp.stack(out, axis=-1)
        for k in self._nfw:
            theta = theta.at[k].set(10.0 ** self._nfw_log10m(theta, v[k], k))
        return theta

    def theta_to_v(self, theta):
        theta = jnp.asarray(theta)
        out = []
        for k, kind in enumerate(self.kind):
            if kind == "log10":
                out.append(jnp.log10(theta[k]))
            elif kind == "offset":
                out.append(theta[k] + theta[self.ref[k]])
            elif kind == "sum_below":
                out.append((theta[k] - self._capped[k][0]) / self._capped_width(k, theta[self.ref[k]]))
            elif kind == "nfw_alpha":
                out.append(self._nfw_log10alpha(theta, jnp.log10(theta[k]), k))
            else:
                out.append(theta[k])
        return jnp.stack(out)

    # --- NFW mass <-> deflection at R ---------------------------------
    def _nfw_log10alpha(self, theta, log10m, k):
        """log10 of the spherical NFW deflection at R [arcsec] (source at zs_ref)."""
        from . import cosmology as cm
        from .lens_models import PlaneContext, calc_bbtt_nfw, dphi_nfw_dl
        il, R, zref, kc = self._nfw[k]
        m = self.model
        P = m.unflatten(theta)
        cosmo = P["cosmo"]
        zl = P["lens"][il, 0]
        c = theta[kc] if kc >= 0 else P["lens"][il, 6]
        s = m.settings
        ctx = PlaneContext(zl, cm.dis_angulard(0.0, zref, cosmo), cm.dis_angulard(0.0, zl, cosmo),
                           cm.dis_angulard(zl, zref, cosmo),
                           cm.deltaomega(zl, cosmo, s.flag_hodensity, s.hodensity), cosmo, s, None)
        bb, tt = calc_bbtt_nfw(10.0 ** log10m, c, ctx)
        return jnp.log10(bb * tt * dphi_nfw_dl(R / tt))

    def _nfw_log10m(self, theta, log10a, k, iters=30):
        """Invert log10 alpha(M) = log10a at fixed concentration and cosmology.
        Newton on a stopped-gradient copy, then one differentiable Newton step
        (exact first derivatives via the implicit-function theorem)."""
        f = lambda u, th: self._nfw_log10alpha(th, u, k)
        df = jax.grad(f)
        th0 = jax.lax.stop_gradient(theta)
        a0 = jax.lax.stop_gradient(log10a)

        def body(_, u):
            return jnp.clip(u - (f(u, th0) - a0) / df(u, th0), 9.0, 19.0)
        u = jax.lax.fori_loop(0, iters, body, jnp.asarray(14.5, dtype=theta.dtype))
        u = jax.lax.stop_gradient(u)
        return u - (f(u, theta) - log10a) / jax.lax.stop_gradient(df(u, th0))

    def _capped_width(self, k, ref_value):
        """Width of the allowed range of a sum_below parameter, given its reference parameter."""
        plo, phi, cap = self._capped[k]
        return jnp.minimum(phi, cap - ref_value) - plo

    def log_prior_reparam(self, v):
        """Extra log-density on the sampled sites so that priors stay flat in
        the original parameters where requested (nfw_alpha: flat in log10 M;
        sum_below: flat in the parameter, via d param / d u = allowed width)."""
        tot = 0.0
        for k in self._capped:
            tot = tot + jnp.log(self._capped_width(k, v[self.ref[k]]))
        if not self._nfw:
            return tot
        theta = self.v_to_theta(v)
        for k in self._nfw:
            u = jnp.log10(theta[k])
            dadu = jax.grad(lambda uu: self._nfw_log10alpha(theta, uu, k))(u)
            tot = tot - jnp.log(jnp.abs(dadu))
        return tot

    def z_to_v(self, z):
        return jnp.stack([t(z[k]) for k, t in enumerate(self.transforms)])

    def v_to_z(self, v):
        return jnp.stack([t.inv(v[k]) for k, t in enumerate(self.transforms)])

    def to_theta(self, z):
        return self.v_to_theta(self.z_to_v(z))

    def to_z(self, theta):
        return self.v_to_z(self.theta_to_v(theta))

    def log_jac(self, z):
        """log |d v / d z| (the flat prior on v becomes this density on z)."""
        v = self.z_to_v(z)
        return sum(jnp.sum(t.log_abs_det_jacobian(z[k], v[k])) for k, t in enumerate(self.transforms))

    def clip_inside(self, theta, frac=1e-6):
        """Move theta strictly inside the sampling ranges."""
        v = np.array(self.theta_to_v(theta), dtype=float)
        for k, c in enumerate(self.constraints):
            lo = getattr(c, "lower_bound", None)
            hi = getattr(c, "upper_bound", None)
            if lo is not None and hi is not None:
                eps = frac * (float(hi) - float(lo))
                v[k] = np.clip(v[k], float(lo) + eps, float(hi) - eps)
            elif lo is not None:
                v[k] = max(v[k], float(lo) + frac * max(1.0, abs(float(lo))))
            elif hi is not None:
                v[k] = min(v[k], float(hi) - frac * max(1.0, abs(float(hi))))
        return self.v_to_theta(jnp.asarray(v))

    def log_prob(self, theta, **kw):
        return self.model.log_prob(theta, **{**self.lp_kw, **kw})

    def potential_z(self, z, **kw):
        """-log posterior density in z (includes the Jacobian)."""
        return -(self.log_prob(self.to_theta(z), **kw) + self.log_jac(z)
                 + self.log_prior_reparam(self.z_to_v(z)))

    def numpyro_model(self, floor=None, **kw):
        """NumPyro model with flat priors on the sampling ranges + glafic likelihood.
        ``floor`` replaces -inf (penalised regions) by a finite value (for SVI)."""
        def f():
            v = jnp.stack([numpyro.sample(s, dist.ImproperUniform(c, (), ()))
                           for s, c in zip(self.site, self.constraints)])
            lp = self.log_prob(self.v_to_theta(v), **kw) + self.log_prior_reparam(v)
            if floor is not None:
                lp = jnp.where(jnp.isfinite(lp), lp, floor)
            numpyro.factor("glafic_loglike", lp)
        return f

    def _sites(self, theta):
        v = self.theta_to_v(theta)
        return {s: v[k] for k, s in enumerate(self.site)}

    def _theta_from_sites(self, d):
        v = jnp.stack([jnp.asarray(d[s]) for s in self.site], axis=-1)
        return self.v_to_theta(jnp.moveaxis(v, -1, 0)) if v.ndim > 1 else self.v_to_theta(v)

    def _chi2(self, theta, **kw):
        return -2.0 * float(jax.jit(lambda t: self.log_prob(t, **kw))(jnp.asarray(theta)))

    # ------------------------------------------------------------------
    def map(self, theta0=None, n_starts=1, jitter=0.1, maxiter=3000, restarts=5, seed=0, verbose=True, **kw):
        """Minimum chi^2 (= MAP for flat priors) in the unconstrained space.

        ``n_starts > 1`` adds starting points jittered by ``jitter`` (in
        unconstrained units) around ``theta0``; the best result is kept.
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
            for _ in range(restarts):
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
        out = dict(theta=np.asarray(theta), chi2=best[1], z=np.asarray(best[0]), names=self.names,
                   time=time.time() - t0)
        if verbose:
            print(f"[MAP] best chi2 = {best[1]:.6g}  ({out['time']:.0f} s)")
        self.results["map"] = out
        return out

    def set_map(self, theta, **kw):
        """Use an externally obtained best fit as the MAP result."""
        self._refresh()
        theta = self.clip_inside(theta)
        self.results["map"] = dict(theta=np.asarray(theta), chi2=self._chi2(theta, **kw),
                                   z=np.asarray(self.to_z(theta)), names=self.names)
        return self.results["map"]

    # ------------------------------------------------------------------
    def laplace(self, theta0=None, num_draws=2000, newton_steps=3, seed=0, verbose=True, **kw):
        """Gaussian (Laplace) approximation at the posterior mode in z.

        A few damped Newton steps move from the chi^2 minimum to the mode of
        the potential in z (they differ by the Jacobian); the Hessian there
        gives the covariance.  Non-positive curvature directions are
        regularised.  Returns draws in theta and the covariance in z.
        """
        self._refresh()
        if theta0 is None:
            theta0 = self.results["map"]["theta"]
        z = self.to_z(self.clip_inside(theta0))
        U = jax.jit(lambda zz: self.potential_z(zz, **kw))
        H_f = jax.jit(jax.hessian(lambda zz: self.potential_z(zz, **kw)))
        g_f = jax.jit(jax.grad(lambda zz: self.potential_z(zz, **kw)))
        t0 = time.time()
        for _ in range(newton_steps):
            H = np.asarray(H_f(z))
            w, V = np.linalg.eigh(0.5 * (H + H.T))
            w = np.maximum(w, 1e-6 * max(w.max(), 1.0))
            step = -(V / w) @ (V.T @ np.asarray(g_f(z)))
            u0 = float(U(z))
            for a in (1.0, 0.5, 0.25, 0.1):
                zn = z + a * step
                if np.isfinite(float(U(zn))) and float(U(zn)) <= u0:
                    z = zn
                    break
        H = np.asarray(H_f(z))
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        nbad = int(np.sum(w <= 0))
        w = np.maximum(w, 1e-6 * max(w.max(), 1.0))
        cov = (V / w) @ V.T
        rng = np.random.default_rng(seed)
        zs = np.asarray(z) + rng.multivariate_normal(np.zeros(len(w)), cov, size=num_draws)
        theta = np.asarray(jax.vmap(self.to_theta)(jnp.asarray(zs)))
        out = dict(z=np.asarray(z), cov_z=cov, theta=theta, theta_median=np.asarray(self.to_theta(z)),
                   n_negative_curvature=nbad, names=self.names, time=time.time() - t0)
        if verbose:
            print(f"[Laplace] mode chi2 = {self._chi2(out['theta_median'], **kw):.6g}; "
                  f"{nbad} non-positive curvature direction(s) regularised ({out['time']:.0f} s)")
        self.results["laplace"] = out
        return out

    # ------------------------------------------------------------------
    def svi(self, theta0=None, num_steps=3000, guide="mvn", rank=None, lr=1e-3, num_particles=8,
            init_scale=0.01, num_draws=2000, seed=0, progress_bar=False, verbose=True, **kw):
        """Stochastic variational inference.

        The guide ('mvn', 'lowrank' or 'diag') is initialised from the Laplace
        approximation if available (otherwise at the MAP with ``init_scale``).
        Returns draws (``theta``), the guide median, the covariance in z, the
        loss history and the median chi^2 (``chi2_median``).
        """
        self._refresh()
        lap = self.results.get("laplace")
        if theta0 is None:
            theta0 = lap["theta_median"] if lap is not None else (
                self.results["map"]["theta"] if "map" in self.results else self.model.theta0())
        theta0 = self.clip_inside(theta0)
        lp0 = float(jax.jit(lambda t: self.log_prob(t, **kw))(theta0))
        if not np.isfinite(lp0):
            raise ValueError("log-probability is -inf at the SVI starting point")
        model_fn = self.numpyro_model(floor=lp0 - 1.0e6, **kw)
        init = init_to_value(values=self._sites(theta0))
        init_params = None
        if guide == "mvn":
            g = autoguide.AutoMultivariateNormal(model_fn, init_loc_fn=init, init_scale=init_scale)
            if lap is not None:
                init_params = {"auto_loc": jnp.asarray(self.to_z(theta0)),
                               "auto_scale_tril": jnp.asarray(np.linalg.cholesky(lap["cov_z"]))}
        elif guide == "lowrank":
            g = autoguide.AutoLowRankMultivariateNormal(model_fn, init_loc_fn=init, init_scale=init_scale, rank=rank)
        elif guide == "diag":
            g = autoguide.AutoNormal(model_fn, init_loc_fn=init, init_scale=init_scale)
        else:
            raise ValueError("guide must be 'mvn', 'lowrank' or 'diag'")
        optim = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=10.0)
        svi = SVI(model_fn, g, optim, Trace_ELBO(num_particles=num_particles))
        t0 = time.time()
        res = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=progress_bar, init_params=init_params)
        params = res.params
        draws = g.sample_posterior(jax.random.PRNGKey(seed + 1), params, sample_shape=(num_draws,))
        theta = np.asarray(self._theta_from_sites(draws))
        med = np.asarray(self._theta_from_sites(g.median(params)))
        if guide in ("mvn", "lowrank"):
            cov = np.asarray(g.get_posterior(params).covariance_matrix)
        else:
            zs = np.asarray(jax.vmap(self.to_z)(jnp.asarray(theta)))
            cov = np.diag(zs.var(axis=0))
        out = dict(theta=theta, theta_median=med, cov_z=cov, losses=np.asarray(res.losses), names=self.names,
                   guide=g, params=params, chi2_median=self._chi2(med, **kw), time=time.time() - t0)
        if verbose:
            print(f"[SVI] final loss {float(np.mean(res.losses[-100:])):.6g}, median chi2 = "
                  f"{out['chi2_median']:.6g} ({out['time']:.0f} s)")
        self.results["svi"] = out
        return out

    # ------------------------------------------------------------------
    def _best_approx(self, tol, verbose, **kw):
        """SVI if its median is consistent with the MAP, else Laplace."""
        svi = self.results.get("svi")
        lap = self.results.get("laplace")
        ref = self.results.get("map", {}).get("chi2")
        if svi is not None:
            ok = ref is None or svi["chi2_median"] <= ref + tol
            if ok:
                return "svi", svi
            if verbose:
                print(f"[HMC] SVI median chi2 {svi['chi2_median']:.4g} >> MAP {ref:.4g}: "
                      f"using {'Laplace' if lap is not None else 'MAP'} instead")
        if lap is not None:
            return "laplace", lap
        return None, None

    def hmc(self, theta0=None, num_warmup=1000, num_samples=1000, num_chains=4, seed=0, target_accept=0.9,
            dense_mass=True, use_approx_mass=True, adapt_mass_matrix=True, max_tree_depth=10,
            chain_method=None, init_spread=0.5, approx_tol=None, init="approx", init_jitter=0.01,
            checkpoint=None, chunk=None, progress_bar=True, verbose=True, **kw):
        """NUTS, warm-started from the best Gaussian approximation.

        ``init="map"`` ignores the Laplace/SVI approximations for the starting
        points and starts each chain from ``theta0``/the MAP plus Gaussian noise
        of size ``init_jitter`` in the unconstrained space (only valid points
        are kept).  Combine with ``use_approx_mass=False`` when the posterior is
        truncated close to the mode (e.g. images near critical curves), where
        Gaussian approximations are meaningless.

        ``checkpoint`` (a file path) makes the run resumable: the sampler
        state is saved after warm-up and after every ``chunk`` samples; calling
        ``hmc`` again with the same ``checkpoint`` continues from there (useful
        for batch jobs with time limits).

        Chains start from independent draws of SVI (if its median chi^2 is
        within ``approx_tol`` of the MAP, default ``5 + ndim``) or of the
        Laplace approximation, shrunk towards the centre by ``init_spread``.
        Returns a dict with ``theta`` (chains, samples, ndim), ``log_prob``,
        ``diverging``, ``names`` and the NumPyro ``MCMC`` object.
        """
        self._refresh()
        lp_kw = {**self.lp_kw, **kw}
        model_fn = self.numpyro_model(**kw)
        ndim = len(self.names)
        tol = (5.0 + ndim) if approx_tol is None else approx_tol
        src, approx = self._best_approx(tol, verbose, **kw) if init == "approx" else ("map", None)
        if theta0 is None:
            theta0 = approx["theta_median"] if approx is not None else (
                self.results["map"]["theta"] if "map" in self.results else self.model.theta0())
        theta0 = self.clip_inside(theta0)
        lpf = jax.jit(lambda t: self.log_prob(t, **kw))
        if not np.isfinite(float(lpf(theta0))):
            raise ValueError("log-probability is -inf at the HMC starting point")
        inits = [np.asarray(theta0)]
        if num_chains > 1 and approx is not None:
            zc = np.asarray(self.to_z(jnp.asarray(theta0)))
            for th in approx["theta"][np.random.default_rng(seed).permutation(len(approx["theta"]))]:
                if len(inits) == num_chains:
                    break
                z = zc + init_spread * (np.asarray(self.to_z(jnp.asarray(th))) - zc)
                th2 = np.asarray(self.to_theta(jnp.asarray(z)))
                if np.isfinite(float(lpf(jnp.asarray(th2)))):
                    inits.append(th2)
        if init == "map" and num_chains > 1:
            inits = self._valid_jitter(theta0, num_chains, init_jitter, seed, lpf)
        while len(inits) < num_chains:
            inits.append(np.asarray(theta0))
        inv_mass = None
        if use_approx_mass and approx is not None:
            inv_mass = approx["cov_z"] if dense_mass else np.diag(approx["cov_z"])
        kernel = NUTS(model_fn, target_accept_prob=target_accept, dense_mass=dense_mass,
                      inverse_mass_matrix=None if inv_mass is None else jnp.asarray(inv_mass),
                      adapt_mass_matrix=adapt_mass_matrix, max_tree_depth=max_tree_depth,
                      init_strategy=init_to_value(values=self._sites(inits[0])))
        if chain_method is None:
            has_point = any(o is not None for o in self.model.obs_point)
            pure = (not has_point) or self.model.settings.chi2_splane == 1 or lp_kw.get("point_mode") == "tracked"
            chain_method = "vectorized" if (pure and num_chains > 1) else "sequential"
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method=chain_method, progress_bar=progress_bar)
        init_params = None
        if num_chains > 1:
            vs = np.stack([np.asarray(self.theta_to_v(jnp.asarray(th))) for th in inits])
            init_params = {s: jnp.asarray([biject_to(c).inv(v[k]) for v in vs])
                           for k, (s, c) in enumerate(zip(self.site, self.constraints))}
        if verbose:
            print(f"[HMC] initialised from {src or 'MAP'}; {num_chains} chain(s), method={chain_method}")
        t0 = time.time()
        smp, ex = _run_chunked(mcmc, jax.random.PRNGKey(seed), init_params, num_samples, chunk, checkpoint,
                               ("potential_energy", "diverging", "num_steps"), verbose)
        theta = np.asarray(self._theta_from_sites(smp))
        lp = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda t: self.log_prob(t, **kw))))(jnp.asarray(theta)))
        out = dict(theta=theta, log_prob=lp, diverging=np.asarray(ex["diverging"]), names=self.names,
                   num_steps=np.asarray(ex["num_steps"]), mcmc=mcmc, init=src, time=time.time() - t0)
        if verbose:
            n = out["diverging"].size
            print(f"[HMC] {num_chains} x {num_samples} samples, {int(out['diverging'].sum())}/{n} divergent "
                  f"({out['time']:.0f} s); median {np.median(out['num_steps']):.0f} leapfrog steps/sample")
        self.results["hmc"] = out
        return out

    def _valid_jitter(self, theta0, n, jitter, seed, lpf, max_tries=10000):
        """n valid points: theta0 plus Gaussian noise of size ``jitter`` in z."""
        rng = np.random.default_rng(seed)
        zc = np.asarray(self.to_z(jnp.asarray(theta0)))
        out = [np.asarray(theta0)]
        tries = 0
        while len(out) < n and tries < max_tries:
            tries += 1
            th = np.asarray(self.to_theta(jnp.asarray(zc + jitter * rng.standard_normal(zc.shape))))
            if np.isfinite(float(lpf(jnp.asarray(th)))):
                out.append(th)
        if len(out) < n:
            raise ValueError(f"only {len(out)} valid starting points found with jitter={jitter}")
        return out

    # ------------------------------------------------------------------
    def ensemble(self, theta0=None, num_walkers=None, num_warmup=2000, num_samples=2000, thinning=1,
                 init_jitter=0.01, moves=None, seed=0, checkpoint=None, chunk=None, progress_bar=True,
                 verbose=True, **kw):
        """Gradient-free affine-invariant ensemble sampler (NumPyro AIES).

        A reference for validating HMC on difficult (truncated, non-smooth)
        posteriors.  Walkers start from valid points near ``theta0``/MAP.
        Returns ``theta`` with shape (walkers, samples, ndim)."""
        from numpyro.infer import AIES
        self._refresh()
        ndim = len(self.names)
        nw = num_walkers or max(4 * ndim, 32)
        if theta0 is None:
            theta0 = self.results["map"]["theta"] if "map" in self.results else self.model.theta0()
        theta0 = self.clip_inside(theta0)
        lpf = jax.jit(lambda t: self.log_prob(t, **kw))
        inits = self._valid_jitter(theta0, nw, init_jitter, seed, lpf)
        vs = np.stack([np.asarray(self.theta_to_v(jnp.asarray(th))) for th in inits])
        init_params = {s: jnp.asarray([biject_to(c).inv(v[k]) for v in vs])
                       for k, (s, c) in enumerate(zip(self.site, self.constraints))}
        kernel = AIES(self.numpyro_model(**kw), moves=moves or {AIES.DEMove(): 0.5, AIES.StretchMove(): 0.5})
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=nw,
                    chain_method="vectorized", thinning=thinning, progress_bar=progress_bar)
        t0 = time.time()
        smp, _ = _run_chunked(mcmc, jax.random.PRNGKey(seed), init_params, num_samples, chunk, checkpoint,
                              (), verbose)
        theta = np.asarray(self._theta_from_sites(smp))
        out = dict(theta=theta, names=self.names, mcmc=mcmc, time=time.time() - t0)
        if verbose:
            print(f"[AIES] {nw} walkers x {num_samples} samples ({out['time']:.0f} s)")
        self.results["ensemble"] = out
        return out

    # ------------------------------------------------------------------
    def run(self, theta0=None, map_kw=None, laplace_kw=None, svi_kw=None, hmc_kw=None):
        """MAP -> Laplace -> SVI -> HMC; returns ``self.results``."""
        self.map(theta0, **(map_kw or {}))
        self.laplace(**(laplace_kw or {}))
        self.svi(**(svi_kw or {}))
        self.hmc(**(hmc_kw or {}))
        return self.results

    def to_arviz(self, stage="hmc"):
        """ArviZ InferenceData of the samples (requires arviz)."""
        import arviz as az
        r = self.results[stage]
        th = r["theta"] if r["theta"].ndim == 3 else r["theta"][None]
        post = {n: th[..., k] for k, n in enumerate(self.names)}
        return az.from_dict(posterior=post)


def _run_chunked(mcmc, key, init_params, num_samples, chunk, ckpt, extra_fields, verbose):
    """Warm-up, then sampling in chunks with an optional on-disk checkpoint.
    Returns (samples, extra_fields) grouped by chain, as numpy arrays."""
    import os
    import pickle
    if ckpt is None and chunk is None:
        mcmc.run(key, init_params=init_params, extra_fields=extra_fields)
        return (jax.device_get(mcmc.get_samples(group_by_chain=True)),
                jax.device_get(mcmc.get_extra_fields(group_by_chain=True)) if extra_fields else {})
    chunk = chunk or num_samples
    st = None
    if ckpt is not None and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        if verbose:
            print(f"[checkpoint] resuming from {ckpt}: {st['done']}/{num_samples} samples done")

    def save():
        if ckpt is not None:
            tmp = ckpt + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(st, f)
            os.replace(tmp, ckpt)

    if st is None:
        mcmc.warmup(key, init_params=init_params, extra_fields=extra_fields, collect_warmup=False)
        st = dict(state=jax.device_get(mcmc.post_warmup_state), done=0, samples=[], extras=[])
        save()
        if verbose:
            print("[checkpoint] warm-up finished" + (f", saved to {ckpt}" if ckpt else ""))
    while st["done"] < num_samples:
        n = min(chunk, num_samples - st["done"])
        mcmc.num_samples = n
        state = jax.tree_util.tree_map(jnp.asarray, st["state"])
        mcmc.post_warmup_state = state
        mcmc.run(state.rng_key, extra_fields=extra_fields)
        st["samples"].append(jax.device_get(mcmc.get_samples(group_by_chain=True)))
        st["extras"].append(jax.device_get(mcmc.get_extra_fields(group_by_chain=True)) if extra_fields else {})
        st["state"] = jax.device_get(mcmc.last_state)
        st["done"] += n
        save()
        if verbose and chunk < num_samples:
            print(f"[checkpoint] {st['done']}/{num_samples} samples")
    cat = lambda parts: {k: np.concatenate([np.asarray(p[k]) for p in parts], axis=1) for k in parts[0]}
    return cat(st["samples"]), (cat(st["extras"]) if extra_fields else {})


def diagnostics(result):
    """Split R-hat and effective sample size per parameter, plus divergences."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    th = result["theta"]
    if th.ndim == 2:
        th = th[None]
    ess = np.asarray(effective_sample_size(th))
    rhat = np.asarray(split_gelman_rubin(th)) if th.shape[0] > 1 else np.full(th.shape[-1], np.nan)
    return dict(names=result["names"], ess=ess, rhat=rhat,
                divergences=int(np.sum(result.get("diverging", 0))))


def angle_bounds(model, theta, extra=None):
    """Single-period sampling windows for orientation angles.

    Position angles are periodic (180 deg for ellipses, external shear,
    sources and PSF; 360/m deg for an m-th order multipole; 120 deg for
    'clus3'), so a [-180, 180] range contains several exact copies of every
    solution and the posterior is multimodal.  This returns
    ``{theta_name: (center - period/2, center + period/2)}`` for every free
    angle, centred on ``theta`` (e.g. the MAP); pass it as ``bounds`` to
    :class:`Pipeline`.  ``extra`` entries are merged in.
    """
    theta = np.asarray(theta, dtype=float)
    out = {}
    for k, n in enumerate(model.theta_names):
        period = None
        if n.startswith("lens") and n.endswith("_p6"):
            i = int(n[4:n.index("_")]) - 1
            mname = model.lens_models[i]
            if mname == "mpole":
                period = 360.0 / float(model.lens_params[i, 6])
            elif mname == "clus3":
                period = 120.0
            elif mname not in ("crline", "point", "gals"):
                period = 180.0
        elif n.startswith("ext") and n.endswith("_p6"):
            period = 180.0
        elif n in ("psf_p3", "psf_p7"):
            period = 180.0
        if period is not None:
            out[n] = (theta[k] - 0.5 * period, theta[k] + 0.5 * period)
    out.update(extra or {})
    return out
