"""Optimisation and Hamiltonian Monte Carlo for glafic_jax models.

* :func:`optimize` -- gradient-based replacement of glafic's ``optimize``
  (downhill simplex) using SciPy L-BFGS-B / BFGS with exact JAX gradients.
* :func:`run_nuts` -- NUTS sampling of the posterior exp(-chi^2/2) with
  NumPyro (default) or BlackJAX.

The posterior is ``exp(-chi^2 / 2)`` where chi^2 includes glafic's priors
(``parprior``); parameters outside glafic's ranges get zero probability.
Flat priors are implied for parameters without a ``gauss`` prior, exactly as
in glafic's own ``mcmc`` command.
"""

import numpy as np
import jax
import jax.numpy as jnp


def _scales(model, theta):
    """Per-parameter scales used to condition the optimiser (similar in spirit
    to glafic's amoeba_dp_* step sizes)."""
    out = []
    for n, v in zip(model.theta_names, np.asarray(theta)):
        if (n.startswith(("lens", "ext")) and n.endswith(("_p3", "_p4"))) or n.startswith("point") and not n.endswith("zs"):
            out.append(0.1)                      # positions [arcsec]
        elif n.startswith(("lens", "ext")) and n.endswith("_p6"):
            out.append(10.0)                     # position angles [deg]
        elif n.startswith(("lens", "ext")) and n.endswith("_p5"):
            out.append(0.05)                     # ellipticities
        else:
            out.append(max(0.1 * abs(float(v)), 1e-2))
    return np.array(out)


def optimize(model, theta0=None, method="L-BFGS-B", maxiter=2000, tol=1e-12, restarts=10,
             verbose=False, **kw):
    """Minimise chi^2 with SciPy using JAX gradients.

    Extra keywords (``point_mode``, ``ext_mode``, ``n_sub``) are passed to
    :meth:`Model.chi2`.  Parameters are rescaled by their initial magnitude and
    the objective by its initial value; regions glafic penalises (wrong number
    of images, out-of-range parameters) are treated as a large finite barrier.
    The minimiser is restarted until chi^2 stops improving (like glafic's
    ``chi2_restart``).  Returns ``(theta_best, chi2_best, scipy_result)``.
    """
    from scipy.optimize import minimize
    theta0 = model.theta0() if theta0 is None else jnp.asarray(theta0)
    sc = _scales(model, theta0)
    pen = 0.5 * model.settings.chi2pen_range

    def chi2_of(u):
        return model.chi2(model.unflatten(u * sc), **kw)

    vg = jax.jit(jax.value_and_grad(chi2_of))
    u = np.asarray(theta0) / sc
    best = float(chi2_of(jnp.asarray(u)))
    if not np.isfinite(best) or best >= pen:
        raise ValueError("chi^2 is penalised at the initial point; start from a valid model")
    res = None
    for it in range(max(1, restarts)):
        norm = max(best, 1.0)

        def f(x):
            v, g = vg(jnp.asarray(x))
            v = float(v)
            g = np.asarray(g, dtype=float)
            if not np.isfinite(v) or v >= pen or not np.all(np.isfinite(g)):
                return 1e6, np.zeros_like(g)
            return v / norm, g / norm

        res = minimize(f, u, jac=True, method=method, tol=tol, options=dict(maxiter=maxiter))
        c2 = float(chi2_of(jnp.asarray(res.x)))
        if verbose:
            print(f"[optimize] pass {it + 1}: chi2 = {c2:.10g} ({res.message})")
        if c2 < pen and c2 < best:
            improved = best - c2 > 1e-8 * max(1.0, abs(best))
            u, best = res.x, c2
            if not improved:
                break
        else:
            break
    return jnp.asarray(u * sc), best, res


def run_nuts(model, theta0=None, num_warmup=500, num_samples=1000, num_chains=1, seed=0,
             backend="numpyro", target_accept=0.8, dense_mass=True, progress_bar=True,
             init_jitter=0.0, **kw):
    """Sample the posterior with NUTS.

    Returns a dict ``{"theta": (chains, samples, ndim), "names": [...],
    "log_prob": (chains, samples), "extra": backend specific}``.
    ``kw`` is forwarded to :meth:`Model.log_prob` (e.g. ``point_mode="tracked"``).

    Notes
    -----
    * Start near the maximum (e.g. from :func:`optimize`); like any HMC the
      sampler cannot cross regions where the image multiplicity changes.
    * Image-plane chi^2 with ``point_mode="glafic"`` uses a host callback, so
      chains are run sequentially; ``point_mode="tracked"`` or the
      source-plane chi^2 are pure JAX and can be vectorised.
    """
    theta0 = model.theta0() if theta0 is None else jnp.asarray(theta0)
    ndim = theta0.shape[0]
    logp = lambda t: model.log_prob(t, **kw)
    lp0 = float(jax.jit(logp)(theta0))
    if not np.isfinite(lp0):
        raise ValueError("log-probability is -inf at the initial point (outside prior ranges, wrong "
                         "number of images, ...); start from a valid model, e.g. the output of optimize()")
    key = jax.random.PRNGKey(seed)
    inits = theta0[None, :] + init_jitter * jax.random.normal(key, (num_chains, ndim)) * _scales(model, theta0)
    if backend == "numpyro":
        from numpyro.infer import MCMC, NUTS
        kernel = NUTS(potential_fn=lambda t: -logp(t), target_accept_prob=target_accept,
                      dense_mass=dense_mass)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method="sequential", progress_bar=progress_bar)
        mcmc.run(jax.random.PRNGKey(seed + 1), init_params=inits if num_chains > 1 else inits[0],
                 extra_fields=("potential_energy", "diverging"))
        th = mcmc.get_samples(group_by_chain=True)
        ex = mcmc.get_extra_fields(group_by_chain=True)
        return {"theta": np.asarray(th), "names": list(model.theta_names),
                "log_prob": -np.asarray(ex["potential_energy"]), "extra": mcmc}
    if backend == "blackjax":
        import blackjax
        outs, lps = [], []
        for c in range(num_chains):
            k1, k2, key = jax.random.split(jax.random.fold_in(key, c), 3)
            warm = blackjax.window_adaptation(blackjax.nuts, logp, target_acceptance_rate=target_accept,
                                              is_mass_matrix_diagonal=not dense_mass)
            (state, params), _ = warm.run(k1, inits[c], num_steps=num_warmup)
            step = jax.jit(blackjax.nuts(logp, **params).step)
            ks = jax.random.split(k2, num_samples)
            th, lp = [], []
            for k in ks:
                state, _ = step(k, state)
                th.append(np.asarray(state.position))
                lp.append(float(state.logdensity))
            outs.append(np.stack(th))
            lps.append(np.array(lp))
        return {"theta": np.stack(outs), "names": list(model.theta_names), "log_prob": np.stack(lps),
                "extra": None}
    raise ValueError("backend must be 'numpyro' or 'blackjax'")


def summary(result):
    """Mean / std / 16-50-84 percentiles of each parameter."""
    th = result["theta"].reshape(-1, result["theta"].shape[-1])
    q = np.percentile(th, [16, 50, 84], axis=0)
    lines = [f"{'parameter':28s} {'mean':>13s} {'std':>11s} {'16%':>13s} {'50%':>13s} {'84%':>13s}"]
    for k, n in enumerate(result["names"]):
        lines.append(f"{n:28s} {th[:, k].mean():13.6g} {th[:, k].std():11.4g} {q[0, k]:13.6g} {q[1, k]:13.6g} {q[2, k]:13.6g}")
    return "\n".join(lines)
