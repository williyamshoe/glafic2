import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import glafic_jax as gj
from glafic_jax import point as pt
from conftest import SAMPLES

BEST = np.array([321.157, 0.00909, -0.00270, 0.30136, 21.458, 0.09691, 91.978, 0.048119, -0.021239])


def _model(position_only=False):
    m = gj.read_input(os.path.join(SAMPLES, "point.input"))
    m.set_secondary("hvary 0")
    m.set_secondary("chi2_splane 1")
    if position_only:  # drop flux / time-delay constraints
        for k in range(m.num_poi):
            m.obs_point[k]["tab"][:, 4] = 0.0
            m.obs_point[k]["tab"][:, 6] = 0.0
    m.model_init()
    return m


def test_source_marginalisation_matches_numerical_integral():
    """chi2_min + ln det M equals -2 ln of the integral of exp(-chi2/2) over the
    source position (up to the constant 2 ln 2pi), for position-only data."""
    m = _model(position_only=True)
    P = m.unflatten(jnp.asarray(BEST))
    marg = pt.chi2_point_batched(m, P, "splane", return_terms=True, source_mode="marginalize")
    prof = pt.chi2_point_batched(m, P, "splane", return_terms=True, source_mode="profile")
    xy0 = np.asarray(prof[5][0])
    g = np.linspace(-0.02, 0.02, 201)
    X, Y = np.meshgrid(xy0[0] + g, xy0[1] + g)
    f = jax.jit(jax.vmap(lambda x, y: pt.chi2_point_batched(
        m, dict(P, point=P["point"].at[0, 1].set(x).at[0, 2].set(y)), "splane")))
    c2 = np.asarray(f(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    integral = np.sum(np.exp(-0.5 * (c2 - c2.min()))) * (g[1] - g[0]) ** 2
    num = c2.min() - 2.0 * np.log(integral)
    ana = float(marg[0][0]) - 2.0 * np.log(2.0 * np.pi)
    assert abs(num - ana) < 1e-3


def test_pipeline_map_svi_hmc():
    pytest.importorskip("numpyro")
    m = _model(position_only=True)
    m.set_source_mode("profile")
    r = gj.Pipeline(m).map(jnp.asarray(BEST[:7]), verbose=False)
    assert r["chi2"] < 1.0           # 4 images, positions only: nearly exact fit
    m.set_source_mode("marginalize")
    pipe = gj.Pipeline(m, reparam={"lens1_sie_p2": "log10"}, bounds={"lens1_sie_p2": (100.0, 1000.0)})
    pipe.set_map(r["theta"])
    lap = pipe.laplace(num_draws=200, verbose=False)
    assert np.all(np.linalg.eigvalsh(lap["cov_z"]) > 0)
    s = pipe.svi(num_steps=300, num_draws=200, verbose=False)
    assert np.all(np.isfinite(s["theta"]))
    h = pipe.hmc(num_warmup=150, num_samples=100, num_chains=2, progress_bar=False, verbose=False)
    assert h["theta"].shape == (2, 100, 7)
    assert np.all(np.isfinite(h["log_prob"]))
    assert h["diverging"].mean() < 0.05
    # posterior mean of the velocity dispersion close to the best fit
    assert abs(h["theta"][..., 0].mean() - r["theta"][0]) < 10.0


def test_reparam_roundtrip():
    m = _model(position_only=True)
    m.set_source_mode("marginalize")
    pipe = gj.Pipeline(m, reparam={"lens1_sie_p2": "log10", "lens1_sie_p6": ("offset", "lens1_sie_p5", -10.0, 10.0)},
                       bounds={"lens1_sie_p2": (100.0, 1000.0)})
    th = pipe.clip_inside(jnp.asarray(BEST[:7]))
    assert np.allclose(np.asarray(pipe.to_theta(pipe.to_z(th))), np.asarray(th), rtol=1e-12, atol=1e-12)
    assert "000_log10_lens1_sie_p2" in pipe.site
