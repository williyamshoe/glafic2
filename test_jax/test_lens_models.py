import numpy as np
import pytest
import jax
import jax.numpy as jnp

import glafic_jax as gj
from conftest import c_glafic

XS = np.array([0.3, -1.2, 2.5, 0.7, -0.4, 15.0, 0.31])
YS = np.array([0.8, 0.4, -1.7, -0.9, 3.1, -8.0, 1.13])
CASES = {
    "sie": [0.5, 300.0, 0.1, -0.05, 0.35, 30.0, 0.05, 0.0],
    "jaffe": [0.5, 250.0, 0.1, -0.05, 0.3, 30.0, 3.0, 0.1],
    "point": [0.5, 1e12, 0.1, -0.05, 0.0, 0.0, 0.0, 0.0],
    "pert": [0.5, 2.0, 0.1, -0.05, 0.05, 60.0, 0.0, 0.02],
    "clus3": [0.5, 2.0, 0.1, -0.05, 0.02, 20.0, 0.0, 0.0],
    "mpole": [0.5, 2.0, 0.1, -0.05, 0.03, 20.0, 4.0, 1.0],
    "nfwpot": [0.5, 1e14, 0.1, -0.05, 0.2, 30.0, 8.0, 0.0],
    "nfw": [0.5, 1e14, 0.1, -0.05, 0.3, 30.0, 8.0, 0.0],
    "anfw": [0.5, 1e14, 0.1, -0.05, 0.3, 30.0, 8.0, 0.0],
    "acnfw": [0.5, 1e14, 0.1, -0.05, 0.3, 30.0, 8.0, 0.02],
    "hernpot": [0.5, 1e12, 0.1, -0.05, 0.2, 30.0, 1.0, 0.0],
    "hern": [0.5, 1e12, 0.1, -0.05, 0.3, 30.0, 1.0, 0.0],
    "ahern": [0.5, 1e12, 0.1, -0.05, 0.3, 30.0, 1.0, 0.0],
    "powpot": [0.5, 2.0, 0.1, -0.05, 0.2, 30.0, 1.2, 2.1],
    "pow": [0.5, 2.0, 0.1, -0.05, 0.3, 30.0, 1.2, 2.1],
    "gnfwpot": [0.5, 1e14, 0.1, -0.05, 0.2, 30.0, 8.0, 1.3],
    "gnfw": [0.5, 1e14, 0.1, -0.05, 0.3, 30.0, 8.0, 1.3],
    "serspot": [0.5, 1e12, 0.1, -0.05, 0.2, 30.0, 1.0, 3.0],
    "sers": [0.5, 1e12, 0.1, -0.05, 0.3, 30.0, 1.0, 3.0],
    "tnfwpot": [0.5, 1e14, 0.1, -0.05, 0.2, 30.0, 8.0, 5.0],
    "tnfw": [0.5, 1e14, 0.1, -0.05, 0.3, 30.0, 8.0, 5.0],
    "einpot": [0.5, 1e14, 0.1, -0.05, 0.2, 30.0, 8.0, 0.2],
    "ein": [0.5, 1e14, 0.1, -0.05, 0.3, 30.0, 8.0, 0.2],
    "crline": [0.5, 2.0, 0.1, -0.05, 0.0, 30.0, 0.1, 0.4],
    "gaupot": [0.5, 2.0, 0.1, -0.05, 0.2, 30.0, 1.0, 0.5],
}
# models evaluated with numerical integrals / tables: C uses Romberg with
# tolerance ~5e-4, so agreement is at that level
NUMERICAL = {"nfw", "hern", "sers", "tnfw", "ein", "einpot", "gnfw", "gnfwpot"}
S = gj.default_settings(xmin=-5, ymin=-5, xmax=5, ymax=5, pix_ext=0.02, pix_poi=0.5)


def jax_eval(name, p, x=XS, y=YS, **kw):
    s = gj.default_settings(**kw) if kw else S
    o = gj.LensSystem([name], [p[0]], s).ray_shoot(jnp.asarray(x), jnp.asarray(y), jnp.asarray([p]), 2.0, s.cosmo)
    return np.stack([o.ax, o.ay, o.td, o.kap, o.gam1, o.gam2, o.muinv, o.rot], axis=1)


@pytest.mark.parametrize("name", sorted(CASES))
def test_against_c(name):
    glafic = c_glafic()
    if glafic is None:
        pytest.skip("C glafic not available (set GLAFIC_C_PATH)")
    p = CASES[name]
    glafic.init(0.3, 0.7, -1.0, 0.7, "out", -5.0, -5.0, 5.0, 5.0, 0.02, 0.5, 5, verb=0)
    glafic.startup_setnum(1, 0, 0)
    glafic.set_lens(1, name, *p)
    glafic.model_init(verb=0)
    c = np.array([glafic.calcimage(2.0, x, y, verb=0) for x, y in zip(XS, YS)])
    glafic.quit()
    j = jax_eval(name, p)
    tol = 3e-3 if name in NUMERICAL else (1e-7 if name == "pow" else 1e-10)  # pow: C truncates TM15 series at 1e-8
    err = np.abs(j - c).max(axis=0) / np.maximum(np.abs(c).max(axis=0), 1e-12)
    assert err.max() < tol, err


@pytest.mark.parametrize("dens,pot,p", [
    ("nfw", "nfwpot", [0.5, 1e14, 0.1, -0.05, 0.0, 30, 8, 0]),
    ("hern", "hernpot", [0.5, 1e12, 0.1, -0.05, 0.0, 30, 1, 0]),
    ("sers", "serspot", [0.5, 1e12, 0.1, -0.05, 0.0, 30, 1, 3]),
    ("tnfw", "tnfwpot", [0.5, 1e14, 0.1, -0.05, 0.0, 30, 8, 5]),
    ("gnfw", "gnfwpot", [0.5, 1e14, 0.1, -0.05, 0.0, 30, 8, 1.3]),
    ("anfw", "nfwpot", [0.5, 1e14, 0.1, -0.05, 0.0, 30, 8, 0]),
])
def test_spherical_limit(dens, pot, p):
    """For e = 0 the elliptical-density models must equal the potential ones."""
    a = jax_eval(dens, p, ell_accurate=1)
    b = jax_eval(pot, p, ell_accurate=1)
    for k in (0, 1, 3):  # ax, ay, kappa
        assert np.allclose(a[:, k], b[:, k], rtol=2e-4, atol=1e-8 * np.abs(b[:, k]).max())


@pytest.mark.parametrize("name", ["sie", "nfw", "pow", "sers", "gnfw", "anfw", "tnfw", "ein", "mpole"])
def test_hessian_consistency(name):
    """kappa/gamma returned by the model equal autodiff derivatives of alpha."""
    p = jnp.asarray([CASES[name]])
    ls = gj.LensSystem([name], [CASES[name][0]], S)

    def alpha(xy):
        o = ls.ray_shoot(xy[0], xy[1], p, 2.0, S.cosmo)
        return jnp.stack([o.ax, o.ay])

    for x, y in zip(XS[:5], YS[:5]):
        J = jax.jacfwd(alpha)(jnp.array([x, y]))
        o = ls.ray_shoot(x, y, p, 2.0, S.cosmo)
        tol = 2e-3 * float(abs(o.kap) + abs(o.gam1) + abs(o.gam2))
        assert abs(0.5 * (J[0, 0] + J[1, 1]) - o.kap) < tol
        assert abs(0.5 * (J[0, 0] - J[1, 1]) - o.gam1) < tol
        assert abs(J[0, 1] - o.gam2) < tol


@pytest.mark.parametrize("name", ["sie", "nfw", "pow", "sers", "gnfw", "acnfw", "tnfw"])
def test_param_gradients_finite(name):
    p0 = jnp.asarray(CASES[name])
    ls = gj.LensSystem([name], [CASES[name][0]], S)
    f = lambda p: jnp.sum(jnp.stack(ls.ray_shoot(jnp.asarray(XS), jnp.asarray(YS), p[None], 2.0, S.cosmo)[:7]))
    g = jax.grad(f)(p0)
    assert np.all(np.isfinite(np.asarray(g)))
    k = 1
    e = 1e-6 * abs(float(p0[k]))
    fd = (f(p0.at[k].add(e)) - f(p0.at[k].add(-e))) / (2 * e)
    assert abs(float(g[k]) - float(fd)) < 1e-4 * abs(float(fd)) + 1e-8
