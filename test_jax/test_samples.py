"""End-to-end tests on the sample input files."""
import os

import numpy as np
import pytest
import jax

import glafic_jax as gj
from conftest import SAMPLES, c_glafic

POINT = os.path.join(SAMPLES, "point.input")


@pytest.fixture(scope="module")
def point_model():
    m = gj.read_input(POINT)
    m.set_secondary("hvary 0")
    m.model_init()
    return m


def test_point_images_against_c(point_model):
    glafic = c_glafic()
    if glafic is None:
        pytest.skip("C glafic not available")
    glafic.init(0.3, 0.7, -1.0, 0.7, "out", -5.0, -5.0, 5.0, 5.0, 0.02, 0.5, 5, verb=0)
    glafic.startup_setnum(2, 0, 1)
    glafic.set_lens(1, "sie", 0.5, 300.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0)
    glafic.set_lens(2, "pert", 0.5, 2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)
    glafic.set_point(1, 2.0, -0.15, 0.05)
    glafic.model_init(verb=0)
    c = np.array(glafic.point_solve(2.0, -0.15, 0.05, verb=0))
    glafic.quit()
    j = point_model.point_solve(2.0, -0.15, 0.05)
    assert c.shape == j.shape
    assert np.allclose(c, j, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("splane", [0, 1])
def test_point_chi2_against_c(splane):
    glafic = c_glafic()
    if glafic is None:
        pytest.skip("C glafic not available")
    glafic.init(0.3, 0.7, -1.0, 0.7, "out", -5.0, -5.0, 5.0, 5.0, 0.02, 0.5, 5, verb=0)
    for l in ["chi2_splane %d" % splane, "chi2_checknimg 1", "chi2_usemag 0", "hvary 0"]:
        glafic.set_secondary(l, verb=0)
    glafic.startup_setnum(2, 0, 1)
    glafic.set_lens(1, "sie", 0.5, 300.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0)
    glafic.set_lens(2, "pert", 0.5, 2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)
    glafic.set_point(1, 2.0, -0.15, 0.05)
    glafic.setopt_point(1, 0, 0, 0)  # C would otherwise optimise the source position
    glafic.model_init(verb=0)
    glafic.readobs_point(os.path.join(SAMPLES, "obs_point.dat"), verb=0)
    glafic.parprior(os.path.join(SAMPLES, "prior_point.dat"), verb=0)
    c2c = glafic.c2calc()
    glafic.quit()
    m = gj.read_input(POINT)
    m.set_secondary("hvary 0")
    m.set_secondary("chi2_splane %d" % splane)
    m.model_init()
    assert abs(float(m.chi2()) / c2c - 1) < 1e-8


@pytest.mark.parametrize("splane,mode", [(1, "glafic"), (0, "glafic"), (0, "tracked")])
def test_point_gradient(point_model, splane, mode):
    m = point_model
    m.set_secondary("chi2_splane %d" % splane)
    m.model_init()
    # a point near the best fit (all modes valid)
    th = np.array([321.157, 0.00909, -0.00270, 0.30136, 21.458, 0.09691, 91.978, 0.048119, -0.021239])
    f = jax.jit(lambda t: m.log_prob(t, point_mode=mode))
    g = np.asarray(jax.jit(jax.grad(lambda t: m.log_prob(t, point_mode=mode)))(th))
    assert np.isfinite(float(f(th)))
    eps = 1e-7 * np.maximum(1, np.abs(th))
    fd = np.array([(float(f(th + eps[k] * np.eye(9)[k])) - float(f(th - eps[k] * np.eye(9)[k]))) / (2 * eps[k])
                   for k in range(9)])
    assert np.all(np.abs(g - fd) <= 1e-4 * np.abs(fd) + 1e-3)
    m.set_secondary("chi2_splane 0")
    m.model_init()


EXT = [("quadhost.input", "obs_quadhost.fits", None, 2.60858569e+05),
       ("multilens_extend.input", "obs_multilens_extend.fits", "obs_psf_multilens_extend.fits", 3.27862445e+05),
       ("extend.input", None, None, 1.646e+05)]


@pytest.mark.parametrize("name,obs,psf,c2ref", EXT)
def test_extend_chi2(name, obs, psf, c2ref):
    """chi^2 of the sample extended-source models (reference values from C glafic c2calc)."""
    pytest.importorskip("astropy")
    m = gj.read_input(os.path.join(SAMPLES, name), run_commands=(obs is None))
    if psf:
        m.readpsf(os.path.join(SAMPLES, psf))
    if obs:
        m.readobs_extend(os.path.join(SAMPLES, obs))
    m.model_init()
    tol = 1e-3 if name == "extend.input" else 1e-6   # C prints only 4 digits for the masked case
    assert abs(float(jax.jit(m.chi2)(m.params())) / c2ref - 1) < tol
