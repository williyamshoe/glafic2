import numpy as np
import jax
import jax.numpy as jnp
import pytest

import glafic_jax as gj
from glafic_jax import cosmology as cm


def test_flat_lcdm_against_astropy():
    astropy = pytest.importorskip("astropy.cosmology")
    c = astropy.FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=0)
    for z1, z2 in [(0.0, 0.5), (0.0, 2.0), (0.5, 2.0), (0.3, 1.2)]:
        ref = c.angular_diameter_distance(z1, z2).value * 0.7 / gj.constants.COVERH_MPCH
        assert abs(float(cm.dis_angulard(z1, z2, (0.3, 0.7, -1.0, 0.7))) / ref - 1) < 1e-9


def test_open_closed_wcdm():
    astropy = pytest.importorskip("astropy.cosmology")
    for om, ol, w in [(0.3, 0.6, -1.0), (0.3, 0.8, -1.0), (0.25, 0.75, -0.8)]:
        c = astropy.wCDM(H0=70, Om0=om, Ode0=ol, w0=w, Tcmb0=0)
        ref = c.angular_diameter_distance(0.4, 1.7).value * 0.7 / gj.constants.COVERH_MPCH
        assert abs(float(cm.dis_angulard(0.4, 1.7, (om, ol, w, 0.7))) / ref - 1) < 1e-8


def test_distance_gradient():
    f = lambda o: cm.dis_angulard(0.5, 2.0, (o, 1.0 - o, -1.0, 0.7))
    g = jax.grad(f)(0.3)
    fd = (f(0.3 + 1e-6) - f(0.3 - 1e-6)) / 2e-6
    assert abs(float(g) / float(fd) - 1) < 1e-6


def test_w0wa_against_astropy():
    astropy = pytest.importorskip("astropy.cosmology")
    for om, ol, w0, wa in [(0.3, 0.7, -0.9, -0.5), (0.25, 0.7, -1.1, 0.4), (0.35, 0.65, -0.7, -1.2)]:
        c = astropy.w0waCDM(H0=70, Om0=om, Ode0=ol, w0=w0, wa=wa, Tcmb0=0)
        for z1, z2 in [(0.0, 0.234), (0.234, 1.5), (0.234, 9.51)]:
            ref = c.angular_diameter_distance(z1, z2).value * 0.7 / gj.constants.COVERH_MPCH
            assert abs(float(cm.dis_angulard(z1, z2, (om, ol, w0, 0.7, wa))) / ref - 1) < 1e-10


def test_wa_zero_is_wcdm():
    a = cm.dis_angulard(0.5, 2.0, (0.3, 0.7, -0.8, 0.7))
    b = cm.dis_angulard(0.5, 2.0, (0.3, 0.7, -0.8, 0.7, 0.0))
    assert float(a) == float(b)
