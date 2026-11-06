"""Radial lookup tables for the gNFW and Einasto profiles (gnfw_tab.c, ein_tab.c).

glafic (with the default ``gnfw_usetab = ein_usetab = 1``) tabulates
ln kappa(x), ln(-dkappa/dx)(x) and ln dphi/dx(x) on a grid of (alpha, ln x) and
bilinearly interpolates.  The tables are computed here once with NumPy (with
more accurate quadrature than the C Romberg rules) and cached on disk.  The
JAX-side interpolation is differentiable in both x and alpha.
"""

import functools
import os

import numpy as np
import jax.numpy as jnp
from scipy import special as sps

NUM_LNX, LNX_MIN, DLNX = 801, -23.0, 0.0575
GNFW_NUM_ALP, GNFW_ALP_MIN, GNFW_DALP = 101, 0.0, 0.02
EIN_NUM_ALP, EIN_ALP_MIN, EIN_DALP = 99, 0.02, 0.01
OFFSET_LOG = 1.0e-300
_VERSION = 2


def _cgl(a, b, panels, nodes):
    """Composite Gauss-Legendre nodes/weights on [a, b] (arrays broadcast)."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    e = np.linspace(0.0, 1.0, panels + 1)
    t = np.concatenate([0.5 * (e[i + 1] - e[i]) * x + 0.5 * (e[i + 1] + e[i]) for i in range(panels)])
    wt = np.concatenate([0.5 * (e[i + 1] - e[i]) * w for i in range(panels)])
    a = np.asarray(a)[..., None]
    b = np.asarray(b)[..., None]
    return a + (b - a) * t, (b - a) * wt


# --------------------------------------------------------------------------
# gNFW


def _gnfw_kappa(x, alp):
    t, w = _cgl(-25.0, 25.0, 50, 10)
    xx = np.exp(t)
    a = np.sqrt(x[..., None] ** 2 + xx * xx)
    aa = 1.0 + a
    f = xx / (aa ** 3 * (a / aa) ** alp)
    return 0.5 * np.sum(f * w, axis=-1)


def _gnfw_dkappa(x, alp):
    t, w = _cgl(-25.0, 25.0, 50, 10)
    xx = np.exp(t)
    xs = x[..., None]
    a = np.sqrt(xs ** 2 + xx * xx)
    aa = 1.0 + a
    f = xs * xx * (-3.0 * a - alp) / (a * a * (a / aa) ** alp * aa ** 4)
    return 0.5 * np.sum(f * w, axis=-1)


def _hgnfw(x, alp):
    lx = np.log(x)
    t, w = _cgl(lx - 40.0, lx, 40, 10)
    s = np.exp(t)
    f = s * (s / (1.0 + s)) ** (2.0 - alp) / (1.0 + s)
    return np.sum(f * w, axis=-1)


def _gnfw_dphi(x, alp):
    xs = x[..., None]
    # s in [0, 0.5]: log spacing ; s in [0.5, 1]: s = sin(theta)
    t, w = _cgl(-40.0, np.log(0.5), 40, 10)
    s = np.exp(t)
    f1 = np.sum(s * (s + xs) ** (alp - 3.0) * (1.0 - np.sqrt(1.0 - s * s)) / s * w, axis=-1)
    th, w2 = _cgl(np.pi / 6.0, np.pi / 2.0, 8, 16)
    s = np.sin(th)
    f2 = np.sum((s + xs) ** (alp - 3.0) * (1.0 - np.cos(th)) / s * np.cos(th) * w2, axis=-1)
    return _hgnfw(x, alp) / x + x ** (2.0 - alp) * (f1 + f2)


# --------------------------------------------------------------------------
# Einasto


def _log_hein(x, alp):
    a3 = 3.0 / alp
    p = sps.gammainc(a3, 2.0 * np.power(x, alp) / alp)
    return a3 * np.log(0.5 * alp) + sps.gammaln(a3) + np.log(p + 1e-300) - np.log(alp)


def _ein_kappa(x, alp):
    t, w = _cgl(-25.0, 25.0, 50, 10)
    xx = np.exp(t)
    yy = x[..., None] ** 2 + xx * xx
    ff = -2.0 * yy ** (0.5 * alp) / alp
    return 0.5 * np.sum(np.exp(ff) * xx * w, axis=-1)


def _ein_dkappa(x, alp):
    t, w = _cgl(-25.0, 25.0, 50, 10)
    xx = np.exp(t)
    xs = x[..., None]
    yy = xs ** 2 + xx * xx
    pp = yy ** (0.5 * alp)
    ff = -2.0 * pp / alp
    return 0.5 * np.sum(np.exp(ff) * xx * (-2.0) * (pp / yy) * xs * w, axis=-1)


def _ein_dphi(x, alp):
    t, w = _cgl(-30.0, 0.0, 30, 10)
    s = np.exp(t)
    xs = x[..., None]
    xx = np.sqrt(1.0 + s * s)
    h1 = np.exp(_log_hein(xs * xx, alp))
    h2 = np.exp(_log_hein(xs * xx / s, alp))
    f = s * (h1 + s * h2) / ((1.0 + s * s) * xx)
    return np.sum(f * w, axis=-1) / x


# --------------------------------------------------------------------------


def _build(kind):
    lnx = LNX_MIN + DLNX * np.arange(NUM_LNX)
    x = np.exp(lnx)
    if kind == "gnfw":
        alps = GNFW_ALP_MIN + GNFW_DALP * np.arange(GNFW_NUM_ALP)
        fk, fd, fp = _gnfw_kappa, _gnfw_dkappa, _gnfw_dphi
    else:
        alps = EIN_ALP_MIN + EIN_DALP * np.arange(EIN_NUM_ALP)
        fk, fd, fp = _ein_kappa, _ein_dkappa, _ein_dphi
    tk = np.empty((alps.size, NUM_LNX))
    td = np.empty_like(tk)
    tp = np.empty_like(tk)
    with np.errstate(all="ignore"):
        for i, a in enumerate(alps):
            tk[i] = np.log(fk(x, a) + OFFSET_LOG)
            td[i] = np.log(-fd(x, a) + OFFSET_LOG)
            tp[i] = np.log(fp(x, a) + OFFSET_LOG)
    return np.stack([tk, td, tp])


def _cache_dir():
    d = os.environ.get("GLAFIC_JAX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "glafic_jax"))
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return d


@functools.lru_cache(maxsize=None)
def get_table(kind):
    d = _cache_dir()
    fn = None if d is None else os.path.join(d, f"{kind}_table_v{_VERSION}.npy")
    if fn is not None and os.path.exists(fn):
        return np.load(fn)
    tab = _build(kind)
    if fn is not None:
        try:
            np.save(fn, tab)
        except OSError:
            pass
    return tab


def _intpol(tab, lnx, alp, alp_min, dalp, nalp):
    """Bilinear interpolation identical in spirit to intpol_*_lin in the C code."""
    tab = jnp.asarray(tab)
    fi = jnp.trunc((alp - alp_min) / dalp)
    fj = jnp.trunc((lnx - LNX_MIN) / DLNX)
    i = jnp.clip(fi, 0, nalp - 2).astype(jnp.int32)
    j = jnp.clip(fj, 0, NUM_LNX - 2).astype(jnp.int32)
    alp_c = jnp.where(fi < 0, alp_min, alp)
    lnx_c = jnp.where(fj < 0, LNX_MIN, lnx)
    t = (alp_c - (alp_min + i * dalp)) / dalp
    u = (lnx_c - (LNX_MIN + j * DLNX)) / DLNX
    u = jnp.minimum(u, 1.0)
    t = jnp.minimum(t, 1.0)
    return ((1.0 - t) * (1.0 - u) * tab[i, j] + t * (1.0 - u) * tab[i + 1, j]
            + t * u * tab[i + 1, j + 1] + (1.0 - t) * u * tab[i, j + 1])


def gnfw_radial(x, alp):
    """Return (kappa, dkappa, dphi) of the gNFW profile (dimensionless)."""
    tab = get_table("gnfw")
    lx = jnp.log(x)
    k = jnp.exp(_intpol(tab[0], lx, alp, GNFW_ALP_MIN, GNFW_DALP, GNFW_NUM_ALP))
    d = -jnp.exp(_intpol(tab[1], lx, alp, GNFW_ALP_MIN, GNFW_DALP, GNFW_NUM_ALP))
    p = jnp.exp(_intpol(tab[2], lx, alp, GNFW_ALP_MIN, GNFW_DALP, GNFW_NUM_ALP))
    return k, d, p


def ein_radial(x, alp):
    tab = get_table("ein")
    lx = jnp.log(x)
    k = jnp.exp(_intpol(tab[0], lx, alp, EIN_ALP_MIN, EIN_DALP, EIN_NUM_ALP))
    d = -jnp.exp(_intpol(tab[1], lx, alp, EIN_ALP_MIN, EIN_DALP, EIN_NUM_ALP))
    p = jnp.exp(_intpol(tab[2], lx, alp, EIN_ALP_MIN, EIN_DALP, EIN_NUM_ALP))
    return k, d, p
