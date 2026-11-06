"""Differentiable cosmological distances (port of distance.c).

``cosmo`` is a tuple ``(omega, lambda, weos, hubble[, wa])`` of scalars
(``weos`` = w0 and ``wa`` the CPL evolution parameter, w0waCDM) (they
may be traced JAX values, so gradients w.r.t. cosmology work).  Distances are
in units of c/H0 exactly as in glafic.
"""

import jax.numpy as jnp

from .constants import (COVERH_MPCH, ARCSEC2RADIAN, FAC_CRITDENS, FAC_TDELAY_DAY,
                        OFFSET_TDELAY_FAC, TOL_CURVATURE)
from .quadrature import gauss_legendre

# 64-point Gauss-Legendre is accurate to <1e-10 for any sensible cosmology;
# GSL's cquad in the C code is run at epsrel = 1e-6.
_NQ = 64


def hubble_ez2(z, cosmo):
    """E(z)^2 for (o)wCDM / w0waCDM.

    ``cosmo = (omega, lambda, w0, hubble[, wa])`` with the CPL equation of
    state w(a) = w0 + wa (1 - a).  For wa = 0 (or a length-4 tuple) this is
    exactly glafic's hubble_ez2.
    """
    o, l, w = cosmo[0], cosmo[1], cosmo[2]
    zp = 1.0 + z
    de = zp ** (3.0 * (1.0 + w))
    if len(cosmo) > 4:
        wa = cosmo[4]
        de = de * zp ** (3.0 * wa) * jnp.exp(-3.0 * wa * z / zp)
    return (1.0 + o * z - l) * zp * zp + l * de


def hubble_ez(z, cosmo):
    return jnp.sqrt(hubble_ez2(z, cosmo))


def dis_chi(a, b, cosmo):
    """Radial comoving distance between redshifts a < b (units of c/H0)."""
    x, wq = gauss_legendre(_NQ)
    lo = 1.0 / (1.0 + jnp.asarray(b))
    hi = 1.0 / (1.0 + jnp.asarray(a))
    h = 0.5 * (hi - lo)
    aa = lo[..., None] + h[..., None] * (x + 1.0)
    f = 1.0 / (aa * aa * hubble_ez(1.0 / aa - 1.0, cosmo))
    return jnp.sum(f * wq, axis=-1) * h


def dis_comoving(a, b, cosmo):
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    chi = dis_chi(a, b, cosmo)
    k = cosmo[0] + cosmo[1] - 1.0
    flat = jnp.abs(k) < TOL_CURVATURE
    sk = jnp.sqrt(jnp.where(flat, 1.0, jnp.abs(k)))
    closed = jnp.sin(chi * sk) / sk
    opened = jnp.sinh(chi * sk) / sk
    r = jnp.where(flat, chi, jnp.where(k > 0.0, closed, opened))
    return jnp.where(a >= b, 0.0, r)


def dis_angulard(a, b, cosmo):
    return dis_comoving(a, b, cosmo) / (1.0 + jnp.asarray(b))


def dis_luminosity(a, b, cosmo):
    return dis_comoving(a, b, cosmo) * (1.0 + jnp.asarray(b)) / (1.0 + jnp.asarray(a))


def dis_mod(z, cosmo):
    return 25.0 + 5.0 * jnp.log10(COVERH_MPCH * dis_luminosity(0.0, z, cosmo))


def derivative_dis_proper(z, cosmo):
    return (1.0 / hubble_ez(z, cosmo)) / (1.0 + z)


def disratio(zl, zs_fid, zs, cosmo):
    d1 = dis_angulard(zl, zs_fid, cosmo) / dis_angulard(0.0, zs_fid, cosmo)
    d2 = dis_angulard(zl, zs, cosmo) / dis_angulard(0.0, zs, cosmo)
    return jnp.where(zs_fid <= zl, 0.0, d2 / jnp.where(zs_fid <= zl, 1.0, d1))


def tdelay_fac(zl, dos, dol, dls, cosmo):
    ddd = (1.0 + zl) * dos * dol / (dls + OFFSET_TDELAY_FAC)
    return FAC_TDELAY_DAY * ARCSEC2RADIAN * ARCSEC2RADIAN * ddd / cosmo[3]


def thetator_dis(theta, dis):
    return COVERH_MPCH * dis * ARCSEC2RADIAN * theta


def rtotheta_dis(r, dis):
    return r / (COVERH_MPCH * dis * ARCSEC2RADIAN)


def inv_sigma_crit_dis(dos, dol, dls):
    return dol * dls / (FAC_CRITDENS * dos)


def sigma_crit_dis(dos, dol, dls):
    return FAC_CRITDENS * dos / (dol * dls)


def critdensz(zl, cosmo):
    return hubble_ez2(zl, cosmo)


def omegaz(zl, cosmo):
    return cosmo[0] * (1.0 + zl) ** 3 / critdensz(zl, cosmo)


def delta_vir(zl, cosmo):
    o, l = cosmo[0], cosmo[1]
    ov = omegaz(zl, cosmo)
    ov2 = 1.0 / ov - 1.0
    case1 = (l > 0.0) & (l < 1.0) & (o > 0.0) & (o < 1.0)
    case2 = (o < 1.0) & (l < 1.0e-10) & (l > -1.0e-10)
    d1 = 177.6528 * (1.0 + 0.40929 * jnp.power(jnp.where(case1, ov2, 1.0), 0.90524))
    ov3 = 1.0 + 2.0 * jnp.where(case2, ov2, 1.0)
    eta = jnp.log(ov3 + jnp.sqrt(ov3 * ov3 - 1.0))
    f = (jnp.exp(eta) - jnp.exp(-eta)) / 2.0 - eta
    d2 = 315.827 * ov2 ** 3 / jnp.where(case2, f * f, 1.0)
    return jnp.where(case1, d1, jnp.where(case2, d2, 177.6528))


def deltaomega(zl, cosmo, flag_hodensity=0, hodensity=200.0):
    """Halo overdensity times mean density (static flag)."""
    if flag_hodensity == 1:
        return hodensity * cosmo[0] * (1.0 + zl) ** 3
    if flag_hodensity == 2:
        return hodensity * critdensz(zl, cosmo)
    return delta_vir(zl, cosmo) * cosmo[0] * (1.0 + zl) ** 3
