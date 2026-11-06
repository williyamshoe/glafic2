"""Lens mass models (port of mass.c, app_ell.c and app_cnfw.c).

Every model is a function ``f(tx, ty, p, ctx)`` where

* ``tx, ty`` are image-plane positions in arcsec (arrays, broadcasting),
* ``p`` is the length-8 parameter vector of the lens exactly as in glafic
  (``p[0]`` = z_lens, ``p[1..7]`` model parameters),
* ``ctx`` is a :class:`PlaneContext` holding distances of that lens plane for
  the current source redshift,

and returns ``(ax, ay, phi, kap, gam1, gam2)``.  All branches of the C code
are reproduced with NaN-safe ``jnp.where`` constructs so that the functions
are jittable and differentiable with respect to positions *and* parameters.
"""

import os
from typing import NamedTuple, Any

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln, expi

from . import cosmology as cosmo_mod
from .constants import (ARCSEC2RADIAN, COVERH_MPCH, C_LIGHT_KMS, MPC2METER, R_SCHWARZ,
                        NFW_RS_NORM, NFW_B_NORM, ULIM_JHK, LENS_MODELS)
from .quadrature import composite_gl, qng21
from . import tables

D2R = np.pi / 180.0
TINY = 1.0e-300


class PlaneContext(NamedTuple):
    zl: Any
    dos: Any
    dol: Any
    dls: Any
    delome: Any
    cosmo: Any
    settings: Any          # static glafic Settings
    gals: Any = None       # galaxy catalogue (N, 5) for model 'gals'


# ---------------------------------------------------------------------------
# small helpers


def ssqrt(x):
    """sqrt that has a finite gradient at 0."""
    return jnp.sqrt(jnp.where(x > 0.0, x, TINY))


def _sc(ang_deg):
    return jnp.sin(ang_deg * D2R), jnp.cos(ang_deg * D2R)


def fac_pert(zs_fid, ctx):
    d = cosmo_mod.dis_angulard(ctx.zl, zs_fid, ctx.cosmo) / cosmo_mod.dis_angulard(0.0, zs_fid, ctx.cosmo)
    return (ctx.dls / ctx.dos) / d


def thetator(theta, ctx):
    return cosmo_mod.thetator_dis(theta, ctx.dol)


def rtotheta(r, ctx):
    return cosmo_mod.rtotheta_dis(r, ctx.dol)


def inv_sigma_crit(ctx):
    return cosmo_mod.inv_sigma_crit_dis(ctx.dos, ctx.dol, ctx.dls)


def _kapgam_from_hessian(pxx, pxy, pyy):
    return 0.5 * (pxx + pyy), 0.5 * (pxx - pyy), pxy


def _rotate_back(ax_r, ay_r, pxx, pxy, pyy, si, co):
    """Rotate deflection/hessian computed in a frame rotated by (si, co)."""
    ax = ax_r * co + ay_r * si
    ay = -ax_r * si + ay_r * co
    rpxx = co * co * pxx + 2.0 * co * si * pxy + si * si * pyy
    rpyy = si * si * pxx - 2.0 * co * si * pxy + co * co * pyy
    rpxy = si * co * (pyy - pxx) + (co * co - si * si) * pxy
    return ax, ay, rpxx, rpxy, rpyy


# ---------------------------------------------------------------------------
# elliptical potential (u_calc)


def u_calc(dx, dy, e, si, co, smallcore):
    si2 = 2.0 * si * co
    co2 = co * co - si * si
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    u0 = ssqrt((1.0 + e) * ddx * ddx + (1.0 - e) * ddy * ddy) + smallcore
    u1 = (dx + e * (dx * co2 - dy * si2)) / u0
    u2 = (dy - e * (dy * co2 + dx * si2)) / u0
    u3 = (1.0 + e * co2 - u1 * u1) / u0
    u4 = (-e * si2 - u1 * u2) / u0
    u5 = (1.0 - e * co2 - u2 * u2) / u0
    return u0, u1, u2, u3, u4, u5


def ellpot(tx, ty, cx, cy, tt, e, pa, bb, dphi_f, ddphi_f, phi_f, ctx):
    si, co = _sc(-pa)
    u0, u1, u2, u3, u4, u5 = u_calc((tx - cx) / tt, (ty - cy) / tt, e, si, co, ctx.settings.smallcore)
    a = bb * dphi_f(u0)
    b = bb * ddphi_f(u0)
    ax = a * u1 * tt
    ay = a * u2 * tt
    pxx = b * u1 * u1 + a * u3
    pxy = b * u1 * u2 + a * u4
    pyy = b * u2 * u2 + a * u5
    kap, g1, g2 = _kapgam_from_hessian(pxx, pxy, pyy)
    phi = bb * phi_f(u0) * tt * tt
    return ax, ay, phi, kap, g1, g2


def ddphi_kappadphi(kappa, dphi, x):
    return 2.0 * kappa - dphi / x


# ---------------------------------------------------------------------------
# elliptical density: integrals J_n, K_n, I (ell_integ_* in mass.c)


def elldens(tx, ty, cx, cy, tt, q, pa, bb, radial, lowlim, ctx):
    """Generic elliptical-density lens.

    ``radial(se)`` returns (kappa, dkappa, dphi) of the dimensionless radial
    profile; ``lowlim`` is ``'std'`` or ``'steep'`` selecting how the lower
    integration limit is chosen (exactly as in the C code).
    """
    s = ctx.settings
    sc = s.smallcore
    si, co = _sc(-pa)
    dx = tx - cx
    dy = ty - cy
    bx = (co * dx - si * dy) / tt
    by = (si * dx + co * dy) / tt

    xi2_1 = by * by + bx * bx + sc * sc
    uu = 1.0 / xi2_1
    if lowlim == "std":
        if getattr(s, "ell_accurate", 0):
            lo = 1.0e-10 * jnp.minimum(uu, 1.0)   # continuous lower limit
        else:
            lo = jnp.where(uu > 0.1, ULIM_JHK, 1.0e-4 * uu)
    else:
        lo = sc * sc * uu
    lnlo = jnp.log(lo)
    return _ell_core(bx, by, lnlo, q, si, co, bb, tt, radial, s)


# ---------------------------------------------------------------------------
# radial helper functions


def func_hern_dl(x):
    gt = x > 1.0 + 1.0e-9
    lt = (x < 1.0 - 1.0e-9) & (x > 1.0e-5)
    tiny = x <= 1.0e-5
    xg = jnp.where(gt, x, 2.0)
    xl = jnp.where(lt, x, 0.5)
    xt = jnp.where(tiny, x, 1.0e-6)
    s1 = jnp.sqrt(xg * xg - 1.0)
    s2 = jnp.sqrt(1.0 - xl * xl)
    return jnp.where(gt, jnp.arctan(s1) / s1,
                     jnp.where(lt, jnp.arctanh(s2) / s2,
                               jnp.where(tiny, jnp.log(2.0 / xt), 1.0)))


# NFW ---------------------------------------------------------------------

def kappa_nfw_dl(x):
    gt = x > 1.0 + 1.0e-6
    lt = x < 1.0 - 1.0e-6
    xg = jnp.where(gt, x, 2.0)
    xl = jnp.where(lt, x, 0.5)
    fg = 0.5 * (1.0 - 2.0 * jnp.arctan(jnp.sqrt((xg - 1.0) / (xg + 1.0))) / jnp.sqrt(xg * xg - 1.0)) / (xg * xg - 1.0)
    fl = 0.5 * (2.0 * jnp.arctanh(jnp.sqrt((1.0 - xl) / (xl + 1.0))) / jnp.sqrt(1.0 - xl * xl) - 1.0) / (1.0 - xl * xl)
    return jnp.where(gt, fg, jnp.where(lt, fl, 0.5 / 3.0))


def dkappa_nfw_dl(x):
    ok = (x > 1.0 + 1.0e-5) | (x < 1.0 - 1.0e-5)
    xs = jnp.where(ok, x, 2.0)
    f = 0.5 * (3.0 * xs * xs * func_hern_dl(xs) - 2.0 * xs * xs - 1.0) / (xs * (xs * xs - 1.0) ** 2)
    return jnp.where(ok, f, -0.2)


def dphi_nfw_dl(x):
    gt = x > 1.0 + 1.0e-9
    lt = (x < 1.0 - 1.0e-9) & (x > 1.0e-5)
    tiny = x <= 1.0e-5
    xg = jnp.where(gt, x, 2.0)
    xl = jnp.where(lt, x, 0.5)
    xt = jnp.where(tiny, x, 1.0e-6)
    fg = 2.0 * jnp.arctan(jnp.sqrt((xg - 1.0) / (xg + 1.0))) / (xg * jnp.sqrt(xg * xg - 1.0)) + jnp.log(0.5 * xg) / xg
    fl = 2.0 * jnp.arctanh(jnp.sqrt((1.0 - xl) / (xl + 1.0))) / (xl * jnp.sqrt(1.0 - xl * xl)) + jnp.log(0.5 * xl) / xl
    ft = 0.5 * xt * jnp.log(2.0 / xt)
    return jnp.where(gt, fg, jnp.where(lt, fl, jnp.where(tiny, ft, 1.0 + np.log(0.5))))


def ddphi_nfw_dl(x):
    return ddphi_kappadphi(kappa_nfw_dl(x), dphi_nfw_dl(x), x)


def phi_nfw_dl(x):
    big = x > 1.0e-6
    xb = jnp.where(big, x, 0.5)
    xs = jnp.where(big, 1.0e-7, x)
    a = jnp.log(0.5 * xb)
    lt = xb < 1.0 - 1.0e-9
    gt = xb > 1.0 + 1.0e-9
    bl = jnp.arccosh(1.0 / jnp.where(lt, xb, 0.5))
    bg = jnp.arccos(1.0 / jnp.where(gt, xb, 2.0))
    bb = jnp.where(lt, -bl * bl, jnp.where(gt, bg * bg, 0.0))
    return jnp.where(big, 0.5 * (a * a + bb), 0.25 * xs * xs * jnp.log(2.0 / xs))


def nfw_radial(x):
    return kappa_nfw_dl(x), dkappa_nfw_dl(x), dphi_nfw_dl(x)


def hnfw(c):
    small = c < 1.0e-6
    cl = jnp.where(small, 1.0, c)
    return jnp.where(small, 0.5 * c * c, jnp.log(1.0 + cl) - cl / (1.0 + cl))


def rs_nfw(m, c, ctx):
    return NFW_RS_NORM * jnp.power(m / ctx.delome, 1.0 / 3.0) / c


def b_func(m, c, ctx, h_func=hnfw):
    return NFW_B_NORM * ctx.dol * ctx.dls * jnp.power(ctx.delome * ctx.delome * m, 1.0 / 3.0) * (c * c / h_func(c)) / ctx.dos


def calc_bbtt_nfw(m, c, ctx):
    if ctx.settings.nfw_users == 0:
        return b_func(m, c, ctx), rtotheta(rs_nfw(m, c, ctx), ctx)
    cc = rs_nfw(m, 1.0, ctx) / thetator(c, ctx)
    return b_func(m, cc, ctx), c


# Hernquist ---------------------------------------------------------------

def kappa_hern_dl(x):
    ok = (x > 1.0 + 1.0e-5) | (x < 1.0 - 1.0e-5)
    xs = jnp.where(ok, x, 2.0)
    f = ((2.0 + xs * xs) * func_hern_dl(xs) - 3.0) / ((xs * xs - 1.0) ** 2)
    return jnp.where(ok, f, 4.0 / 15.0)


def dkappa_hern_dl(x):
    ok = (x > 1.0 + 1.0e-4) | (x < 1.0 - 1.0e-4)
    xs = jnp.where(ok, x, 2.0)
    f = (2.0 + 13.0 * xs * xs - 3.0 * xs * xs * (xs * xs + 4.0) * func_hern_dl(xs)) / (xs * (xs * xs - 1.0) ** 3)
    return jnp.where(ok, f, -0.4571429)


def dphi_hern_dl(x):
    ok = (x > 1.0 + 1.0e-7) | (x < 1.0 - 1.0e-7)
    xs = jnp.where(ok, x, 2.0)
    f = 2.0 * xs * (1.0 - func_hern_dl(xs)) / (xs * xs - 1.0)
    return jnp.where(ok, f, 2.0 / 3.0)


def phi_hern_dl(x):
    big = x > 1.0e-3
    xb = jnp.where(big, x, 0.5)
    xs = jnp.where(big, 1.0e-4, x)
    return jnp.where(big, jnp.log(xb * xb / 4.0) + 2.0 * func_hern_dl(xb), xs * xs * (jnp.log(2.0 / xs) - 0.5))


def ddphi_hern_dl(x):
    return ddphi_kappadphi(kappa_hern_dl(x), dphi_hern_dl(x), x)


def hern_radial(x):
    return kappa_hern_dl(x), dkappa_hern_dl(x), dphi_hern_dl(x)


def b_func_hern(m, rb, ctx):
    rr = thetator(rb, ctx)
    return (m * inv_sigma_crit(ctx) / (rr * rr)) / (2.0 * np.pi)


# power law ---------------------------------------------------------------

def pow_radial_funcs(gam):
    def kappa(x):
        return 0.5 * (3.0 - gam) * jnp.power(x, 1.0 - gam)

    def dkappa(x):
        return 0.5 * (3.0 - gam) * (1.0 - gam) * jnp.power(x, -gam)

    def dphi(x):
        return jnp.power(x, 2.0 - gam)

    def ddphi(x):
        return (2.0 - gam) * jnp.power(x, 1.0 - gam)

    def phi(x):
        return jnp.power(x, 3.0 - gam) / (3.0 - gam)

    return kappa, dkappa, dphi, ddphi, phi


# Sersic ------------------------------------------------------------------

def bn_sers(n):
    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n
    big = 2.0 * n - 1.0 / 3.0 + (4.0 / 405.0) / n + (46.0 / 25515.0) / n2 + (131.0 / 1148175.0) / n3 \
        - (2194697.0 / 30690717750.0) / n4
    small = 0.01945 - 0.8902 * n + 10.95 * n2 - 19.67 * n3 + 13.43 * n4
    return jnp.where(n > 0.36, big, small)


def bnn_sers(n):
    return jnp.power(bn_sers(n), -n)


def gam2n1_sers(n):
    return jnp.exp(gammaln(2.0 * n + 1.0))


def sers_radial_funcs(n):
    g = gam2n1_sers(n)

    def kappa(x):
        return jnp.exp(-jnp.power(x, 1.0 / n))

    def dkappa(x):
        xx = jnp.power(x, 1.0 / n)
        return -xx * jnp.exp(-xx) / (x * n)

    def dphi(x):
        xx = jnp.power(x, 1.0 / n)
        return g * gammainc(2.0 * n, xx) / x

    def ddphi(x):
        return ddphi_kappadphi(kappa(x), dphi(x), x)

    def phi(x):
        return qng21(dphi, 1.0e-12, x)

    return kappa, dkappa, dphi, ddphi, phi


def b_func_sers(m, rs, n, ctx):
    rr = thetator(rs, ctx)
    return (m * inv_sigma_crit(ctx) / (np.pi * rr * rr)) / gam2n1_sers(n)


# truncated NFW -----------------------------------------------------------

def func_tnfw_dl(x, tau):
    return jnp.log(x / (jnp.sqrt(tau * tau + x * x) + tau))


def _kappa_tnfw_f1(x, u, t2, t4, tau):
    near = (x > 1.0 - 1.0e-7) & (x < 1.0 + 1.0e-7)
    st1 = jnp.sqrt(1.0 + t2)
    fn = (tau / (24.0 * (1.0 + t2) ** 4)) * (tau * (-6.0 + 62.0 * t4 + 4.0 * t4 * t2 + t2 * (52.0 - 30.0 * np.pi * st1))
                                             + 3.0 * st1 * (1.0 + 7.0 * t2 - 4.0 * t4) * jnp.log(1.0 + 2.0 * tau * (tau + st1)))
    xs = jnp.where(near, 2.0, x)
    us = xs * xs
    st2u = jnp.sqrt(t2 + us)
    fx = func_hern_dl(xs)
    ff = (t4 / (4.0 * (t2 + 1.0) ** 3)) * (2.0 * (t2 + 1.0) * (1.0 - fx) / (us - 1.0) + 8.0 * fx + (t4 - 1.0) / (t2 * (t2 + us))
                                          - np.pi * (4.0 * (t2 + us) + t2 + 1.0) / (st2u * (t2 + us))
                                          + func_tnfw_dl(xs, tau) * (t2 * (t4 - 1.0) + (t2 + us) * (3.0 * t4 - 6.0 * t2 - 1.0))
                                          / (tau * t2 * st2u * (t2 + us)))
    return jnp.where(near, fn, ff)


def _kappa_tnfw_f2(x, u, t2, t4):
    u3 = u * u * u
    return (4.0 * t4 / (15.0 * u3) - 5.0 * np.pi * t4 / (32.0 * u3 * x) - 8.0 * (t4 * (-3.0 + 2.0 * t2)) / (35.0 * u3 * u)
            + 35.0 * t4 * np.pi * (t2 - 1.0) / (128.0 * u3 * u * x))


def _dkappa_tnfw_f1(x, u, t2, t4, tau):
    near = (x > 1.0 - 1.0e-6) & (x < 1.0 + 1.0e-6)
    st1 = jnp.sqrt(1.0 + t2)
    fn = (-tau / (120.0 * (1.0 + t2) ** 5)) * (2.0 * tau * (-15.0 + 271.0 * t4 + 56.0 * t4 * t2 + 12.0 * t4 * t4
                                                             + t2 * (212.0 - 105.0 * np.pi * st1))
                                               - 15.0 * st1 * (-1.0 - 9.0 * t2 + 6.0 * t4) * jnp.log(1.0 + 2.0 * tau * (tau + st1)))
    xs = jnp.where(near, 2.0, x)
    us = xs * xs
    st2u = jnp.sqrt(t2 + us)
    t2u2 = (t2 + us) ** 2
    t2u52 = t2u2 * st2u
    fx = func_hern_dl(xs)
    ff = (t4 / (4.0 * (t2 + 1.0) ** 3)) * (
        2.0 * (t2 + 1.0) * (3.0 * us * fx - 2.0 * us - 1.0) / (xs * (us - 1.0) ** 2)
        + 8.0 * (1.0 - us * fx) / (xs * (us - 1.0)) - 2.0 * xs * (t4 - 1.0) / (t2 * t2u2)
        + np.pi * xs * (4.0 * (t2 + us) + 3.0 * (t2 + 1.0)) / t2u52
        + (t2 * (t4 - 1.0) + (t2 + us) * (3.0 * t4 - 6.0 * t2 - 1.0)) / (xs * t2 * t2u2)
        - func_tnfw_dl(xs, tau) * xs * (3.0 * t2 * (t4 - 1.0) + (t2 + us) * (3.0 * t4 - 6.0 * t2 - 1.0)) / (tau * t2 * t2u52))
    return jnp.where(near, fn, ff)


def _dkappa_tnfw_f2(x, u, t2, t4):
    u3 = u * u * u
    return ((-8.0 * t4) / (5.0 * u3 * x) + 35.0 * np.pi * t4 / (32.0 * u3 * u) + 64.0 * (t4 * (-3.0 + 2.0 * t2)) / (35.0 * u3 * u * x)
            - 315.0 * t4 * np.pi * (t2 - 1.0) / (128.0 * u3 * u * u))


def _tnfw_select(x, tau, f1, f2):
    t2 = tau * tau
    t4 = t2 * t2
    u = x * x
    big_tau = t2 > 1.0e-4
    use1 = jnp.where(big_tau, u < 1.0e5 * t2, (u <= 1.0e4) & (u < 1.0e6 * t2))
    use2 = jnp.where(big_tau, ~(u < 1.0e5 * t2), u > 1.0e4)
    x1 = jnp.where(use1, x, 0.5)
    x2 = jnp.where(use2, x, 1.0e3)
    v1 = f1(x1, x1 * x1, t2, t4, tau)
    v2 = f2(x2, x2 * x2, t2, t4)
    # interpolation regime
    xa = 1.0e2
    xb = 1.0e3 * tau
    fa = f2(jnp.asarray(xa), xa * xa, t2, t4)
    fb = f1(xb, xb * xb, t2, t4, tau)
    sgn = jnp.sign(fa)
    la = jnp.log(jnp.abs(fa))
    lb = jnp.log(jnp.abs(fb))
    xi = jnp.where(use1 | use2, xa, x)
    vi = sgn * jnp.exp((la - lb) / (np.log(xa) - jnp.log(xb)) * (jnp.log(xi) - jnp.log(xb)) + lb)
    return jnp.where(use1, v1, jnp.where(use2, v2, vi))


def kappa_tnfw_dl(x, tau):
    return _tnfw_select(x, tau, _kappa_tnfw_f1, _kappa_tnfw_f2)


def dkappa_tnfw_dl(x, tau):
    return _tnfw_select(x, tau, _dkappa_tnfw_f1, _dkappa_tnfw_f2)


def dphi_tnfw_dl(x, tau):
    t2 = tau * tau
    t4 = t2 * t2
    u = x * x
    lnt = jnp.log(tau)
    reg = ((x > 3.0e-3) & (tau > 3.0e-3)) | (u > t2 * 1.0e-3)
    xs = jnp.where(reg, x, 1.0)
    us = xs * xs
    st2u = jnp.sqrt(t2 + us)
    fx = func_hern_dl(xs)
    lx = func_tnfw_dl(xs, tau)
    f_reg = (t4 / (2.0 * (t2 + 1.0) ** 3 * xs)) * (2.0 * (t2 + 4.0 * us - 3.0) * fx + (np.pi * (3.0 * t2 - 1.0) + 2.0 * tau * (t2 - 3.0) * lnt) / tau
                                                   + (-t2 * tau * np.pi * (4.0 * us + 3.0 * t2 - 1.0) + (2.0 * t4 * t2 - 6.0 * t4 + us * (3.0 * t4 - 6.0 * t2 - 1.0)) * lx)
                                                   / (t2 * tau * st2u))
    xt = jnp.where(reg, 1.0e-4, x)
    ln2 = np.log(2.0)
    f_a = (1.0 / (4.0 * (t2 + 1.0) ** 3)) * (3.0 * t2 - np.pi * tau - 5.0 * np.pi * tau * t2 + t4 * t2 * (-1.0 + 2.0 * ln2) + t4 * (2.0 + 10.0 * ln2)
                                            + (2.0 + 6.0 * t2 - 4.0 * t4) * (ln2 + lnt) - 2.0 * (t2 + 1.0) ** 3 * jnp.log(xt)) * xt
    f_b = (xt * (2.0 * ln2 + 2.0 * lnt - 2.0 * jnp.log(xt) - np.pi * tau) / 4.0) + 3.0 * xt * t2 / 4.0
    return jnp.where(reg, f_reg, jnp.where(tau > 3.0e-3, f_a, f_b))


def phi_tnfw_dl(x, tau):
    t2 = tau * tau
    t4 = t2 * t2
    u = x * x
    ln2 = np.log(2.0)
    lnt = jnp.log(tau)
    reg = ((x > 3.0e-3) & (tau > 3.0e-3)) | (u > t2 * 1.0e-3)
    xs = jnp.where(reg, x, 1.0)
    us = xs * xs
    st2u = jnp.sqrt(t2 + us)
    fx = func_hern_dl(xs)
    lx = func_tnfw_dl(xs, tau)
    f_reg = (1.0 / (4.0 * (t2 + 1.0) ** 3)) * (
        2.0 * t2 * tau * np.pi * (-4.0 * tau * st2u + (3.0 * t2 - 1.0) * jnp.log(st2u + tau))
        + 2.0 * tau * (3.0 * t4 - 6.0 * t2 - 1.0) * st2u * lx + 2.0 * t4 * (t2 - 3.0) * lx * lx
        + 16.0 * t4 * (us - 1.0) * fx + 2.0 * t4 * (t2 - 3.0) * (us - 1.0) * fx * fx
        + t2 * (2.0 * t2 * (t2 - 3.0) * lnt - 3.0 * t4 - 2.0 * t2 + 1.0) * jnp.log(us)
        + 2.0 * t2 * (t2 * (4.0 * tau * np.pi + (t2 - 3.0) * ln2 * ln2 + 8.0 * ln2)
                      - (ln2 + lnt) * (1.0 + 6.0 * t2 - 3.0 * t4 + t2 * (t2 - 3.0) * (ln2 + lnt))
                      - tau * np.pi * (3.0 * t2 - 1.0) * (ln2 + lnt)))
    xt = jnp.where(reg, 1.0e-4, x)
    ut = xt * xt
    f_a = (1.0 / (8.0 * (t2 + 1.0) ** 3)) * (1.0 + tau * (-np.pi + tau * (6.0 + tau * (-5.0 * np.pi + tau * (5.0 + (5.0 + t2) * 2.0 * ln2))))
                                            + (2.0 + 6.0 * t2 - 4.0 * t4) * (ln2 + lnt) - 2.0 * (t2 + 1.0) ** 3 * jnp.log(xt)) * ut
    f_b = ut * (1.0 + 2.0 * ln2 + 2.0 * lnt - 2.0 * jnp.log(xt) - np.pi * tau) / 8.0
    return jnp.where(reg, f_reg, jnp.where(tau > 3.0e-3, f_a, f_b))


def calc_bbtt_tnfw(m, c, ctx):
    if ctx.settings.nfw_users == 0:
        return b_func(m, c, ctx), rtotheta(rs_nfw(m, c, ctx), ctx), c
    cc = rs_nfw(m, 1.0, ctx) / thetator(c, ctx)
    return b_func(m, cc, ctx), c, cc


# gNFW / Einasto ------------------------------------------------------------

def hgnfw(c, alpha, n=48):
    """\\int_0^c (s/(1+s))^{2-alpha}/(1+s) ds (integrated in ln s)."""
    t, w = composite_gl(4, 12)
    lc = jnp.log(c)
    lo = lc - 40.0
    ls = lo[..., None] + 40.0 * t
    s = jnp.exp(ls)
    f = s * jnp.power(s / (1.0 + s), 2.0 - alpha) / (1.0 + s)
    return jnp.sum(f * w, axis=-1) * 40.0


def hein(x, alpha):
    a3 = 3.0 / alpha
    lg = a3 * jnp.log(0.5 * alpha) + gammaln(a3) - jnp.log(alpha)
    return jnp.exp(lg) * gammainc(a3, 2.0 * jnp.power(x, alpha) / alpha)


def calc_bbtt_gnfw(m, c, alpha, ctx):
    h = lambda cc: hgnfw(cc, alpha)
    if ctx.settings.nfw_users == 0:
        cc = (2.0 - alpha) * c
        return b_func(m, cc, ctx, h), rtotheta(rs_nfw(m, cc, ctx), ctx)
    cc = rs_nfw(m, 1.0, ctx) / thetator(c, ctx)
    return b_func(m, cc, ctx, h), c


def calc_bbtt_ein(m, c, alpha, ctx):
    h = lambda cc: hein(cc, alpha)
    if ctx.settings.nfw_users == 0:
        return b_func(m, c, ctx, h), rtotheta(rs_nfw(m, c, ctx), ctx)
    cc = rs_nfw(m, 1.0, ctx) / thetator(c, ctx)
    return b_func(m, cc, ctx, h), c


def _tab_funcs(kind, alpha):
    rad = tables.gnfw_radial if kind == "gnfw" else tables.ein_radial

    def radial(x):
        return rad(x, alpha)

    def dphi(x):
        return rad(x, alpha)[2]

    def ddphi(x):
        k, _, p = rad(x, alpha)
        return ddphi_kappadphi(k, p, x)

    def phi(x):
        lx = jnp.log(x)
        return qng21(lambda t: jnp.exp(t) * rad(jnp.exp(t), alpha)[2], -25.0, lx)

    return radial, dphi, ddphi, phi


# ---------------------------------------------------------------------------
# SIE family


def facq_sie(q):
    return 1.0 / jnp.sqrt(q)


def b_sie(sig, q, ctx):
    ss = sig / C_LIGHT_KMS
    return facq_sie(q) * (4.0 * np.pi * ss * ss * ctx.dls / ctx.dos) / ARCSEC2RADIAN


def alpha_sie_dl(x, y, s, q):
    ell = (1.0 - q) > 1.0e-5
    qe = jnp.where(ell, q, 0.5)
    sq = jnp.sqrt(1.0 - qe * qe)
    psi = jnp.sqrt(qe * qe * (s * s + x * x) + y * y)
    ax_e = (qe / sq) * jnp.arctan(sq * x / (psi + s))
    ay_e = (qe / sq) * jnp.arctanh(sq * y / (psi + qe * qe * s))
    psi1 = jnp.sqrt(s * s + x * x + y * y)
    return jnp.where(ell, ax_e, x / (psi1 + s)), jnp.where(ell, ay_e, y / (psi1 + s))


def ddphi_sie_dl(x, y, s, q):
    psi = jnp.sqrt(q * q * (s * s + x * x) + y * y)
    f = (1.0 + q * q) * s * s + 2.0 * psi * s + x * x + y * y
    pxx = (q / psi) * (q * q * s * s + y * y + s * psi) / f
    pyy = (q / psi) * (s * s + x * x + s * psi) / f
    pxy = (q / psi) * (-x * y) / f
    return pxx, pxy, pyy


def phi_sie_dl(x, y, s, q):
    px, py = alpha_sie_dl(x, y, s, q)
    psi = jnp.sqrt(q * q * (s * s + x * x) + y * y)
    aa = jnp.log((1.0 + q) * s) - jnp.log(jnp.sqrt((psi + s) ** 2 + (1.0 - q * q) * x * x))
    return x * px + y * py + q * s * aa


def sie_bq(dx, dy, bb, s, q, si, co):
    """Deflection, potential and hessian of an SIE with Einstein-radius-like
    normalisation ``bb`` (alpha_sie_bq + kapgam_sie_bq)."""
    ss = s * facq_sie(q)
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    aax, aay = alpha_sie_dl(ddx, ddy, ss, q)
    pxx, pxy, pyy = ddphi_sie_dl(ddx, ddy, ss, q)
    ax, ay, rpxx, rpxy, rpyy = _rotate_back(aax, aay, pxx, pxy, pyy, si, co)
    phi = bb * phi_sie_dl(ddx, ddy, ss, q)
    kap = 0.5 * bb * (rpxx + rpyy)
    g1 = 0.5 * bb * (rpxx - rpyy)
    g2 = bb * rpxy
    return bb * ax, bb * ay, phi, kap, g1, g2


# ---------------------------------------------------------------------------
# CSE approximations (app_ell.c / app_cnfw.c)

_CSE = None


def _cse_tables():
    global _CSE
    if _CSE is None:
        d = np.load(os.path.join(os.path.dirname(__file__), "data", "cse_tables.npz"))
        _CSE = {k: d[k] for k in d.files}
    return _CSE


def cse_all(x, y, s, a, q):
    """Sum over CSE components; s, a have shape (..., n) or (n,)."""
    x = x[..., None]
    y = y[..., None]
    psi = jnp.sqrt(q * q * (s * s + x * x) + y * y)
    f = (psi + s) ** 2 + (1.0 - q * q) * x * x
    fac = q / (s * psi * f)
    ax = jnp.sum(a * fac * x * (psi + q * q * s), axis=-1)
    ay = jnp.sum(a * fac * y * (psi + s), axis=-1)
    pi_ = 1.0 / psi
    fi = 1.0 / f
    qs = q / s
    q2 = q * q
    pxx = qs * fi * (1.0 + pi_ ** 3 * q2 * s * (q2 * s * s + y * y) - 2.0 * pi_ * pi_ * fi * x * x * (psi + q2 * s) ** 2)
    pyy = qs * fi * (1.0 + pi_ ** 3 * q2 * s * (s * s + x * x) - 2.0 * pi_ * pi_ * fi * y * y * (psi + s) ** 2)
    pxy = -qs * x * y * fi * (pi_ ** 3 * q2 * s + 2.0 * pi_ * pi_ * fi * (psi + q2 * s) * (psi + s))
    phi = (q / s) * (0.5 * jnp.log(f) - jnp.log((1.0 + q) * s))
    return (ax, ay, jnp.sum(a * phi, axis=-1), jnp.sum(a * pxx, axis=-1),
            jnp.sum(a * pxy, axis=-1), jnp.sum(a * pyy, axis=-1))


def _cse_lens(tx, ty, cx, cy, tt, q, pa, bb, s, a):
    si, co = _sc(-(pa - 90.0))
    dx = (tx - cx) / tt
    dy = (ty - cy) / tt
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    aax, aay, phi, pxx, pxy, pyy = cse_all(ddx, ddy, s, a, q)
    ax, ay, rpxx, rpxy, rpyy = _rotate_back(aax, aay, pxx, pxy, pyy, si, co)
    return (bb * ax * tt, bb * ay * tt, bb * phi * tt * tt, 0.5 * bb * (rpxx + rpyy),
            0.5 * bb * (rpxx - rpyy), bb * rpxy)


def _acnfw_coeffs(b):
    tab = jnp.asarray(_cse_tables()["cnfw"])
    lbmin, dlb, nb = -8.0, 0.05, 201
    lb = jnp.log10(jnp.maximum(b, 1.0e-300))
    fj = jnp.floor((lb - lbmin) / dlb)
    low = b <= 10.0 ** lbmin
    top = fj >= nb - 1
    j = jnp.clip(fj, 0, nb - 2).astype(jnp.int32)
    p = (lb - (lbmin + j * dlb)) / dlb
    p = jnp.where(low, 0.0, jnp.where(top, 1.0, p))
    j = jnp.where(low, 0, j)
    c0 = tab[j]
    c1 = tab[j + 1]
    s = jnp.concatenate([c0[:, 0], c1[:, 0]])
    a = jnp.concatenate([(1.0 - p) * c0[:, 1], p * c1[:, 1]])
    return s, a


# ---------------------------------------------------------------------------
# the models


def m_gals(tx, ty, p, ctx):
    gal = jnp.asarray(ctx.gals)
    sig, a, alpha = p[1], p[6], p[7]
    q = 1.0 - gal[:, 3]
    bb = b_sie(jnp.power(gal[:, 2], 0.25) * sig, q, ctx)
    aa = jnp.power(gal[:, 2], alpha) * a
    si, co = _sc(-(gal[:, 4] - 90.0))
    dx = tx[..., None] - gal[:, 0]
    dy = ty[..., None] - gal[:, 1]
    r1 = sie_bq(dx, dy, bb, ctx.settings.smallcore, q, si, co)
    r2 = sie_bq(dx, dy, bb, aa, q, si, co)
    return tuple(jnp.sum(u - v, axis=-1) for u, v in zip(r1, r2))


def m_nfwpot(tx, ty, p, ctx):
    bb, tt = calc_bbtt_nfw(p[1], p[6], ctx)
    return ellpot(tx, ty, p[2], p[3], tt, p[4], p[5], bb, dphi_nfw_dl, ddphi_nfw_dl, phi_nfw_dl, ctx)


def m_sie(tx, ty, p, ctx):
    sig, s, e, pa = p[1], p[6], p[4], p[5]
    q = 1.0 - e
    bb = b_sie(sig, q, ctx)
    s = jnp.maximum(s, ctx.settings.smallcore)
    si, co = _sc(-(pa - 90.0))
    return sie_bq(tx - p[2], ty - p[3], bb, s, q, si, co)


def m_jaffe(tx, ty, p, ctx):
    sig, a, rco, e, pa = p[1], p[6], p[7], p[4], p[5]
    rco = jnp.maximum(rco, ctx.settings.smallcore)
    q = 1.0 - e
    bb = b_sie(sig, q, ctx)
    si, co = _sc(-(pa - 90.0))
    dx = tx - p[2]
    dy = ty - p[3]
    r1 = sie_bq(dx, dy, bb, rco, q, si, co)
    r2 = sie_bq(dx, dy, bb, jnp.maximum(a, rco * 1.000001), q, si, co)
    on = a > rco
    return tuple(jnp.where(on, u - v, 0.0) for u, v in zip(r1, r2))


def re2_point(m, ctx):
    d = ctx.dls / (COVERH_MPCH * ctx.dol * ctx.dos)
    return (2.0 * (R_SCHWARZ * m / MPC2METER) * d) / (ARCSEC2RADIAN * ARCSEC2RADIAN)


def m_point(tx, ty, p, ctx):
    sc = ctx.settings.smallcore
    re2 = re2_point(p[1], ctx)
    dx = tx - p[2]
    dy = ty - p[3]
    r2 = dx * dx + dy * dy
    rr = re2 / (r2 + sc * sc)
    ax = rr * dx
    ay = rr * dy
    inner = r2 < sc * sc
    r2s = jnp.where(inner, 1.0, r2)
    g1 = jnp.where(inner, re2 / (2.0 * sc * sc), (re2 / (r2s * r2s)) * (dy * dy - dx * dx))
    g2 = jnp.where(inner, re2 / (2.0 * sc * sc), (re2 / (r2s * r2s)) * (-2.0 * dx * dy))
    phi = re2 * 0.5 * jnp.log(jnp.maximum(r2, sc * sc))
    return ax, ay, phi, jnp.zeros_like(ax), g1, g2


def m_pert(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx)
    k, g, tg = p[7], p[4], p[5]
    si, co = _sc(2.0 * (tg - 90.0))
    dx = tx - p[2]
    dy = ty - p[3]
    ax = fac * (dx * k - dx * g * co - dy * g * si)
    ay = fac * (dy * k + dy * g * co - dx * g * si)
    kap = k * fac
    g1 = -g * fac * co
    g2 = -g * fac * si
    phi = 0.5 * (dx * dx + dy * dy) * kap + 0.5 * (dx * dx - dy * dy) * g1 + dx * dy * g2
    kap = jnp.broadcast_to(kap, ax.shape)
    g1 = jnp.broadcast_to(g1, ax.shape)
    g2 = jnp.broadcast_to(g2, ax.shape)
    return ax, ay, phi, kap, g1, g2


def m_clus3(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx)
    g, tg = p[4], p[5]
    sc = ctx.settings.smallcore
    dx = tx - p[2]
    dy = ty - p[3]
    r = ssqrt(dx * dx + dy * dy) + sc
    cox = dx / r
    six = dy / r
    cox3 = cox * (cox * cox - 3.0 * six * six)
    six3 = (3.0 * cox * cox - six * six) * six
    sig, cog = _sc(tg)
    cog3 = cog * (cog * cog - 3.0 * sig * sig)
    sig3 = (3.0 * cog * cog - sig * sig) * sig
    co = cox * cog + six * sig
    si = six * cog - cox * sig
    co3 = cox3 * cog3 + six3 * sig3
    si3 = six3 * cog3 - cox3 * sig3
    f = fac * (g / 4.0)
    ax = f * (3.0 * r * dx * (si + si3) - r * dy * (co + 3.0 * co3))
    ay = f * (3.0 * r * dy * (si + si3) + r * dx * (co + 3.0 * co3))
    pxx = f * (3.0 * (r + dx * dx / r) * (si + si3) - 4.0 * (dx * dy / r) * (co + 3.0 * co3) - (dy * dy / r) * (si + 9.0 * si3))
    pyy = f * (3.0 * (r + dy * dy / r) * (si + si3) + 4.0 * (dx * dy / r) * (co + 3.0 * co3) - (dx * dx / r) * (si + 9.0 * si3))
    pxy = f * (3.0 * (dx * dy / r) * (si + si3) + 2.0 * ((dx * dx - dy * dy) / r) * (co + 3.0 * co3) + (dx * dy / r) * (si + 9.0 * si3))
    kap, g1, g2 = _kapgam_from_hessian(pxx, pxy, pyy)
    phi = f * r ** 3 * (si + si3)
    return ax, ay, phi, kap, g1, g2


def m_mpole(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx)
    g, tg, m, n = p[4], p[5], p[6], p[7]
    sc = ctx.settings.smallcore
    dx = tx - p[2]
    dy = ty - p[3]
    r = ssqrt(dx * dx + dy * dy) + sc
    t = jnp.arctan2(dy, dx + sc) / D2R
    co = jnp.cos(m * (t - tg - 90.0) * D2R)
    si = jnp.sin(m * (t - tg - 90.0) * D2R)
    f2 = -g * jnp.power(r, n - 2.0) / m
    ff = fac * f2
    ax = ff * (n * dx * co + m * dy * si)
    ay = ff * (n * dy * co - m * dx * si)
    r2 = r * r
    pxx = ff * ((n - 2.0) * dx * (n * dx * co + m * dy * si) + n * r2 * co + n * m * dx * dy * si - m * m * dy * dy * co) / r2
    pyy = ff * ((n - 2.0) * dy * (n * dy * co - m * dx * si) + n * r2 * co - n * m * dx * dy * si - m * m * dx * dx * co) / r2
    pxy = ff * ((n * (n - 2.0) + m * m) * dx * dy * co + m * (n - 1.0) * (dy * dy - dx * dx) * si) / r2
    kap, g1, g2 = _kapgam_from_hessian(pxx, pxy, pyy)
    phi = ff * r2 * co
    return ax, ay, phi, kap, g1, g2


def m_hernpot(tx, ty, p, ctx):
    bb = b_func_hern(p[1], p[6], ctx)
    return ellpot(tx, ty, p[2], p[3], p[6], p[4], p[5], bb, dphi_hern_dl, ddphi_hern_dl, phi_hern_dl, ctx)


def m_nfw(tx, ty, p, ctx):
    q = 1.0 - p[4]
    bb, tt = calc_bbtt_nfw(p[1], p[6], ctx)
    return elldens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, nfw_radial, "std", ctx)


def m_hern(tx, ty, p, ctx):
    q = 1.0 - p[4]
    bb = b_func_hern(p[1], p[6], ctx)
    return elldens(tx, ty, p[2], p[3], p[6] / jnp.sqrt(q), q, p[5], bb, hern_radial, "std", ctx)


def m_powpot(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx)
    re, gam = p[6], p[7]
    _, _, dphi, ddphi, phi = pow_radial_funcs(gam)
    return ellpot(tx, ty, p[2], p[3], re, p[4], p[5], fac, dphi, ddphi, phi, ctx)


def _pow_direct(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx)
    re, gam = p[6], p[7]
    q = 1.0 - p[4]
    tt = re / jnp.sqrt(q)
    kappa, dkappa, dphi, _, _ = pow_radial_funcs(gam)
    radial = lambda x: (kappa(x), dkappa(x), dphi(x))
    # glafic: lower limit ULIM_JHK if gam < 1.7, otherwise smallcore-based
    si, co = _sc(-p[5])
    dx = tx - p[2]
    dy = ty - p[3]
    bx = (co * dx - si * dy) / tt
    by = (si * dx + co * dy) / tt
    sc = ctx.settings.smallcore
    uu = 1.0 / (bx * bx + by * by + sc * sc)
    lnlo = jnp.where(gam < 1.7, np.log(ULIM_JHK), jnp.log(sc * sc * uu))
    return _ell_core(bx, by, lnlo, q, si, co, fac, tt, radial, ctx.settings)


def _ell_core(bx, by, lnlo, q, si, co, bb, tt, radial, s):
    sc = s.smallcore
    t01, w01 = composite_gl(s.ell_quad_panels, s.ell_quad_nodes)
    lnlo_ = lnlo[..., None]
    lu = lnlo_ + (0.0 - lnlo_) * t01
    wt = (0.0 - lnlo_) * w01
    u = jnp.exp(lu)
    equ = 1.0 - (1.0 - q * q) * u
    bx_ = bx[..., None]
    by_ = by[..., None]
    se = ssqrt(u * (by_ * by_ + bx_ * bx_ / equ + sc * sc))
    kap_r, dkap_r, dphi_r = radial(se)
    sq = jnp.sqrt(equ)
    n0, n1 = sq, sq * equ
    n2 = n1 * equ
    uw = u * wt
    j0 = jnp.sum(kap_r / n0 * uw, axis=-1)
    j1 = jnp.sum(kap_r / n1 * uw, axis=-1)
    kfac = u * dkap_r / (2.0 * se)
    k0 = jnp.sum(kfac / n0 * uw, axis=-1)
    k1 = jnp.sum(kfac / n1 * uw, axis=-1)
    k2 = jnp.sum(kfac / n2 * uw, axis=-1)
    ii = jnp.sum(se * dphi_r / (u * n0) * uw, axis=-1)
    bpx = q * bx * j1
    bpy = q * by * j0
    ax = bb * tt * (bpx * co + bpy * si)
    ay = bb * tt * (-bpx * si + bpy * co)
    bpxx = 2.0 * q * bx * bx * k2 + q * j1
    bpyy = 2.0 * q * by * by * k0 + q * j0
    bpxy = 2.0 * q * bx * by * k1
    pxx = bpxx * co * co + bpxy * 2.0 * co * si + bpyy * si * si
    pxy = -bpxx * si * co + bpxy * (co * co - si * si) + bpyy * co * si
    pyy = bpxx * si * si - bpxy * 2.0 * co * si + bpyy * co * co
    return (ax, ay, 0.5 * q * bb * ii * tt * tt, 0.5 * bb * (pxx + pyy), 0.5 * bb * (pxx - pyy), bb * pxy)


def pow_tm15_omega(psi, q, gam, nterms):
    t = 3.0 - gam
    f = (1.0 - q) / (1.0 + q)
    c2 = jnp.cos(2.0 * psi)
    s2 = jnp.sin(2.0 * psi)
    a0 = jnp.cos(psi)
    a1 = jnp.sin(psi)

    def body(carry, kk):
        a0, a1, o0, o1 = carry
        fac = -f * (2.0 * kk - t) / (2.0 * kk + t)
        b0 = fac * (c2 * a0 - s2 * a1)
        b1 = fac * (s2 * a0 + c2 * a1)
        return (b0, b1, o0 + b0, o1 + b1), None

    ks = jnp.arange(1, nterms + 1, dtype=float)
    (a0, a1, o0, o1), _ = jax.lax.scan(body, (a0, a1, a0, a1), ks)
    return o0, o1


def _pow_tm15(tx, ty, p, ctx):
    s = ctx.settings
    sc = s.smallcore
    fac = fac_pert(p[1], ctx)
    re, gam, e, pa = p[6], p[7], p[4], p[5]
    q = 1.0 - e
    tt = re * jnp.sqrt(q)
    si, co = _sc(-(pa - 90.0))
    dx = tx - p[2]
    dy = ty - p[3]
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    r = ssqrt(q * q * ddx * ddx + ddy * ddy) + sc
    psi = jnp.arctan2(ddy, q * ddx)
    aa = 2.0 * tt * jnp.power(tt / r, gam - 2.0) / (1.0 + q)
    o0, o1 = pow_tm15_omega(psi, q, gam, s.n_tm15_terms)
    aax = aa * o0
    aay = aa * o1
    ax = fac * (aax * co + aay * si)
    ay = fac * (-aax * si + aay * co)
    kap = fac * 0.5 * (3.0 - gam) * jnp.power(r / tt, 1.0 - gam)
    r2 = ssqrt(ddx * ddx + ddy * ddy) + sc
    c1 = ddx / r2
    s1 = ddy / r2
    cc2 = c1 * c1 - s1 * s1
    ss2 = 2.0 * c1 * s1
    g1 = -cc2 * kap + fac * (2.0 - gam) * (c1 * aax - s1 * aay) / r2
    g2 = -ss2 * kap + fac * (2.0 - gam) * (c1 * aay + s1 * aax) / r2
    c2 = co * co - si * si
    s2 = 2.0 * co * si
    gam1 = c2 * g1 + s2 * g2
    gam2 = -s2 * g1 + c2 * g2
    phi = fac * (ddx * aax + ddy * aay) / (3.0 - gam)
    return ax, ay, phi, kap, gam1, gam2


def m_pow(tx, ty, p, ctx):
    if ctx.settings.flag_pow_tm15 == 0:
        return _pow_direct(tx, ty, p, ctx)
    return _pow_tm15(tx, ty, p, ctx)


def m_gnfwpot(tx, ty, p, ctx):
    alpha = p[7]
    bb, tt = calc_bbtt_gnfw(p[1], p[6], alpha, ctx)
    _, dphi, ddphi, phi = _tab_funcs("gnfw", alpha)
    return ellpot(tx, ty, p[2], p[3], tt, p[4], p[5], bb, dphi, ddphi, phi, ctx)


def m_gnfw(tx, ty, p, ctx):
    alpha = p[7]
    q = 1.0 - p[4]
    bb, tt = calc_bbtt_gnfw(p[1], p[6], alpha, ctx)
    radial, _, _, _ = _tab_funcs("gnfw", alpha)
    r_a = elldens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, radial, "std", ctx)
    r_b = elldens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, radial, "steep", ctx)
    return tuple(jnp.where(alpha <= 1.0, a, b) for a, b in zip(r_a, r_b))


def m_serspot(tx, ty, p, ctx):
    m, re, n = p[1], p[6], p[7]
    tt = re * bnn_sers(n)
    bb = b_func_sers(m, tt, n, ctx)
    _, _, dphi, ddphi, phi = sers_radial_funcs(n)
    return ellpot(tx, ty, p[2], p[3], tt, p[4], p[5], bb, dphi, ddphi, phi, ctx)


def m_sers(tx, ty, p, ctx):
    m, re, n = p[1], p[6], p[7]
    q = 1.0 - p[4]
    tt = re * bnn_sers(n)
    bb = b_func_sers(m, tt, n, ctx)
    kappa, dkappa, dphi, _, _ = sers_radial_funcs(n)
    radial = lambda x: (kappa(x), dkappa(x), dphi(x))
    return elldens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, radial, "std", ctx)


def m_tnfwpot(tx, ty, p, ctx):
    bb, tt, cc = calc_bbtt_tnfw(p[1], p[6], ctx)
    tau = p[7] * cc
    dphi = lambda x: dphi_tnfw_dl(x, tau)
    ddphi = lambda x: ddphi_kappadphi(kappa_tnfw_dl(x, tau), dphi_tnfw_dl(x, tau), x)
    phi = lambda x: phi_tnfw_dl(x, tau)
    return ellpot(tx, ty, p[2], p[3], tt, p[4], p[5], bb, dphi, ddphi, phi, ctx)


def m_tnfw(tx, ty, p, ctx):
    q = 1.0 - p[4]
    bb, tt, cc = calc_bbtt_tnfw(p[1], p[6], ctx)
    tau = p[7] * cc
    radial = lambda x: (kappa_tnfw_dl(x, tau), dkappa_tnfw_dl(x, tau), dphi_tnfw_dl(x, tau))
    return elldens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, radial, "std", ctx)


def m_einpot(tx, ty, p, ctx):
    alpha = p[7]
    bb, tt = calc_bbtt_ein(p[1], p[6], alpha, ctx)
    _, dphi, ddphi, phi = _tab_funcs("ein", alpha)
    return ellpot(tx, ty, p[2], p[3], tt, p[4], p[5], bb, dphi, ddphi, phi, ctx)


def m_ein(tx, ty, p, ctx):
    alpha = p[7]
    q = 1.0 - p[4]
    bb, tt = calc_bbtt_ein(p[1], p[6], alpha, ctx)
    radial, _, _, _ = _tab_funcs("ein", alpha)
    return elldens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, radial, "std", ctx)


def m_anfw(tx, ty, p, ctx):
    q = 1.0 - p[4]
    bb, tt = calc_bbtt_nfw(p[1], p[6], ctx)
    c = _cse_tables()["nfw"]
    return _cse_lens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, c[:, 0], c[:, 1])


def m_ahern(tx, ty, p, ctx):
    q = 1.0 - p[4]
    bb = b_func_hern(p[1], p[6], ctx)
    c = _cse_tables()["hern"]
    return _cse_lens(tx, ty, p[2], p[3], p[6] / jnp.sqrt(q), q, p[5], bb, c[:, 0], c[:, 1])


def m_crline(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx)
    k, pa, eps = p[7], p[5], p[6]
    si, co = _sc(-pa)
    g = 1.0 - k
    dx = (tx - p[2]) * co - (ty - p[3]) * si
    dy = (tx - p[2]) * si + (ty - p[3]) * co
    ax = fac * (dx * k + dx * g - 0.5 * eps * dx * dx)
    ay = fac * (dy * k - dy * g)
    kap = fac * (k - 0.5 * eps * dx)
    g1 = fac * (g - 0.5 * eps * dx)
    phi = fac * (0.5 * (dx * dx + dy * dy) * k + 0.5 * (dx * dx - dy * dy) * g - (1.0 / 6.0) * eps * dx ** 3)
    return ax, ay, phi, kap, g1, jnp.zeros_like(ax)


def _gau_funcs():
    def dphi(x):
        sm = x < 1.0e-4
        xb = jnp.where(sm, 1.0, x)
        return jnp.where(sm, x - x ** 3 / 4.0, 2.0 * (1.0 - jnp.exp(-0.5 * xb * xb)) / xb)

    def ddphi(x):
        sm = x < 1.0e-4
        xb = jnp.where(sm, 1.0, x)
        return jnp.where(sm, 1.0 - 3.0 * x * x / 4.0, 2.0 * jnp.exp(-0.5 * xb * xb) - dphi(xb) / xb)

    def phi(x):
        s1 = x < 9.0e-5
        s2 = x < 1.0e1
        xm = jnp.where(~s1 & s2, x, 1.0)
        return jnp.where(s1, np.log(2.0) - 0.57721566490153286 + 0.5 * x * x,
                         jnp.where(s2, 2.0 * jnp.log(xm) - expi(-0.5 * xm * xm), 2.0 * jnp.log(jnp.maximum(x, 1e-300))))

    return dphi, ddphi, phi


def m_gaupot(tx, ty, p, ctx):
    fac = fac_pert(p[1], ctx) * p[7]
    dphi, ddphi, phi = _gau_funcs()
    return ellpot(tx, ty, p[2], p[3], p[6], p[4], p[5], fac, dphi, ddphi, phi, ctx)


def m_acnfw(tx, ty, p, ctx):
    q = 1.0 - p[4]
    bb, tt = calc_bbtt_nfw(p[1], p[6], ctx)
    s, a = _acnfw_coeffs(p[7])
    return _cse_lens(tx, ty, p[2], p[3], tt / jnp.sqrt(q), q, p[5], bb, s, a)


MODEL_FUNCS = {
    "gals": m_gals, "nfwpot": m_nfwpot, "sie": m_sie, "jaffe": m_jaffe, "point": m_point,
    "pert": m_pert, "clus3": m_clus3, "mpole": m_mpole, "hernpot": m_hernpot, "nfw": m_nfw,
    "hern": m_hern, "powpot": m_powpot, "pow": m_pow, "gnfwpot": m_gnfwpot, "gnfw": m_gnfw,
    "serspot": m_serspot, "sers": m_sers, "tnfwpot": m_tnfwpot, "tnfw": m_tnfw,
    "einpot": m_einpot, "ein": m_ein, "anfw": m_anfw, "ahern": m_ahern, "crline": m_crline,
    "gaupot": m_gaupot, "acnfw": m_acnfw,
}
assert set(MODEL_FUNCS) == set(LENS_MODELS)

# models whose p[1] is a fiducial source redshift (fac_pert normalisation)
ZSFID_MODELS = {"pert", "clus3", "mpole", "powpot", "pow", "crline", "gaupot"}


def lens_eval(model, tx, ty, p, ctx):
    tx = jnp.asarray(tx, dtype=float)
    ty = jnp.asarray(ty, dtype=float)
    tx, ty = jnp.broadcast_arrays(tx, ty)
    out = MODEL_FUNCS[model](tx, ty, p, ctx)
    return tuple(jnp.broadcast_to(o, tx.shape) for o in out)


# ---------------------------------------------------------------------------
# parameter validity (checkmodelpar_* in mass.c).  Returns a boolean scalar.


def param_valid(model, p):
    e = p[4]
    ok_e = (e >= 0.0) & (e < 1.0)
    if model == "sie":
        return (p[1] >= 0.0) & ok_e & (p[6] >= 0.0)
    if model == "jaffe":
        return (p[1] >= 0.0) & (p[7] >= 0.0) & (p[6] > 0.0) & ok_e
    if model == "point":
        return p[1] >= 0.0
    if model in ("nfwpot", "nfw", "anfw"):
        return (p[1] > 0.0) & ok_e & (p[6] > 0.0)
    if model in ("hernpot", "hern", "ahern"):
        return (p[1] >= 0.0) & ok_e & (p[6] > 0.0)
    if model in ("powpot", "pow"):
        return ok_e & (p[6] > 0.0) & (p[7] > 1.0) & (p[7] < 3.0)
    if model in ("gnfwpot", "gnfw"):
        return (p[1] > 0.0) & ok_e & (p[6] > 0.0) & (p[7] >= 0.0) & (p[7] <= 2.0)
    if model in ("serspot", "sers"):
        return (p[1] >= 0.0) & ok_e & (p[6] > 0.0) & (p[7] >= 0.06) & (p[7] <= 20.0)
    if model in ("tnfwpot", "tnfw"):
        return (p[1] > 0.0) & ok_e & (p[6] > 0.0) & (p[7] > 0.0)
    if model in ("einpot", "ein"):
        return (p[1] > 0.0) & ok_e & (p[6] > 0.0) & (p[7] >= 0.02) & (p[7] <= 1.0)
    if model == "gaupot":
        return ok_e & (p[6] > 0.0)
    if model == "mpole":
        return p[6] > 0.0
    if model == "acnfw":
        return (p[1] > 0.0) & ok_e & (p[6] > 0.0) & (p[7] >= 0.0) & (p[7] < 100.0)
    return jnp.asarray(True)
