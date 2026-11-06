"""Fixed-order quadrature rules (static nodes so everything stays jittable)."""

import functools

import numpy as np
import jax.numpy as jnp


@functools.lru_cache(maxsize=None)
def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


@functools.lru_cache(maxsize=None)
def composite_gl(panels, nodes):
    """Nodes/weights of a composite Gauss-Legendre rule on [0, 1]."""
    x, w = gauss_legendre(nodes)
    edges = np.linspace(0.0, 1.0, panels + 1)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (b - a) * x + 0.5 * (a + b))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def integrate(f, a, b, n=64):
    """Gauss-Legendre integral of ``f`` from ``a`` to ``b`` (broadcasting).

    ``a`` and ``b`` may be arrays; the quadrature axis is appended last.
    """
    x, w = gauss_legendre(n)
    a = jnp.asarray(a)[..., None]
    b = jnp.asarray(b)[..., None]
    h = 0.5 * (b - a)
    t = a + h * (x + 1.0)
    return jnp.sum(f(t) * w, axis=-1) * h[..., 0]


# --------------------------------------------------------------------------
# 21-point Gauss-Kronrod rule as used by gsl_integration_qng when the error
# tolerance is huge (glafic's gsl_qgaus).  Reproducing it keeps lens
# potentials (and therefore time delays) identical to the C code.

_X1 = np.array([0.973906528517171720077964012084452, 0.865063366688984510732096688423493,
                0.679409568299024406234327365114874, 0.433395394129247190799265943165784,
                0.148874338981631210884826001129720])
_X2 = np.array([0.995657163025808080735527280689003, 0.930157491355708226001207180059508,
                0.780817726586416897063717578345042, 0.562757134668604683339000099272694,
                0.294392862701460198131126603103866])
_W21A = np.array([0.032558162307964727478818972459390, 0.075039674810919952767043140916190,
                  0.109387158802297641899210590325805, 0.134709217311473325928054001771707,
                  0.147739104901338491374841515972068])
_W21B = np.array([0.011694638867371874278064396062192, 0.054755896574351996031381300244580,
                  0.093125454583697605535065465083366, 0.123491976262065851077208980223048,
                  0.142775938577060080797094273138717, 0.149445554002916905664936468389821])

GK21_X = np.concatenate([[0.0], _X1, -_X1, _X2, -_X2])
GK21_W = np.concatenate([[_W21B[5]], _W21A, _W21A, _W21B[:5], _W21B[:5]])


def qng21(f, a, b):
    """glafic ``gsl_qgaus`` (single 21-point Kronrod rule)."""
    a = jnp.asarray(a)[..., None]
    b = jnp.asarray(b)[..., None]
    h = 0.5 * (b - a)
    c = 0.5 * (b + a)
    t = c + h * GK21_X
    return jnp.sum(f(t) * GK21_W, axis=-1) * h[..., 0]
