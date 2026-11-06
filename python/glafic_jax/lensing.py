"""Multi-plane lens equation (port of lensmodel / multi_calc_all in mass.c).

The *structure* of the lens system (which models, how lenses are grouped into
planes, how many planes lie in front of each source) is static.  Parameter
values (including lens redshifts and cosmology) are traced, so everything is
differentiable.  Plane grouping/ordering is fixed when the system is built,
which matches glafic's behaviour as long as redshifts do not cross each other
during sampling.
"""

from typing import NamedTuple, Any

import numpy as np
import jax.numpy as jnp

from . import cosmology as cm
from .constants import TOL_ZLPL
from .lens_models import PlaneContext, lens_eval


class LensOut(NamedTuple):
    """Equivalent of glafic's ``pout`` array (plus a couple of extras)."""
    ax: Any
    ay: Any
    td: Any       # time delay [day] (without zero point)
    kap: Any
    gam1: Any
    gam2: Any
    muinv: Any
    rot: Any
    phi: Any      # single-plane potential (sum of potentials on plane 0) if nlp == 1
    def_last: Any  # (2, ...) deflection accumulated before the last lens plane


def gen_lensplanes(zl_values):
    """Group lenses into planes exactly like gen_lensplane() in init.c.

    Returns ``(plane_of_lens, first_lens_of_plane, zl_planes)`` with planes
    sorted by redshift.
    """
    zl_values = [float(z) for z in zl_values]
    lpl = [0]
    zpl = [zl_values[0]]
    first = [0]
    for i in range(1, len(zl_values)):
        for j in range(i):
            if abs(zl_values[i] - zl_values[j]) < TOL_ZLPL:
                lpl.append(lpl[j])
                break
        else:
            lpl.append(len(zpl))
            zpl.append(zl_values[i])
            first.append(i)
    order = np.argsort(zpl, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    plane_of_lens = [int(rank[k]) for k in lpl]
    first_lens = [first[k] for k in order]
    return plane_of_lens, first_lens, [zpl[k] for k in order]


class LensSystem:
    """Static description of all lenses."""

    def __init__(self, models, zl_init, settings, gals=None):
        self.models = list(models)
        self.settings = settings
        self.gals = None if gals is None else np.asarray(gals, dtype=float)
        self.plane_of_lens, self.first_lens, self.zl_planes_init = gen_lensplanes(zl_init)
        self.num_planes = len(self.zl_planes_init)
        self.lenses_in_plane = [[i for i, p in enumerate(self.plane_of_lens) if p == k]
                                for k in range(self.num_planes)]

    # ------------------------------------------------------------------
    def nlp_for(self, zs):
        """Number of lens planes in front of a source at (concrete) redshift zs."""
        n = 0
        for i, z in enumerate(self.zl_planes_init):
            if z < float(zs):
                n = i + 1
        return n

    def zl_planes(self, lens_params):
        return jnp.stack([lens_params[i, 0] for i in self.first_lens])

    def distances(self, lens_params, zs, cosmo, nlp):
        """Distances needed for ray tracing (set_distance_lpl_init/zs)."""
        s = self.settings
        zl = self.zl_planes(lens_params)[:nlp]
        dos = cm.dis_angulard(0.0, zs, cosmo)
        dol = cm.dis_angulard(0.0, zl, cosmo)
        dls = cm.dis_angulard(zl, zs, cosmo)
        delome = cm.deltaomega(zl, cosmo, s.flag_hodensity, s.hodensity)
        beta = {}
        tdel = {}
        for j in range(1, nlp):
            dj = dol[j]
            for i in range(j):
                dij = cm.dis_angulard(zl[i], zl[j], cosmo)
                beta[(i, j)] = (dij / dj) * dos / dls[i]
                tdel[(i, j)] = cm.tdelay_fac(zl[i], dj, dol[i], dij, cosmo)
        tlast = cm.tdelay_fac(zl[nlp - 1], dos, dol[nlp - 1], dls[nlp - 1], cosmo)
        for i in range(nlp):
            beta[(i, nlp)] = 1.0
            tdel[(i, nlp)] = tlast
        return dict(zl=zl, dos=dos, dol=dol, dls=dls, delome=delome, beta=beta, tdel=tdel)

    def plane_context(self, dist, k, cosmo):
        return PlaneContext(dist["zl"][k], dist["dos"], dist["dol"][k], dist["dls"][k],
                            dist["delome"][k], cosmo, self.settings, self.gals)

    def plane_sum(self, tx, ty, lens_params, ctx, k):
        tot = None
        for i in self.lenses_in_plane[k]:
            r = lens_eval(self.models[i], tx, ty, lens_params[i], ctx)
            tot = r if tot is None else tuple(a + b for a, b in zip(tot, r))
        return tot

    # ------------------------------------------------------------------
    def ray_shoot(self, tx, ty, lens_params, zs, cosmo, nlp=None, dist=None, return_planes=False):
        """Full lens model at image-plane positions (lensmodel(..., lensid=0))."""
        if nlp is None:
            nlp = self.nlp_for(zs)
        if nlp == 0:
            raise ValueError("source is in front of all lens planes")
        if dist is None:
            dist = self.distances(lens_params, zs, cosmo, nlp)
        tx = jnp.asarray(tx, dtype=float)
        ty = jnp.asarray(ty, dtype=float)
        tx, ty = jnp.broadcast_arrays(tx, ty)
        beta = dist["beta"]
        xs = [tx]
        ys = [ty]
        alps = []
        Us = []
        As = [(jnp.ones_like(tx), jnp.zeros_like(tx), jnp.zeros_like(tx), jnp.ones_like(tx))]
        phis = []
        for j in range(nlp):
            ctx = self.plane_context(dist, j, cosmo)
            ax, ay, phi, kap, g1, g2 = self.plane_sum(xs[j], ys[j], lens_params, ctx, j)
            alps.append((ax, ay))
            Us.append((kap + g1, g2, g2, kap - g1))
            phis.append(phi)
            if j < nlp - 1:
                a1 = 0.0
                a2 = 0.0
                m = [0.0, 0.0, 0.0, 0.0]
                for i in range(j + 1):
                    b = beta[(i, j + 1)]
                    a1 = a1 + b * alps[i][0]
                    a2 = a2 + b * alps[i][1]
                    ua = _matmul(Us[i], As[i])
                    m = [m[k] + b * ua[k] for k in range(4)]
                xs.append(tx - a1)
                ys.append(ty - a2)
                As.append((1.0 - m[0], -m[1], -m[2], 1.0 - m[3]))
        axt = sum(a[0] for a in alps)
        ayt = sum(a[1] for a in alps)
        xs.append(tx - axt)
        ys.append(ty - ayt)
        td = 0.0
        for i in range(nlp):
            dd = (xs[i] - xs[i + 1]) ** 2 + (ys[i] - ys[i + 1]) ** 2
            td = td + dist["tdel"][(i, i + 1)] * (0.5 * dd - beta[(i, i + 1)] * phis[i])
        mat = [0.0, 0.0, 0.0, 0.0]
        for i in range(nlp):
            ua = _matmul(Us[i], As[i])
            mat = [mat[k] + ua[k] for k in range(4)]
        kap = 0.5 * (mat[0] + mat[3])
        g1 = 0.5 * (mat[0] - mat[3])
        g2 = 0.5 * (mat[1] + mat[2])
        muinv = (1.0 - mat[0]) * (1.0 - mat[3]) - mat[1] * mat[2]
        rot = 0.5 * (mat[1] - mat[2])
        def_last = jnp.stack([tx - xs[nlp - 1], ty - ys[nlp - 1]])
        out = LensOut(axt, ayt, td, kap, g1, g2, muinv, rot, phis[0], def_last)
        if return_planes:
            # def_lpl[j] = (alp1, alp2, A11, A12, A21, A22) as in mass.c
            planes = [(tx - xs[j], ty - ys[j]) + tuple(As[j]) for j in range(nlp)]
            return out, planes
        return out

    def single_lens(self, i, tx, ty, lens_params, zs, cosmo):
        """lensmodel(..., lensid = i + 1): one lens on its own plane."""
        k = self.plane_of_lens[i]
        nlp = k + 1
        dist = self.distances(lens_params, zs, cosmo, max(nlp, self.nlp_for(zs)))
        ctx = self.plane_context(dist, k, cosmo)
        ax, ay, phi, kap, g1, g2 = lens_eval(self.models[i], tx, ty, lens_params[i], ctx)
        tdf = cm.tdelay_fac(ctx.zl, ctx.dos, ctx.dol, ctx.dls, cosmo)
        td = tdf * (0.5 * (ax * ax + ay * ay) - phi)
        muinv = (1.0 - kap) ** 2 - (g1 * g1 + g2 * g2)
        return LensOut(ax, ay, td, kap, g1, g2, muinv, jnp.zeros_like(ax), phi,
                       jnp.zeros((2,) + jnp.shape(ax)))


def _matmul(U, A):
    return (U[0] * A[0] + U[1] * A[2], U[0] * A[1] + U[1] * A[3],
            U[2] * A[0] + U[3] * A[2], U[2] * A[1] + U[3] * A[3])
