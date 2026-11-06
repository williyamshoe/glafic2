"""Point-source image finder (port of poi_set_table / findimg in point.c).

This is a faithful re-implementation of glafic's adaptive-mesh algorithm.
It is *not* jittable (the mesh has data-dependent size) -- it drives batched
jitted lens-model evaluations from NumPy.  Image positions returned here are
converged solutions of the lens equation; differentiable image positions are
obtained from them with an implicit-function Newton step (see ``point.py``).

The mesh values are stored in float32, exactly as in the C code, so that the
triangle tests (and therefore the number of images found) agree with glafic.
"""

import numpy as np


def dp_lev(pix_poi, lev):
    return pix_poi * 0.5 ** lev


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _bucket(n):
    m = 64
    while m < n:
        m *= 2
    return m


class _Evaluator:
    """Wrap a jitted lens function so it is called with bucketed sizes."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, y):
        n = x.size
        if n == 0:
            return {k: np.zeros(0) for k in ("ax", "ay", "kap", "gam1", "gam2", "rot", "muinv", "td")}
        m = _bucket(n)
        xp = np.zeros(m)
        yp = np.zeros(m)
        xp[:n] = x
        yp[:n] = y
        out = self.fn(xp, yp)
        return {k: np.asarray(v)[:n] for k, v in out.items()}


class ImageFinder:
    """Adaptive-mesh image finder for one source redshift / parameter set.

    Parameters
    ----------
    lens_fn : callable (x, y) -> dict with ax, ay, kap, gam1, gam2, rot, muinv, td
    planes_fn : callable (x, y) -> list over planes of (alp1, alp2, A11, A12, A21, A22)
        (only needed for multi-plane lens-centre assignment; may be None)
    settings : glafic Settings
    centers0 : (N, 2) lens centres on the first lens plane
    centers_planes : list over planes j>=1 of (M_j, 2) arrays of lens centres
    """

    def __init__(self, lens_fn, settings, centers0, planes_fn=None, centers_planes=None):
        self.s = settings
        self.lens = _Evaluator(lens_fn)
        self.planes_fn = planes_fn
        self.centers = [np.asarray(c, dtype=float).reshape(-1, 2) for c in [centers0]]
        self.centers_planes = centers_planes or []
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        s = self.s
        nx, ny = s.nx_poi, s.ny_poi
        pix = s.pix_poi
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
        gx = (s.xmin + pix * (ii + 0.5)).ravel()
        gy = (s.ymin + pix * (jj + 0.5)).ravel()
        out = self.lens(gx, gy)
        cent = [np.asarray(self.centers[0])]
        if self.planes_fn is not None and len(self.centers_planes) > 0:
            pl = self.planes_fn(gx, gy)
            rr = (s.dr_lens_center * pix) ** 2
            for j, cj in enumerate(self.centers_planes, start=1):
                if cj is None or len(cj) == 0:
                    continue
                a1, a2, m11, m12, m21, m22 = [np.asarray(v) for v in pl[j]]
                ttx = gx - a1
                tty = gy - a2
                det = m11 * m22 - m12 * m21 + s.imag_ceil
                for c in np.asarray(cj).reshape(-1, 2):
                    dx = ttx - c[0]
                    dy = tty - c[1]
                    ddx = (m22 * dx - m12 * dy) / det
                    ddy = (-m12 * dx + m11 * dy) / det
                    sel = (ddx * ddx + ddy * ddy) < rr
                    if np.any(sel):
                        cent.append(np.stack([gx[sel], gy[sel]], axis=1))
        self.lens_centers = np.concatenate(cent, axis=0) if len(cent) else np.zeros((0, 2))

        f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)
        defx = f32(out["ax"]).reshape(ny, nx)
        defy = f32(out["ay"]).reshape(ny, nx)
        smag = f32(out["muinv"]).reshape(ny, nx)
        # level-0 boxes, k = i + j*(nx-1); corners 0:(i,j) 1:(i+1,j) 2:(i,j+1) 3:(i+1,j+1)
        bi, bj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
        bi = bi.ravel()
        bj = bj.ravel()

        def corners(a):
            return np.stack([a[bj, bi], a[bj, bi + 1], a[bj + 1, bi], a[bj + 1, bi + 1]], axis=1)

        lev0 = dict(x=s.xmin + pix * (bi + 0.5), y=s.ymin + pix * (bj + 0.5),
                    dx=corners(defx), dy=corners(defy), mu=corners(smag),
                    kref=None, flag=np.zeros(bi.size, dtype=bool))
        self.levels = [lev0]
        for lev in range(1, s.maxlev):
            prev = self.levels[-1]
            dpp = dp_lev(pix, lev - 1)
            sgn = (prev["mu"] <= 0.0)
            stot = (sgn * np.array([1, 2, 4, 8])).sum(axis=1)
            crit = (stot != 0) & (stot != 15)
            im = np.abs(prev["mu"])
            magflag = np.any((im > s.poi_imag_max) | (im < s.poi_imag_min), axis=1)
            cflag = self._check_mesh_center(prev["x"], prev["y"], dpp, -1.0, 2.0)
            flag = crit | magflag | cflag
            prev["flag"] = flag
            par = np.nonzero(flag)[0]
            dp = dp_lev(pix, lev)
            px = prev["x"][par]
            py = prev["y"][par]
            # unit 3x3 grid; new evaluation points 1,3,4,5,7
            offs = {1: (1, 0), 3: (0, 1), 4: (1, 1), 5: (2, 1), 7: (1, 2)}
            ex = np.concatenate([px + dp * offs[u][0] for u in (1, 3, 4, 5, 7)])
            ey = np.concatenate([py + dp * offs[u][1] for u in (1, 3, 4, 5, 7)])
            o = self.lens(ex, ey)
            npar = par.size
            unit = {}
            for c, u in zip(range(4), (0, 2, 6, 8)):
                unit[u] = (prev["dx"][par, c], prev["dy"][par, c], prev["mu"][par, c])
            for n, u in enumerate((1, 3, 4, 5, 7)):
                sl = slice(n * npar, (n + 1) * npar)
                unit[u] = (f32(o["ax"][sl]), f32(o["ay"][sl]), f32(o["muinv"][sl]))
            num2 = [(0, 1, 3, 4), (1, 2, 4, 5), (3, 4, 6, 7), (4, 5, 7, 8)]
            cdx = np.zeros((npar, 4, 4))
            cdy = np.zeros((npar, 4, 4))
            cmu = np.zeros((npar, 4, 4))
            for ch in range(4):
                for c in range(4):
                    u = num2[ch][c]
                    cdx[:, ch, c] = unit[u][0]
                    cdy[:, ch, c] = unit[u][1]
                    cmu[:, ch, c] = unit[u][2]
            chx = px[:, None] + dp * np.array([0, 1, 0, 1])[None, :]
            chy = py[:, None] + dp * np.array([0, 0, 1, 1])[None, :]
            self.levels.append(dict(x=chx.ravel(), y=chy.ravel(), dx=cdx.reshape(-1, 4), dy=cdy.reshape(-1, 4),
                                    mu=cmu.reshape(-1, 4), kref=np.repeat(par, 4),
                                    flag=np.zeros(4 * npar, dtype=bool)))
        # child lookup for xytolev: parent index -> first child index, per level
        self._child = []
        for lev in range(1, len(self.levels)):
            kref = self.levels[lev]["kref"]
            first = {}
            for j in range(0, kref.size, 4):
                first[int(kref[j])] = j
            self._child.append(first)

    def _check_mesh_center(self, tx, ty, dp, r1, r2):
        c = self.lens_centers
        if c.shape[0] == 0:
            return np.zeros(tx.shape, dtype=bool)
        f = np.zeros(tx.shape, dtype=bool)
        for x, y in c:
            f |= (x >= tx + r1 * dp) & (x <= tx + r2 * dp) & (y >= ty + r1 * dp) & (y <= ty + r2 * dp)
        return f

    def _xytolev(self, x, y):
        s = self.s
        pix = s.pix_poi
        i = int((x - s.xmin) / pix - 0.5)
        j = int((y - s.ymin) / pix - 0.5)
        k0 = i + j * (s.nx_poi - 1)
        if k0 < 0 or k0 >= self.levels[0]["x"].size:
            return -1
        if not self.levels[0]["flag"][k0]:
            return 0
        dx = x - (s.xmin + pix * (i + 0.5))
        dy = y - (s.ymin + pix * (j + 0.5))
        p0 = pix
        r = 0
        k1 = 0
        while True:
            r += 1
            p0 *= 0.5
            ii = 0
            if dx >= p0:
                ii += 1
                dx -= p0
            if dy >= p0:
                ii += 2
                dy -= p0
            if r >= len(self.levels):
                return r - 1
            first = self._child[r - 1].get(k0)
            if first is not None:
                k1 = first + ii
            k0 = k1
            if not self.levels[r]["flag"][k1]:
                return r

    # ------------------------------------------------------------------
    def find(self, xs, ys):
        """Return array (n, 4) of [x, y, mag, td] like glafic's findimg."""
        s = self.s
        pix = s.pix_poi
        cands = []  # (lev, k, sub, x, y, dp)
        gdx = [0.4, 2.4, 0.4, -0.4]
        gdy = [-0.4, 0.4, 2.4, 0.4]
        gdxs1 = [0.0, 2.0, 0.0, 0.0]
        gdys1 = [0.0, 0.0, 2.0, 0.0]
        gdxs2 = [2.0, 2.0, 2.0, 0.0]
        gdys2 = [0.0, 2.0, 2.0, 2.0]
        gdxs3 = [1.0, 2.0, 1.0, 0.0]
        gdys3 = [0.0, 1.0, 2.0, 1.0]
        gj1 = [0, 1, 2, 0]
        gl1 = [0, 1, 2, 0]
        gj2 = [1, 3, 3, 2]
        gl2 = [1, 3, 3, 2]
        gj3 = [0, 1, 2, 0]
        gl3 = [1, 3, 3, 2]
        for lev, L in enumerate(self.levels):
            dp = dp_lev(pix, lev)
            xx = L["x"]
            yy = L["y"]
            DX = L["dx"]
            DY = L["dy"]
            nb = xx.size
            # gap check -------------------------------------------------
            if lev > 0:
                for k in range(0, nb, 4):
                    x0 = xx[k]
                    y0 = yy[k]
                    for i in range(4):
                        if lev > self._xytolev(x0 + gdx[i] * dp, y0 + gdy[i] * dp):
                            g1x = x0 + gdxs1[i] * dp - DX[k + gj1[i], gl1[i]]
                            g1y = y0 + gdys1[i] * dp - DY[k + gj1[i], gl1[i]]
                            g2x = x0 + gdxs2[i] * dp - DX[k + gj2[i], gl2[i]]
                            g2y = y0 + gdys2[i] * dp - DY[k + gj2[i], gl2[i]]
                            g3x = x0 + gdxs3[i] * dp - DX[k + gj3[i], gl3[i]]
                            g3y = y0 + gdys3[i] * dp - DY[k + gj3[i], gl3[i]]
                            g4x = x0 + dp - DX[k, 3]
                            g4y = y0 + dp - DY[k, 3]
                            if _cross(g2x - g1x, g2y - g1y, g3x - g1x, g3y - g1y) * \
                                    _cross(g2x - g1x, g2y - g1y, g4x - g1x, g4y - g1y) > 0.0:
                                d12 = _cross(xs - g1x, ys - g1y, xs - g2x, ys - g2y)
                                d23 = _cross(xs - g2x, ys - g2y, xs - g3x, ys - g3y)
                                d31 = _cross(xs - g3x, ys - g3y, xs - g1x, ys - g1y)
                                if (d12 >= 0 and d23 >= 0 and d31 >= 0) or (d12 <= 0 and d23 <= 0 and d31 <= 0):
                                    cands.append((lev, k, i, x0 + gdxs3[i] * dp, y0 + gdys3[i] * dp, dp))
            # standard triangles ---------------------------------------
            leaf = ~L["flag"]
            d1x = xs - (xx - DX[:, 0])
            d1y = ys - (yy - DY[:, 0])
            d2x = xs - (xx + dp - DX[:, 3])
            d2y = ys - (yy + dp - DY[:, 3])
            d3x = xs - (xx + dp - DX[:, 1])
            d3y = ys - (yy - DY[:, 1])
            d12 = _cross(d1x, d1y, d2x, d2y)
            d23 = _cross(d2x, d2y, d3x, d3y)
            d31 = _cross(d3x, d3y, d1x, d1y)
            t1 = leaf & (((d12 >= 0) & (d23 >= 0) & (d31 >= 0)) | ((d12 <= 0) & (d23 <= 0) & (d31 <= 0)))
            e3x = xs - (xx - DX[:, 2])
            e3y = ys - (yy + dp - DY[:, 2])
            e23 = _cross(d2x, d2y, e3x, e3y)
            e31 = _cross(e3x, e3y, d1x, d1y)
            t2 = leaf & (((d12 > 0) & (e23 > 0) & (e31 > 0)) | ((d12 < 0) & (e23 < 0) & (e31 < 0)))
            for k in np.nonzero(t1)[0]:
                cands.append((lev, int(k), 4, xx[k] + 0.667 * dp, yy[k] + 0.333 * dp, dp))
            for k in np.nonzero(t2)[0]:
                cands.append((lev, int(k), 5, xx[k] + 0.333 * dp, yy[k] + 0.667 * dp, dp))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        cands = cands[:50]
        if not cands:
            return np.zeros((0, 4))
        xi = np.array([c[3] for c in cands])
        yi = np.array([c[4] for c in cands])
        dpi = np.array([c[5] for c in cands])
        xo = xi.copy()
        yo = yi.copy()
        active = np.ones(xi.size, dtype=bool)
        ite = 0
        while np.any(active):
            ite += 1
            o = self.lens(xi[active], yi[active])
            pxx = o["kap"] + o["gam1"]
            pyy = o["kap"] - o["gam1"]
            pxy = o["gam2"] + o["rot"]
            pyx = o["gam2"] - o["rot"]
            ff = xs - xi[active] + o["ax"]
            gg = ys - yi[active] + o["ay"]
            mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pyx
            idx = np.nonzero(active)[0]
            xi[idx] += ((1.0 - pyy) * ff + pxy * gg) / mm
            yi[idx] += ((1.0 - pxx) * gg + pyx * ff) / mm
            conv = ~((np.abs(ff) > s.max_poi_tol) | (np.abs(gg) > s.max_poi_tol))
            stop = conv | (ite > s.nmax_poi_ite)
            active[idx[stop]] = False
        moved = (xo - xi) ** 2 + (yo - yi) ** 2 > 2.0 * dpi * dpi
        xi = np.where(moved, xo, xi)
        yi = np.where(moved, yo, yi)
        fff = np.where(moved, 1, 0)
        o = self.lens(xi, yi)
        mag = 1.0 / (o["muinv"] + s.imag_ceil)
        td = o["td"].copy()
        n = xi.size
        for a in range(n):
            for b in range(a + 1, n):
                mm = abs(mag[a] * mag[b])
                dis2 = ((xi[a] - xi[b]) ** 2 + (yi[a] - yi[b]) ** 2) / mm
                if dis2 <= 10.0 * s.max_poi_tol ** 2 and fff[a] >= 0:
                    fff[a] = -1
        fff[fff == 1] = -1
        keep = fff >= 0
        if np.any(keep):
            td = td - td[keep].min()
        return np.stack([xi[keep], yi[keep], mag[keep], td[keep]], axis=1)
