"""Point-source chi^2 (port of opt_point.c), differentiable with JAX.

Two chi^2 definitions are available, as in glafic:

* source plane (``chi2_splane 1``): fully differentiable, pure JAX;
* image plane (``chi2_splane 0``): model images are located with glafic's
  adaptive-mesh image finder on the host (``jax.pure_callback``), matched to
  the observed images exactly as in the C code, and then made differentiable
  by one Newton step of the lens equation (implicit-function theorem).  The
  value is identical to glafic; gradients are exact derivatives of it.

A third, pure-JAX option ``mode="tracked"`` solves the lens equation by
Newton iteration starting from the observed image positions.  It needs no
host callback (good for GPUs / vmap) but does not check the number of images.

Source positions are *parameters* here (glafic instead optimises them inside
every chi^2 evaluation).  ``profile_source_splane`` provides the profiled
version for the source-plane chi^2.
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import cosmology as cm
from .constants import COVERH_MPCH, DIS2_MIN_SET


def _lensout_dict(o):
    return dict(ax=o.ax, ay=o.ay, kap=o.kap, gam1=o.gam1, gam2=o.gam2, rot=o.rot,
                muinv=o.muinv, td=o.td)


def set_matrix(o, imag_ceil, mag_cap=0.0):
    """Magnification-tensor weights of the source-plane chi^2 (set_matrix in
    opt_point.c).  ``mag_cap > 0`` replaces 1/muinv by the smooth, bounded
    muinv / (muinv^2 + e^2) with e = 1 / (2 mag_cap), so |mu| <= mag_cap: this
    removes the infinite spikes of the source-plane chi^2 for images lying
    on critical curves (not in glafic; off by default)."""
    if mag_cap and mag_cap > 0.0:
        e = 0.5 / mag_cap
        norm = o.muinv / (o.muinv * o.muinv + e * e)
    else:
        norm = 1.0 / (o.muinv + imag_ceil)
    k, g1, g2, r = o.kap, o.gam1, o.gam2, o.rot
    a00 = norm * norm * ((1.0 - k + g1) ** 2 + (g2 - r) ** 2)
    a01 = 2.0 * norm * norm * (g2 * (1.0 - k) + g1 * r)
    a11 = norm * norm * ((1.0 - k - g1) ** 2 + (g2 + r) ** 2)
    m00 = norm * (1.0 - k + g1)
    m01 = norm * (g2 + r)
    m10 = norm * (g2 - r)
    m11 = norm * (1.0 - k - g1)
    return (a00, a01, a01, a11), (m00, m01, m10, m11)


def _flux_td_chi2(s, obs, mag, td, dis_lum, hubble, parity, valid=None):
    """Flux and time-delay chi^2 terms with analytic normalisation.

    Works with traced ``obs`` (for vmapping over sources); ``valid`` masks
    padded image slots."""
    obs = jnp.asarray(obs)
    parity = jnp.asarray(parity)
    valid = jnp.ones(obs.shape[0], dtype=bool) if valid is None else jnp.asarray(valid)
    f_obs, f_err = obs[:, 2], obs[:, 4]
    t_obs, t_err = obs[:, 5], obs[:, 6]
    hasf = valid & (f_err > 0.0)
    hast = valid & (t_err > 0.0)
    fe2 = jnp.where(hasf, f_err * f_err, 1.0)
    te2 = jnp.where(hast, t_err * t_err, 1.0)
    amag = jnp.abs(mag)
    if s.chi2_usemag == 0:
        f1 = jnp.sum(jnp.where(hasf, jnp.abs(f_obs * mag) / fe2, 0.0))
        f2 = jnp.sum(jnp.where(hasf, mag * mag / fe2, 0.0))
    else:
        f1 = jnp.sum(jnp.where(hasf, (f_obs + 2.5 * jnp.log10(amag)) / fe2, 0.0))
        f2 = jnp.sum(jnp.where(hasf, 1.0 / fe2, 0.0))
    fnorm = jnp.where(f2 > 0.0, f1 / jnp.where(f2 > 0.0, f2, 1.0), 1.0)
    if s.chi2_usedl == 0:
        if s.chi2_usemag == 0:
            r = jnp.abs(f_obs) - amag * fnorm
        elif s.chi2_usemag == -1:
            r = jnp.abs(f_obs) - amag
        else:
            r = f_obs + 2.5 * jnp.log10(amag) - fnorm
    else:
        r = jnp.abs(f_obs) - dis_lum / (hubble * jnp.sqrt(amag))
    c2f = jnp.sum(jnp.where(hasf, r * r / fe2, 0.0))
    # time-delay chi^2 with the zero point t0 profiled analytically.  glafic
    # input files commonly give the reference image a tiny error (e.g. 1e-6
    # day), so  sum w (r - t0)^2  with  t0 = sum w r / sum w  suffers from
    # catastrophic cancellation (w ~ 1e12).  Use the equivalent pairwise form
    #   r_k - t0 = sum_l (w_l / W) (r_k - r_l)
    # which is numerically stable (also for gradients).
    w = jnp.where(hast, 1.0 / te2, 0.0)
    W = jnp.sum(w)
    Ws = jnp.where(W > 0.0, W, 1.0)
    rr = t_obs - td
    dev = jnp.sum((w / Ws)[None, :] * (rr[:, None] - rr[None, :]), axis=1)
    c2t = jnp.sum(w * dev * dev)
    t0 = jnp.sum(w * rr) / Ws
    bad_parity = jnp.any(valid & (parity != 0) & (parity * mag < 0.0))
    c2f = jnp.where(bad_parity, s.chi2pen_parity, c2f)
    return c2f, c2t, fnorm, t0


def chi2prior_point(prior, i, xs, ys):
    c2 = 0.0
    for j, v in ((1, xs), (2, ys)):
        sig = prior["sig"][i, j]
        if sig > 0.0:
            c2 = c2 + (prior["med"][i, j] - v) ** 2 / (sig * sig)
    return c2


def _range_bad(prior, i, xs, ys):
    return ((xs < prior["min"][i, 1]) | (xs > prior["max"][i, 1]) |
            (ys < prior["min"][i, 2]) | (ys > prior["max"][i, 2]))


# ---------------------------------------------------------------------------
# source plane


def chi2_splane_single(model, P, i):
    """Return (c2tot, c2pos, c2flux, c2td, c2prior) for point source i."""
    s = model.settings
    obs = model.obs_point[i]["tab"]
    parity = model.obs_point[i]["parity"]
    zs = P["point"][i, 0]
    xs = P["point"][i, 1]
    ys = P["point"][i, 2]
    nlp = model.nlp_point[i]
    lens = model.lens_system
    cosmo = P["cosmo"]
    dist = lens.distances(P["lens"], zs, cosmo, nlp)
    ox = jnp.asarray(obs[:, 0])
    oy = jnp.asarray(obs[:, 1])
    o = lens.ray_shoot(ox, oy, P["lens"], zs, cosmo, nlp=nlp, dist=dist)
    amat, mumat = set_matrix(o, s.imag_ceil, getattr(s, 'mag_cap', 0.0))
    ux = ox - o.ax
    uy = oy - o.ay
    mag0 = 1.0 / (o.muinv + s.imag_ceil)

    def muinv_at(x, y):
        return lens.ray_shoot(x, y, P["lens"], zs, cosmo, nlp=nlp, dist=dist).muinv

    dmx = jax.vmap(jax.grad(muinv_at, argnums=0))(ox, oy)
    dmy = jax.vmap(jax.grad(muinv_at, argnums=1))(ox, oy)
    dmudx = -mag0 * mag0 * dmx
    dmudy = -mag0 * mag0 * dmy

    pe = obs[:, 3]
    hasp = pe > 0.0
    pe2 = np.where(hasp, pe * pe, 1.0)
    rx = ux - xs
    ry = uy - ys
    cc = (amat[0] * rx * rx + (amat[1] + amat[2]) * rx * ry + amat[3] * ry * ry) / pe2
    cc = jnp.where(cc < 0.0, s.chi2pen_nimg, cc)
    c2p = jnp.sum(jnp.where(hasp, cc, 0.0))
    dx = mumat[0] * rx + mumat[1] * ry
    dy = mumat[2] * rx + mumat[3] * ry
    mumod = mag0 - (dmudx * dx + dmudy * dy)
    tlast = dist["tdel"][(nlp - 1, nlp)]
    tdmod = o.td - tlast * ((ux - (ox - o.def_last[0])) * rx + (uy - (oy - o.def_last[1])) * ry)
    dis_lum = COVERH_MPCH * cm.dis_luminosity(0.0, zs, cosmo)
    c2f, c2t, _, _ = _flux_td_chi2(s, obs, mumod, tdmod, dis_lum, cosmo[3], parity)
    c2pr = chi2prior_point(model.prior_point, i, xs, ys)
    tot = c2p + c2f + c2t + c2pr
    bad = _range_bad(model.prior_point, i, xs, ys)
    tot = jnp.where(bad, s.chi2pen_range, tot)
    return tot, c2p, c2f, c2t, c2pr


def profile_source_splane(model, P, i, n_newton=3):
    """Analytic weighted-least-squares source position (umod in opt_point.c),
    refined by a few Newton steps on the full source-plane chi^2."""
    s = model.settings
    obs = model.obs_point[i]["tab"]
    zs = P["point"][i, 0]
    nlp = model.nlp_point[i]
    ox = jnp.asarray(obs[:, 0])
    oy = jnp.asarray(obs[:, 1])
    o = model.lens_system.ray_shoot(ox, oy, P["lens"], zs, P["cosmo"], nlp=nlp)
    amat, _ = set_matrix(o, s.imag_ceil)
    ux = ox - o.ax
    uy = oy - o.ay
    pe = obs[:, 3]
    w = np.where(pe > 0.0, 1.0 / np.where(pe > 0, pe * pe, 1.0), 0.0)
    a00 = jnp.sum(amat[0] * w)
    a01 = jnp.sum(amat[1] * w)
    a10 = jnp.sum(amat[2] * w)
    a11 = jnp.sum(amat[3] * w)
    b0 = jnp.sum((amat[0] * ux + amat[1] * uy) * w)
    b1 = jnp.sum((amat[2] * ux + amat[3] * uy) * w)
    pr = model.prior_point
    for j, (aa, bb) in ((1, (0, 0)), (2, (1, 1))):
        sig = pr["sig"][i, j]
        if sig > 0.0:
            if j == 1:
                a00 = a00 + 1.0 / sig ** 2
                b0 = b0 + pr["med"][i, j] / sig ** 2
            else:
                a11 = a11 + 1.0 / sig ** 2
                b1 = b1 + pr["med"][i, j] / sig ** 2
    det = a00 * a11 - a01 * a10
    u = jnp.stack([(a11 * b0 - a01 * b1) / det, (a00 * b1 - a10 * b0) / det])

    def f(uv):
        PP = dict(P)
        PP["point"] = P["point"].at[i, 1].set(uv[0]).at[i, 2].set(uv[1])
        return chi2_splane_single(model, PP, i)[0]

    for _ in range(n_newton):
        g = jax.grad(f)(u)
        H = jax.hessian(f)(u)
        u = u - jnp.linalg.solve(H + 1e-12 * jnp.eye(2), g)
    return u


# ---------------------------------------------------------------------------
# image plane


def _match_images(s, obs, parity, imgs):
    """Greedy nearest matching of chi2calc_point_iplane.  Returns
    (indices into imgs or None, status) where status 0=ok, 1=nimg penalty."""
    nobs = obs.shape[0]
    ni = imgs.shape[0]
    if (ni != nobs and s.chi2_checknimg == 1) or ni < nobs:
        return None, 1
    used = np.zeros(ni, dtype=bool)
    kk = np.zeros(nobs, dtype=int)
    for j in range(nobs):
        dm = DIS2_MIN_SET
        fp = False
        for k in range(ni):
            if not used[k]:
                if parity[j] == 0 or parity[j] * imgs[k, 2] > 0.0:
                    d2 = (imgs[k, 0] - obs[j, 0]) ** 2 + (imgs[k, 1] - obs[j, 1]) ** 2
                    if d2 < dm:
                        dm = d2
                        kk[j] = k
                        fp = True
        if not fp:
            return None, 1
        used[kk[j]] = True
    return kk, 0


def _newton_refine(lens, x, y, xs, ys, Pl, zs, cosmo, nlp, dist):
    """One Newton step from a (stop-gradient) solution: same value, exact
    implicit derivatives d x_img / d params."""
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    o = lens.ray_shoot(x, y, Pl, zs, cosmo, nlp=nlp, dist=dist)
    pxx = o.kap + o.gam1
    pyy = o.kap - o.gam1
    pxy = o.gam2 + o.rot
    pyx = o.gam2 - o.rot
    ff = xs - x + o.ax
    gg = ys - y + o.ay
    mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pyx
    return x + ((1.0 - pyy) * ff + pxy * gg) / mm, y + ((1.0 - pxx) * gg + pyx * ff) / mm


_ALPHAS = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])


def _newton_step_parity(lens, x, y, xs, ys, Pl, zs, cosmo, nlp, target):
    """Damped Newton step for the lens equation that never crosses a critical
    curve: the largest step fraction keeping sign(1/mu) == target is taken
    (target 0 = unconstrained).  Used to track each observed image on its own
    side of the critical curve (fold pairs would otherwise collapse onto one
    image).  All candidate step lengths are evaluated in parallel."""
    o = lens.ray_shoot(x, y, Pl, zs, cosmo, nlp=nlp)
    pxx = o.kap + o.gam1
    pyy = o.kap - o.gam1
    pxy = o.gam2 + o.rot
    pyx = o.gam2 - o.rot
    ff = xs - x + o.ax
    gg = ys - y + o.ay
    mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pyx
    dx = ((1.0 - pyy) * ff + pxy * gg) / mm
    dy = ((1.0 - pxx) * gg + pyx * ff) / mm
    a = jnp.asarray(_ALPHAS)[:, None]
    cx = x[None, :] + a * dx[None, :]
    cy = y[None, :] + a * dy[None, :]
    mu_c = lens.ray_shoot(cx, cy, Pl, zs, cosmo, nlp=nlp).muinv
    ok = (target[None, :] == 0) | (jnp.sign(mu_c) == target[None, :])
    ok = ok & jnp.isfinite(cx) & jnp.isfinite(cy)
    first = jnp.argmax(ok, axis=0)
    alpha = jnp.where(jnp.any(ok, axis=0), jnp.asarray(_ALPHAS)[first], 0.0)
    return x + alpha * dx, y + alpha * dy


def chi2_iplane_single(model, P, i, mode="glafic", n_track=20):
    s = model.settings
    obs = model.obs_point[i]["tab"]
    parity = model.obs_point[i]["parity"]
    nobs = obs.shape[0]
    zs = P["point"][i, 0]
    xs = P["point"][i, 1]
    ys = P["point"][i, 2]
    nlp = model.nlp_point[i]
    lens = model.lens_system
    cosmo = P["cosmo"]
    dist = lens.distances(P["lens"], zs, cosmo, nlp)

    if mode == "glafic":
        def host(lp, cp, z, x, y):
            imgs = model.find_images_numeric(np.asarray(lp), np.asarray(cp), float(z), float(x), float(y),
                                             point_index=i)
            kk, st = _match_images(s, obs, parity, imgs)
            if st != 0:
                return np.zeros((nobs, 2)), np.int32(1)
            return imgs[kk, :2].astype(np.float64), np.int32(0)

        sg = jax.lax.stop_gradient
        shapes = (jax.ShapeDtypeStruct((nobs, 2), jnp.float64), jax.ShapeDtypeStruct((), jnp.int32))
        pos, status = jax.pure_callback(host, shapes, sg(P["lens"]), sg(cosmo), sg(zs), sg(xs), sg(ys),
                                        vmap_method="sequential")
        x0 = pos[:, 0]
        y0 = pos[:, 1]
    else:  # tracked: Newton from the observed positions
        sg = jax.lax.stop_gradient
        x0 = jnp.asarray(obs[:, 0])
        y0 = jnp.asarray(obs[:, 1])

        def step(carry, _):
            x, y = carry
            return _newton_refine(lens, x, y, sg(xs), sg(ys), sg(P["lens"]), sg(zs), sg(cosmo), nlp, None), None

        (x0, y0), _ = jax.lax.scan(step, (x0, y0), None, length=n_track)
        status = jnp.int32(0)

    xi, yi = _newton_refine(lens, x0, y0, xs, ys, P["lens"], zs, cosmo, nlp, dist)
    o = lens.ray_shoot(xi, yi, P["lens"], zs, cosmo, nlp=nlp, dist=dist)
    mag = 1.0 / (o.muinv + s.imag_ceil)
    pe = obs[:, 3]
    hasp = pe > 0.0
    pe2 = np.where(hasp, pe * pe, 1.0)
    d2 = (xi - obs[:, 0]) ** 2 + (yi - obs[:, 1]) ** 2
    c2p = jnp.sum(jnp.where(hasp, d2 / pe2, 0.0))
    dis_lum = COVERH_MPCH * cm.dis_luminosity(0.0, zs, cosmo)
    c2f, c2t, _, _ = _flux_td_chi2(s, obs, mag, o.td, dis_lum, cosmo[3], parity)
    c2pr = chi2prior_point(model.prior_point, i, xs, ys)
    tot = c2p + c2f + c2t + c2pr
    if mode == "tracked":
        # penalise two observed images converging to the same model image
        dd = (xi[:, None] - xi[None, :]) ** 2 + (yi[:, None] - yi[None, :]) ** 2
        dup = jnp.any((dd < 1e-10) & ~np.eye(nobs, dtype=bool))
        status = jnp.where(dup, 1, 0)
    tot = jnp.where(status != 0, s.chi2pen_nimg, tot)
    bad = _range_bad(model.prior_point, i, xs, ys)
    tot = jnp.where(bad, s.chi2pen_range, tot)
    return tot, c2p, c2f, c2t, c2pr


# ---------------------------------------------------------------------------
# batched (vmapped over sources) chi^2 for the pure-JAX modes.  Compile time
# no longer grows with the number of sources.


def batch_data(model):
    """Padded static arrays for all point sources that have observations."""
    idx = [i for i, ob in enumerate(model.obs_point) if ob is not None and ob["tab"].shape[0] > 0]
    if not idx:
        return None
    nmax = max(model.obs_point[i]["tab"].shape[0] for i in idx)
    S = len(idx)
    obs = np.zeros((S, nmax, 7))
    par = np.zeros((S, nmax), dtype=int)
    val = np.zeros((S, nmax), dtype=bool)
    for k, i in enumerate(idx):
        t = model.obs_point[i]["tab"]
        n = t.shape[0]
        obs[k, :n] = t
        obs[k, n:] = t[0]                 # pad with a real image (finite values)
        obs[k, n:, 2:] = 0.0              # ... but without any constraint
        par[k, :n] = model.obs_point[i]["parity"]
        val[k, :n] = True
    pr = model.prior_point
    ii = np.array(idx)
    return dict(idx=ii, obs=obs, parity=par, valid=val, track_parity=np.sign(par).astype(float),
                pmed=pr["med"][ii, 1:3], psig=pr["sig"][ii, 1:3],
                pmin=pr["min"][ii, 1:3], pmax=pr["max"][ii, 1:3],
                nlp=sorted({model.nlp_point[i] for i in idx}))


def _prior_xy(xy, pmed, psig):
    ps = jnp.where(psig > 0.0, psig, 1.0)
    return jnp.sum(jnp.where(psig > 0.0, (pmed - xy) ** 2 / (ps * ps), 0.0))


def chi2_point_batched(model, P, mode="splane", n_track=20, return_terms=False, source_mode="sample"):
    """Point-source chi^2 for all sources at once (source plane, or image plane
    with ``mode='tracked'``).  Requires all sources to have the same number of
    lens planes in front of them (always true for a single lens plane).

    source_mode
        'sample'      : source positions are taken from ``P``;
        'profile'     : source positions are replaced by the weighted
                        least-squares solution of the source-plane chi^2
                        (exact minimum for position-only constraints);
        'marginalize' : (source plane only) analytic marginalisation over the
                        source positions with flat priors:
                        chi^2 -> chi^2_min + ln det M  (M = sum A_k / sigma_k^2),
                        exact for position-only constraints.
    With ``return_terms`` the per-source terms and the source positions used
    are returned: ``(tot, c2pos, c2flux, c2td, c2prior, xy)``.
    """
    s = model.settings
    bd = model._point_batch
    if len(bd["nlp"]) != 1:
        raise ValueError("batched point chi^2 needs the same number of lens planes for all sources")
    nlp = bd["nlp"][0]
    lens = model.lens_system
    cosmo = P["cosmo"]
    Pl = P["lens"]
    zs_all = P["point"][bd["idx"], 0]
    xy_all = P["point"][bd["idx"], 1:3]
    sg = jax.lax.stop_gradient

    if source_mode == "marginalize" and mode != "splane":
        raise ValueError("source_mode='marginalize' requires the source-plane chi^2")
    nmax = bd["obs"].shape[1]
    S = len(bd["idx"])
    if mode == "glafic":
        if source_mode == "profile":
            xy_all = chi2_point_batched(model, P, mode="splane", return_terms=True, source_mode="profile")[5]
        # exact glafic image finding + greedy matching for all sources (host)
        def host(lp, cp, zz, xy):
            lp, cp, zz, xy = np.asarray(lp), np.asarray(cp), np.asarray(zz), np.asarray(xy)
            pos = np.array(bd["obs"][:, :, :2])
            st = np.zeros(S, dtype=np.int32)
            for k, i in enumerate(bd["idx"]):
                n = int(bd["valid"][k].sum())
                imgs = model.find_images_numeric(lp, cp, float(zz[k]), float(xy[k, 0]), float(xy[k, 1]),
                                                 point_index=int(i))
                kk, stat = _match_images(s, bd["obs"][k, :n], bd["parity"][k, :n], imgs)
                if stat != 0:
                    st[k] = 1
                else:
                    pos[k, :n] = imgs[kk, :2]
            return pos.astype(np.float64), st

        shapes = (jax.ShapeDtypeStruct((S, nmax, 2), jnp.float64), jax.ShapeDtypeStruct((S,), jnp.int32))
        gpos, gst = jax.pure_callback(host, shapes, sg(Pl), sg(cosmo), sg(zs_all), sg(xy_all),
                                      vmap_method="sequential")
    else:
        gpos = jnp.asarray(bd["obs"][:, :, :2])
        gst = jnp.zeros(S, dtype=jnp.int32)
    src_mode_inner = "sample" if mode == "glafic" else source_mode

    tpar = jnp.asarray(bd["track_parity"])

    def one(zs, xy, obs, parity, valid, pmed, psig, pmin, pmax, gp, gs, tp):
        dist = lens.distances(Pl, zs, cosmo, nlp)
        ox, oy = obs[:, 0], obs[:, 1]
        pe = obs[:, 3]
        hasp = valid & (pe > 0.0)
        pe2 = jnp.where(hasp, pe * pe, 1.0)
        dis_lum = COVERH_MPCH * cm.dis_luminosity(0.0, zs, cosmo)
        logdet = 0.0
        if mode == "splane" or src_mode_inner != "sample":
            o = lens.ray_shoot(ox, oy, Pl, zs, cosmo, nlp=nlp, dist=dist)
            amat, mumat = set_matrix(o, s.imag_ceil, getattr(s, 'mag_cap', 0.0))
            ux = ox - o.ax
            uy = oy - o.ay
        if src_mode_inner != "sample":
            w = jnp.where(hasp, 1.0 / pe2, 0.0)
            ps = jnp.where(psig > 0.0, psig, 1.0)
            wp = jnp.where(psig > 0.0, 1.0 / (ps * ps), 0.0)
            m00 = jnp.sum(w * amat[0]) + wp[0]
            m01 = jnp.sum(w * amat[1])
            m11 = jnp.sum(w * amat[3]) + wp[1]
            b0 = jnp.sum(w * (amat[0] * ux + amat[1] * uy)) + wp[0] * pmed[0]
            b1 = jnp.sum(w * (amat[2] * ux + amat[3] * uy)) + wp[1] * pmed[1]
            det = m00 * m11 - m01 * m01
            xy = jnp.stack([(m11 * b0 - m01 * b1) / det, (m00 * b1 - m01 * b0) / det])
            if source_mode == "marginalize":
                logdet = jnp.log(jnp.abs(det))
        xs, ys = xy[0], xy[1]
        if mode == "splane":
            mag0 = 1.0 / (o.muinv + s.imag_ceil)
            mu_at = lambda x, y: lens.ray_shoot(x, y, Pl, zs, cosmo, nlp=nlp, dist=dist).muinv
            dmx = jax.vmap(jax.grad(mu_at, argnums=0))(ox, oy)
            dmy = jax.vmap(jax.grad(mu_at, argnums=1))(ox, oy)
            rx = ux - xs
            ry = uy - ys
            cc = (amat[0] * rx * rx + (amat[1] + amat[2]) * rx * ry + amat[3] * ry * ry) / pe2
            cc = jnp.where(cc < 0.0, s.chi2pen_nimg, cc)
            c2p = jnp.sum(jnp.where(hasp, cc, 0.0))
            dx = mumat[0] * rx + mumat[1] * ry
            dy = mumat[2] * rx + mumat[3] * ry
            mag = mag0 + mag0 * mag0 * (dmx * dx + dmy * dy)
            tlast = dist["tdel"][(nlp - 1, nlp)]
            td = o.td - tlast * ((ux - (ox - o.def_last[0])) * rx + (uy - (oy - o.def_last[1])) * ry)
            status = jnp.int32(0)
        else:
            if mode == "glafic":
                x0, y0 = gp[:, 0], gp[:, 1]
            else:
                def step(c, _):
                    x, y = c
                    return _newton_step_parity(lens, x, y, sg(xs), sg(ys), sg(Pl), sg(zs), sg(cosmo), nlp, tp), None

                (x0, y0), _ = jax.lax.scan(step, (ox, oy), None, length=n_track)
            xi, yi = _newton_refine(lens, x0, y0, xs, ys, Pl, zs, cosmo, nlp, dist)
            o = lens.ray_shoot(xi, yi, Pl, zs, cosmo, nlp=nlp, dist=dist)
            mag = 1.0 / (o.muinv + s.imag_ceil)
            td = o.td
            d2 = (xi - ox) ** 2 + (yi - oy) ** 2
            c2p = jnp.sum(jnp.where(hasp, d2 / pe2, 0.0))
            n = ox.shape[0]
            dd = (xi[:, None] - xi[None, :]) ** 2 + (yi[:, None] - yi[None, :]) ** 2
            dup = jnp.any((dd < 1e-10) & ~jnp.eye(n, dtype=bool) & valid[:, None] & valid[None, :])
            # Newton did not converge onto an image (e.g. no image near the data)
            res = jnp.max(jnp.where(valid, jnp.abs(xi - x0) + jnp.abs(yi - y0), 0.0))
            if mode == "glafic":
                status = jnp.where((gs != 0) | ~jnp.all(jnp.isfinite(xi)), 1, 0)
            else:
                status = jnp.where(dup | (res > 1e-4) | ~jnp.all(jnp.isfinite(xi)), 1, 0)
        c2f, c2t, _, _ = _flux_td_chi2(s, obs, mag, td, dis_lum, cosmo[3], parity, valid)
        c2pr = _prior_xy(xy, pmed, psig)
        tot = c2p + c2f + c2t + c2pr + logdet
        tot = jnp.where(status != 0, s.chi2pen_nimg, tot)
        bad = jnp.any((xy < pmin) | (xy > pmax))
        tot = jnp.where(bad, s.chi2pen_range, tot)
        return tot, c2p, c2f, c2t, c2pr, xy

    out = jax.vmap(one)(zs_all, xy_all, jnp.asarray(bd["obs"]), jnp.asarray(bd["parity"]),
                        jnp.asarray(bd["valid"]), jnp.asarray(bd["pmed"]), jnp.asarray(bd["psig"]),
                        jnp.asarray(bd["pmin"]), jnp.asarray(bd["pmax"]), gpos, gst, tpar)
    if return_terms:
        return out
    return jnp.sum(out[0])


def chi2_point(model, P, mode=None):
    """Total point-source chi^2 (sum over sources with >= 1 observed image)."""
    s = model.settings
    tot = 0.0
    for i in range(len(model.obs_point)):
        if model.obs_point[i] is None or model.obs_point[i]["tab"].shape[0] == 0:
            continue
        if s.chi2_splane == 1:
            tot = tot + chi2_splane_single(model, P, i)[0]
        else:
            tot = tot + chi2_iplane_single(model, P, i, mode=mode or "glafic")[0]
    return tot
