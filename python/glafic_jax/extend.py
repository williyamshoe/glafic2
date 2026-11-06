"""Extended sources, PSF convolution and image chi^2 (source.c, extend.c, opt_extend.c).

Everything is differentiable.  Two sub-pixel integration modes:

* ``"glafic"`` (default) reproduces glafic: a pixel is subdivided into
  ``num_pixint^2`` (or ``(4 num_pixint)^2`` very close to the source centre)
  sub-pixels when it maps close to the source.  Because the switch depends on
  parameters, the model image has tiny jumps -- harmless for optimisation
  but they make HMC trajectories slightly less smooth.
* ``"uniform"`` supersamples every pixel on a fixed ``n_sub x n_sub`` grid
  (smooth in all parameters; recommended for HMC).
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import cosmology as cm
from .lens_models import bn_sers, bnn_sers, gam2n1_sers, D2R


# ---------------------------------------------------------------------------
# source profiles


def ucalc(dx, dy, e, pa):
    si = jnp.sin(-pa * D2R)
    co = jnp.cos(-pa * D2R)
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    return ddx * ddx / (1.0 - e) + (1.0 - e) * ddy * ddy


def _checkdis(x, y, x0, y0, r0, calcr0):
    rr = calcr0 * r0
    return (jnp.abs(x - x0) > rr) | (jnp.abs(y - y0) > rr)


def source_gauss(x, y, x0, y0, e, pa, r0, n, s):
    u = ucalc(x - x0, y - y0, e, pa)
    return jnp.where(_checkdis(x, y, x0, y0, r0, s.source_calcr0), 0.0, jnp.exp(-0.5 * u / (r0 * r0)))


def source_sersic(x, y, x0, y0, e, pa, r0, n, s):
    u = ucalc(x - x0, y - y0, e, pa)
    bn = bn_sers(n)
    v = jnp.where(u > 0.0, u / (r0 * r0), 1e-300)
    f = jnp.exp(-bn * jnp.power(v, 0.5 / n))
    f = jnp.where(u > 0.0, f, 1.0)
    return jnp.where(_checkdis(x, y, x0, y0, r0, s.source_calcr0), 0.0, f)


def source_tophat(x, y, x0, y0, e, pa, r0, n, s):
    u = ucalc(x - x0, y - y0, e, pa)
    f = jnp.where(u <= r0 * r0, 1.0, 0.0)
    return jnp.where(_checkdis(x, y, x0, y0, r0, s.source_calcr0), 0.0, f)


def source_moffat(x, y, x0, y0, e, pa, a, b, s):
    u = ucalc(x - x0, y - y0, e, pa)
    return jnp.where(_checkdis(x, y, x0, y0, a, s.source_calcr0), 0.0, jnp.power(1.0 + u / (a * a), -b))


def source_jaffe(x, y, x0, y0, e, pa, a, rco, s):
    rco = jnp.maximum(rco, s.smallcore)
    u = ucalc(x - x0, y - y0, e, pa)
    f = (1.0 / jnp.sqrt(rco * rco + u) - 1.0 / jnp.sqrt(a * a + u)) / ((1.0 / rco) - (1.0 / a))
    f = jnp.where(a > rco, f, 0.0)
    return jnp.where(_checkdis(x, y, x0, y0, a, s.source_calcr0), 0.0, f)


SOURCE_FUNCS = {"gauss": source_gauss, "sersic": source_sersic, "tophat": source_tophat,
                "moffat": source_moffat, "jaffe": source_jaffe}


def source_norm(model, r0, n, pix):
    if model == "gauss":
        return 2.0 * np.pi * r0 * r0 / (pix * pix)
    if model == "sersic":
        bn = bnn_sers(n)
        return (np.pi * r0 * r0 / (pix * pix)) * bn * bn * gam2n1_sers(n)
    if model == "tophat":
        return np.pi * r0 * r0 / (pix * pix)
    if model == "moffat":
        return np.pi * r0 * r0 / ((n - 1.0) * pix * pix)
    if model == "jaffe":
        return 2.0 * np.pi * r0 * n / (pix * pix)
    return 1.0


def moffat_fwhmtoa(fwhm, b):
    return fwhm * 0.5 / jnp.sqrt(jnp.power(2.0, 1.0 / b) - 1.0)


def source_moffat_psf(x, y, x0, y0, e, pa, fwhm, b):
    a = moffat_fwhmtoa(fwhm, b)
    u = ucalc(x - x0, y - y0, e, pa)
    return (b - 1.0) / (np.pi * a * a) * jnp.power(1.0 + u / (a * a), -b)


def _subgrid(n, pix):
    h = pix / n
    o = -0.5 * pix + (np.arange(n) + 0.5) * h
    dx, dy = np.meshgrid(o, o, indexing="ij")
    return dx.ravel(), dy.ravel()


def sourcemodel(model, p, sx, sy, pxx, pxy, pyx, pyy, pix, s, mode="glafic", n_sub=3):
    """Surface brightness at source positions with sub-pixel integration
    (source_all in source.c).  ``p`` is the length-8 extend parameter vector."""
    f0 = SOURCE_FUNCS[model]
    norm_amp = p[1]
    x0, y0, e, pa, r0, n = p[2], p[3], p[4], p[5], p[6], p[7]

    def evalf(dx, dy):
        dsx = (1.0 - pxx[..., None]) * dx - pxy[..., None] * dy
        dsy = (1.0 - pyy[..., None]) * dy - pyx[..., None] * dx
        v = f0(sx[..., None] + dsx, sy[..., None] + dsy, x0, y0, e, pa, r0, n, s)
        return jnp.mean(v, axis=-1)

    if mode == "uniform":
        dx, dy = _subgrid(n_sub, pix)
        f = evalf(dx, dy)
    else:
        plain = f0(sx, sy, x0, y0, e, pa, r0, n, s)
        if s.flag_extref != 0 and model != "tophat" and pix > 0.0:
            muinv = jnp.abs((1.0 - pxx) * (1.0 - pyy) - pxy * pyx + s.imag_ceil)
            dr2s = (sx - x0) ** 2 + (sy - y0) ** 2
            dr2 = dr2s / muinv
            refine = (dr2s < (s.source_refr0 * r0) ** 2) | (dr2 < 4.0 * s.pix_ext ** 2)
            fine = dr2 < s.pix_ext ** 2
            dx1, dy1 = _subgrid(s.num_pixint, pix)
            dx2, dy2 = _subgrid(4 * s.num_pixint, pix)
            f1 = evalf(dx1, dy1)
            f2 = evalf(dx2, dy2)
            f = jnp.where(refine, jnp.where(fine, f2, f1), plain)
        else:
            f = plain
    if s.flag_extnorm != 0:
        f = f / source_norm(model, r0, n, s.pix_ext)
    return norm_amp * f


# ---------------------------------------------------------------------------
# PSF


def psf_value(model, psf, x, y, x0, y0, pix):
    """source_psf in source.c (per unit flux, times pix^2)."""
    if model.flag_seeing == 1:
        x1 = source_moffat_psf(x, y, x0, y0, psf[1], psf[2], psf[0], psf[3])
        x2 = source_moffat_psf(x, y, x0, y0, psf[5], psf[6], psf[4], psf[7])
        return pix * pix * (psf[8] * x1 + (1.0 - psf[8]) * x2)
    # tabulated PSF (flag_seeing == -1)
    s = model.settings
    arr = jnp.asarray(model.psf_image)
    nk = model.fpsf_nk
    nnk = 2 * nk + 1
    pix2 = pix / s.seeing_sub
    xx = (x - x0) / pix2 + nk
    yy = (y - y0) / pix2 + nk
    i = jnp.floor(xx)
    j = jnp.floor(yy)
    p = xx - i
    q = yy - j
    ok = (i >= 0) & (i <= nnk - 2) & (j >= 0) & (j <= nnk - 2)
    ii = jnp.clip(i, 0, nnk - 2).astype(jnp.int32)
    jj = jnp.clip(j, 0, nnk - 2).astype(jnp.int32)
    v = ((1 - p) * (1 - q) * arr[jj, ii] + p * (1 - q) * arr[jj, ii + 1]
         + (1 - p) * q * arr[jj + 1, ii] + p * q * arr[jj + 1, ii + 1]) * (s.seeing_sub ** 2)
    return jnp.where(ok, v, 0.0)


def psf_value_pix(model, psf, x, y, x0, y0, pix):
    """source_psf_pix: PSF with sub-pixel integration near the centre."""
    s = model.settings
    if model.flag_seeing != 1:
        return psf_value(model, psf, x, y, x0, y0, pix)
    dr2 = (x - x0) ** 2 + (y - y0) ** 2
    r02 = moffat_fwhmtoa(psf[0], psf[3]) * moffat_fwhmtoa(psf[4], psf[7])
    plain = psf_value(model, psf, x, y, x0, y0, pix)
    if pix <= 0.0:
        return plain
    refine = (dr2 < s.source_refr0 ** 2 * r02) | (dr2 < 4.0 * s.pix_ext ** 2)
    fine = dr2 < s.pix_ext ** 2

    def sub(n):
        dx, dy = _subgrid(n, pix)
        return jnp.mean(psf_value(model, psf, x[..., None] + dx, y[..., None] + dy, x0, y0, pix), axis=-1)

    return jnp.where(refine, jnp.where(fine, sub(4 * s.num_pixint), sub(s.num_pixint)), plain)


def calc_psfnk(model):
    s = model.settings
    if model.flag_seeing == 1:
        return (int(s.psfconv_size / s.pix_ext) + 1) * s.seeing_sub
    return model.fpsf_nk


def psf_kernel(model, psf, pix_psf):
    nk = calc_psfnk(model)
    if model.flag_seeing == 1:
        o = (np.arange(2 * nk + 1) - nk) * pix_psf
        X, Y = np.meshgrid(o, o)           # [j, i]
        k = psf_value_pix(model, psf, jnp.asarray(X), jnp.asarray(Y), 0.0, 0.0, pix_psf)
    else:
        k = jnp.asarray(model.psf_image)
    return k / jnp.sum(k), nk


def _bicub1(x, a):
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0


def _bicub2(x, a):
    return a * (((x - 5.0) * x + 8.0) * x - 4.0)


def upsample(img, sub, a):
    """set_psfconv_img: bicubic (interior) / nearest (border) sub-sampling."""
    ny, nx = img.shape
    if sub == 1:
        return img
    # coordinates of sub-pixels in units of pixels (same as the C code)
    t = 1.0e-10 - 1.0 + (2.0 * np.arange(nx * sub) + sub + 1.0) / (2.0 * sub)
    u = 1.0e-10 - 1.0 + (2.0 * np.arange(ny * sub) + sub + 1.0) / (2.0 * sub)
    ix = np.trunc(t).astype(int)   # C (int) truncation
    sx = t - ix
    iy = np.trunc(u).astype(int)
    sy = u - iy
    wx = np.stack([_bicub2(sx + 1.0, a), _bicub1(sx, a), _bicub1(1.0 - sx, a), _bicub2(2.0 - sx, a)], 1)
    wy = np.stack([_bicub2(sy + 1.0, a), _bicub1(sy, a), _bicub1(1.0 - sy, a), _bicub2(2.0 - sy, a)], 1)
    okx = (ix >= 1) & (ix <= nx - 3)
    oky = (iy >= 1) & (iy <= ny - 3)
    # bicubic part
    cx = np.clip(ix[:, None] + np.arange(-1, 3)[None, :], 0, nx - 1)
    cy = np.clip(iy[:, None] + np.arange(-1, 3)[None, :], 0, ny - 1)
    g = img[cy[:, :, None, None], cx[None, None, :, :]]      # (NY,4,NX,4)
    bic = jnp.einsum("yaxb,ya,xb->yx", g, wy, wx)
    near = img[np.clip(iy, 0, ny - 1)[:, None], np.clip(ix, 0, nx - 1)[None, :]]
    inside = ((ix >= 0) & (ix <= nx - 1))[None, :] & ((iy >= 0) & (iy <= ny - 1))[:, None]
    near = jnp.where(inside, near, 0.0)
    use_b = oky[:, None] & okx[None, :]
    return jnp.where(use_b, bic, near)


def convolve(img_sub, kern, nk):
    """Zero-padded FFT convolution identical to the C implementation."""
    ny, nx = img_sub.shape
    py, px = ny + nk, nx + nk
    k = jnp.zeros((py, px))
    kc = kern  # (2nk+1, 2nk+1), centre at [nk, nk]
    # place kernel with wrap-around so that its centre sits at (0, 0)
    iy = (np.arange(2 * nk + 1) - nk) % py
    ix = (np.arange(2 * nk + 1) - nk) % px
    k = k.at[iy[:, None], ix[None, :]].add(kc)
    im = jnp.zeros((py, px)).at[:ny, :nx].set(img_sub)
    out = jnp.fft.ifft2(jnp.fft.fft2(im) * jnp.fft.fft2(k)).real
    return out[:ny, :nx]


def downsample(img_sub, sub):
    if sub == 1:
        return img_sub
    ny, nx = img_sub.shape
    return img_sub.reshape(ny // sub, sub, nx // sub, sub).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# full model image


def pixel_grid(s):
    nx, ny = s.nx_ext, s.ny_ext
    x = s.xmin + s.pix_ext * (np.arange(nx) + 0.5)
    y = s.ymin + s.pix_ext * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(x, y)  # [j, i]
    return X, Y


def deflection_table(model, P, zs, nlp, mask):
    """ext_set_table: deflections on the pixel grid and finite-difference
    magnification matrix (with glafic's edge handling)."""
    s = model.settings
    X, Y = pixel_grid(s)
    o = model.lens_system.ray_shoot(jnp.asarray(X), jnp.asarray(Y), P["lens"], zs, P["cosmo"], nlp=nlp)
    ax = o.ax
    ay = o.ay
    if mask is not None:
        ax = jnp.where(mask, 0.0, ax)
        ay = jnp.where(mask, 0.0, ay)
    pix = s.pix_ext

    def fd(a, axis):
        n = a.shape[axis]
        idx = np.arange(n)
        ip = np.minimum(idx + 1, n - 1)
        im = np.maximum(idx - 1, 0)
        d = pix * ((idx + 1 < n).astype(float) + (idx - 1 >= 0).astype(float))
        if axis == 1:
            return a[:, ip], a[:, im], d[None, :]
        return a[ip, :], a[im, :], d[:, None]

    axxp, axxm, ddx = fd(ax, 1)
    ayxp, ayxm, _ = fd(ay, 1)
    axyp, axym, ddy = fd(ax, 0)
    ayyp, ayym, _ = fd(ay, 0)
    pxx = (axxp - axxm) / ddx
    pxy = (ayxp - ayxm) / ddy
    pyx = (axyp - axym) / ddx
    pyy = (ayyp - ayym) / ddy
    return ax, ay, pxx, pxy, pyx, pyy


def model_images(model, P, compute_all=False, mode="glafic", n_sub=3, point_mode="glafic", flag_source=0):
    """Return model images, shape (num_ext, ny, nx) (without sky)."""
    s = model.settings
    mask = None
    if (not compute_all) and model.obs_mask is not None:
        mask = jnp.asarray(model.obs_mask)
    X, Y = pixel_grid(s)
    Xj = jnp.asarray(X)
    Yj = jnp.asarray(Y)
    pix_psf = s.pix_ext / s.seeing_sub if (s.seeing_sub > 1 and model.flag_seeing != 0) else s.pix_ext
    cosmo = P["cosmo"]
    lens = model.lens_system
    single = lens.num_planes == 1
    tables = {}
    if single:
        zs_all = [float(z) for z in model.ext_zs_init]
        fid = int(np.argmax(zs_all)) if len(zs_all) else 0
        zs_fid = P["ext"][fid, 0]
        zl0 = lens.zl_planes(P["lens"])[0]
        nlp_fid = lens.nlp_for(zs_all[fid]) if len(zs_all) else 1
        if nlp_fid > 0:
            tab_fid = deflection_table(model, P, zs_fid, nlp_fid, mask)
    imgs = []
    for ii, mname in enumerate(model.ext_models):
        p = P["ext"][ii]
        if mname in ("point",):
            imgs.append(jnp.zeros_like(Xj))
            continue
        if flag_source:
            sx, sy = Xj, Yj
            z = jnp.zeros_like(Xj)
            pxx = pxy = pyx = pyy = z
        else:
            if single:
                if nlp_fid == 0:
                    fac = 0.0
                    ax = ay = pxx = pxy = pyx = pyy = jnp.zeros_like(Xj)
                else:
                    fac = cm.disratio(zl0, zs_fid, p[0], cosmo)
                    ax, ay, pxx, pxy, pyx, pyy = tab_fid
            else:
                nlp = lens.nlp_for(model.ext_zs_init[ii])
                key = nlp
                if nlp == 0:
                    fac = 0.0
                    ax = ay = pxx = pxy = pyx = pyy = jnp.zeros_like(Xj)
                else:
                    ax, ay, pxx, pxy, pyx, pyy = deflection_table(model, P, p[0], nlp, mask)
                    fac = 1.0
            sx = Xj - ax * fac
            sy = Yj - ay * fac
            pxx, pxy, pyx, pyy = pxx * fac, pxy * fac, pyx * fac, pyy * fac
        img = sourcemodel(mname, p, sx, sy, pxx, pxy, pyx, pyy, pix_psf, s, mode=mode, n_sub=n_sub)
        if mask is not None:
            img = jnp.where(mask, 0.0, img)
        imgs.append(img)
    imgs = jnp.stack(imgs) if imgs else jnp.zeros((0,) + X.shape)

    if model.flag_seeing != 0 and len(model.ext_models) > 0:
        kern, nk = psf_kernel(model, P["psf"], pix_psf)
        out = []
        for ii, mname in enumerate(model.ext_models):
            im = imgs[ii]
            if mname != "point":
                up = upsample(im, s.seeing_sub, s.bicub_a)
                cv = convolve(up, kern, nk)
                im = downsample(cv, s.seeing_sub)
            else:
                im = point_source_image(model, P, ii, Xj, Yj, flag_source, point_mode)
            if mask is not None:
                im = jnp.where(mask, 0.0, im)
            out.append(im)
        imgs = jnp.stack(out)
    return imgs


def point_source_image(model, P, ii, Xj, Yj, flag_source, point_mode):
    """Images of an ``extend point`` component (PSF at each lensed image)."""
    from .point import _newton_refine
    s = model.settings
    p = P["ext"][ii]
    zs, flux, xs, ys = p[0], p[1], p[2], p[3]
    if flag_source:
        return flux * psf_value_pix(model, P["psf"], Xj, Yj, xs, ys, s.pix_ext)
    lens = model.lens_system
    nlp = lens.nlp_for(model.ext_zs_init[ii])
    nmax = model.max_point_images
    sg = jax.lax.stop_gradient
    if point_mode == "glafic":
        def host(lp, cp, z, x, y):
            imgs = model.find_images_numeric(np.asarray(lp), np.asarray(cp), float(z), float(x), float(y))
            out = np.zeros((nmax, 2))
            ok = np.zeros(nmax, dtype=bool)
            n = min(imgs.shape[0], nmax)
            out[:n] = imgs[:n, :2]
            ok[:n] = True
            return out, ok
        shapes = (jax.ShapeDtypeStruct((nmax, 2), jnp.float64), jax.ShapeDtypeStruct((nmax,), jnp.bool_))
        pos, ok = jax.pure_callback(host, shapes, sg(P["lens"]), sg(P["cosmo"]), sg(zs), sg(xs), sg(ys),
                                    vmap_method="sequential")
    else:
        init = model.point_ext_guess[ii]
        pos = jnp.zeros((nmax, 2)).at[:init.shape[0]].set(init)
        ok = jnp.arange(nmax) < init.shape[0]

        def step(c, _):
            x, y = c
            return _newton_refine(lens, x, y, sg(xs), sg(ys), sg(P["lens"]), sg(zs), sg(P["cosmo"]), nlp, None), None

        (px, py), _ = jax.lax.scan(step, (pos[:, 0], pos[:, 1]), None, length=20)
        pos = jnp.stack([px, py], 1)
    xi, yi = _newton_refine(lens, pos[:, 0], pos[:, 1], xs, ys, P["lens"], zs, P["cosmo"], nlp, None)
    o = lens.ray_shoot(xi, yi, P["lens"], zs, P["cosmo"], nlp=nlp)
    mag = jnp.abs(1.0 / (o.muinv + s.imag_ceil))
    img = jnp.zeros_like(Xj)
    hw = point_window(model)
    ny, nx = Xj.shape
    for l in range(nmax):
        # plain PSF everywhere, sub-pixel refined PSF only in a window around the image
        pv = psf_value(model, P["psf"], Xj, Yj, xi[l], yi[l], s.pix_ext)
        if model.flag_seeing == 1 and hw > 0:
            K = 2 * hw + 1
            ic = jnp.clip(jnp.round((xi[l] - s.xmin) / s.pix_ext - 0.5).astype(jnp.int32) - hw, 0, max(nx - K, 0))
            jc = jnp.clip(jnp.round((yi[l] - s.ymin) / s.pix_ext - 0.5).astype(jnp.int32) - hw, 0, max(ny - K, 0))
            K = min(K, nx, ny)
            wx = jax.lax.dynamic_slice(Xj, (jc, ic), (K, K))
            wy = jax.lax.dynamic_slice(Yj, (jc, ic), (K, K))
            ref = psf_value_pix(model, P["psf"], wx, wy, xi[l], yi[l], s.pix_ext)
            pln = psf_value(model, P["psf"], wx, wy, xi[l], yi[l], s.pix_ext)
            pv = jax.lax.dynamic_update_slice(pv, jax.lax.dynamic_slice(pv, (jc, ic), (K, K)) + ref - pln, (jc, ic))
        img = img + jnp.where(ok[l], flux * mag[l] * pv, 0.0)
    return img


def point_window(model):
    """Static half-width (pixels) of the window where glafic refines PSF images.

    Refinement happens for r^2 < source_refr0^2 r02 or r < 2 pix_ext; the
    window is sized from the initial PSF with a factor-2 margin (PSF widths
    are not expected to double during sampling)."""
    s = model.settings
    if model.flag_seeing != 1:
        return 0
    p = np.asarray(model.psf_params, float)
    a = lambda fw, b: fw * 0.5 / np.sqrt(2.0 ** (1.0 / b) - 1.0)
    r = s.source_refr0 * np.sqrt(a(p[0], p[3]) * a(p[4], p[7]))
    r = max(2.0 * r, 2.0 * s.pix_ext)
    return int(np.ceil(r / s.pix_ext)) + 2


# ---------------------------------------------------------------------------
# observations / noise


def calc_obsnoise(obs, mask, s):
    """Sky level and per-pixel noise from the data (calc_obsnoise in fits.c)."""
    a = np.asarray(obs, dtype=np.float32).ravel().astype(np.float64)
    m = np.zeros_like(a, dtype=bool) if mask is None else np.asarray(mask).ravel() > 0
    v = a[~m].astype(np.float32)
    n = v.size
    med = float(np.sum(v.astype(np.float64)) / n)
    sig = float(np.sqrt(np.sum(v.astype(np.float64) ** 2) / n - med * med))
    k = n
    while True:
        ko = k
        sel = (v > med - s.noise_clip * sig) & (v < med + s.noise_clip * sig)
        k = int(sel.sum())
        vs = v[sel].astype(np.float64)
        med = float(vs.sum() / k)
        sig = float(np.sqrt((vs ** 2).sum() / k - med * med))
        if not (k < ko):
            break
    x = np.abs(a - med) / (sig * sig * (s.obs_gain * float(s.obs_ncomb)))
    noise = np.where(m, 0.0, sig * np.sqrt(1.0 + x))
    return med, sig, noise.reshape(np.shape(obs)).astype(np.float32).astype(np.float64)


def chi2_extend_data(model, P, mode="glafic", n_sub=3, point_mode="glafic"):
    imgs = model_images(model, P, compute_all=False, mode=mode, n_sub=n_sub, point_mode=point_mode)
    s = model.settings
    sky = s.skyfix_value if s.skyfix == 1 else P["sky"]
    f = jnp.sum(imgs, axis=0) + sky
    obs = jnp.asarray(model.obs_image)
    sig = jnp.asarray(model.obs_noise)
    good = np.ones(model.obs_image.shape, dtype=bool) if model.obs_mask is None else ~np.asarray(model.obs_mask)
    sig_safe = np.where(good, np.asarray(model.obs_noise), 1.0)
    r = (f - obs) / sig_safe
    return jnp.sum(jnp.where(good, r * r, 0.0))
