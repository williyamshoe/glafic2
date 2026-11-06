"""The glafic model: lenses + sources + data + priors, with a differentiable
log-posterior suitable for HMC / NUTS.

Parameters are held in a pytree ``P``::

    P = {"lens":  (num_len, 8),   # para_lens
         "cosmo": (5,),           # omega, lambda, weos (w0), hubble, wa
         "ext":   (num_ext, 8),   # para_ext
         "point": (num_poi, 3),   # para_poi (zs, xs, ys)
         "psf":   (9,),           # para_psf
         "sky":   ()}             # sky level for extended chi^2

The vector of *free* parameters ``theta`` follows glafic's ordering
(``partopara`` in opt_lens.c):

    {lens flags} -> {ovary, lvary, wvary, hvary, wavary} -> {extend flags} -> sky
    -> {point zs flags} -> {point x, y flags} -> {psf flags}

Point-source positions flagged with setopt are *sampled* here, whereas glafic
optimises them internally for every chi^2 evaluation.
"""

import copy
import functools

import numpy as np
import jax
import jax.numpy as jnp

from .constants import (NPAR_LEN, NPAR_EXT, NPAR_POI, NPAR_PSF, LENS_MODELS, EXTEND_MODELS,
                        INIT_MMIN, INIT_MMAX, INIT_XYMIN, INIT_XYMAX, INIT_EMIN, INIT_EMAX,
                        INIT_ANGMIN, INIT_ANGMAX, INIT_R0MIN, INIT_R0MAX, INIT_NMIN, INIT_NMAX,
                        INIT_ZMIN, INIT_ZMAX, INIT_BETAMIN, INIT_BETAMAX, INIT_FMIN, INIT_FMAX,
                        INIT_OMMIN, INIT_OMMAX, INIT_WMIN, INIT_WMAX, INIT_HMIN, INIT_HMAX, TOL_ZS)
from .settings import Settings
from .lensing import LensSystem
from .lens_models import param_valid
from .imagefinder import ImageFinder
from . import point as point_mod
from . import extend as ext_mod

_LENS_MIN = [INIT_ZMIN, INIT_MMIN, INIT_XYMIN, INIT_XYMIN, INIT_EMIN, INIT_ANGMIN, INIT_R0MIN, INIT_NMIN]
_LENS_MAX = [INIT_ZMAX, INIT_MMAX, INIT_XYMAX, INIT_XYMAX, INIT_EMAX, INIT_ANGMAX, INIT_R0MAX, INIT_NMAX]
_PSF_MIN = [INIT_R0MIN, INIT_EMIN, INIT_ANGMIN, INIT_BETAMIN, INIT_R0MIN, INIT_EMIN, INIT_ANGMIN, INIT_BETAMIN, INIT_FMIN]
_PSF_MAX = [INIT_R0MAX, INIT_EMAX, INIT_ANGMAX, INIT_BETAMAX, INIT_R0MAX, INIT_EMAX, INIT_ANGMAX, INIT_BETAMAX, INIT_FMAX]
_COSMO_MIN = [INIT_OMMIN, INIT_OMMIN, INIT_WMIN, INIT_HMIN, -10.0]
_COSMO_MAX = [INIT_OMMAX, INIT_OMMAX, INIT_WMAX, INIT_HMAX, 10.0]


def _prior_block(n, npar, mins, maxs):
    idx_i = np.repeat(np.arange(n)[:, None], npar, axis=1)
    idx_j = np.repeat(np.arange(npar)[None, :], n, axis=0)
    return dict(min=np.tile(np.array(mins, float), (n, 1)), max=np.tile(np.array(maxs, float), (n, 1)),
                med=np.zeros((n, npar)), sig=np.zeros((n, npar)),
                rai=idx_i.copy(), raj=idx_j.copy(), rat=np.ones((n, npar)), ras=np.zeros((n, npar)),
                rerai=idx_i.copy(), reraj=idx_j.copy(), reral=np.full((n, npar), -1e30),
                rerah=np.full((n, npar), 1e30),
                lrai=np.full((n, npar), -1), lraj=np.full((n, npar), -1),
                lrat=np.ones((n, npar)), lras=np.zeros((n, npar)))


class Model:
    """A complete glafic model.  Use :func:`glafic_jax.io.read_input` or the
    glafic-like setters (``set_lens``, ``set_extend``, ...) to build one."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.lens_models, self.ext_models = [], []
        self.lens_params = np.zeros((0, NPAR_LEN))
        self.ext_params = np.zeros((0, NPAR_EXT))
        self.poi_params = np.zeros((0, NPAR_POI))
        self.psf_params = np.array([1.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 3.0, 1.0])
        self.flag_seeing = 0
        self.psf_image = None
        self.fpsf_nk = 0
        self.gals = None
        self.sky = 0.0
        self.obs_point = []
        self.obs_image = None
        self.obs_mask = None
        self.obs_noise = None
        self.obs_noise_file = None
        self.max_point_images = 6
        self.point_ext_guess = {}
        self._finder_cache = {}
        self._point_batch = None
        #: optional callable P -> additional log-prior (e.g. lambda P: jnp.where(P['cosmo'][2] + P['cosmo'][4] < 0, 0., -jnp.inf))
        self.extra_log_prior = None
        #: 'sample' (source positions are parameters), 'profile' or 'marginalize'
        #: (analytic; source plane, batched chi^2) -- see point.chi2_point_batched
        self.source_mode = "sample"

    # ------------------------------------------------------------------
    # building (mirrors the C python interface)
    def startup_setnum(self, num_len, num_ext, num_poi):
        self.lens_models = ["sie"] * num_len
        self.ext_models = ["gauss"] * num_ext
        self.lens_params = np.zeros((num_len, NPAR_LEN))
        self.ext_params = np.zeros((num_ext, NPAR_EXT))
        self.poi_params = np.zeros((num_poi, NPAR_POI))
        self.flag_lens = np.zeros((num_len, NPAR_LEN), dtype=int)
        self.flag_ext = np.zeros((num_ext, NPAR_EXT), dtype=int)
        self.flag_poi = np.zeros((num_poi, NPAR_POI), dtype=int)
        self.flag_psf = np.zeros(NPAR_PSF, dtype=int)
        self.prior_lens = _prior_block(num_len, NPAR_LEN, _LENS_MIN, _LENS_MAX)
        self.prior_ext = _prior_block(num_ext, NPAR_EXT, _LENS_MIN, _LENS_MAX)
        self.prior_point = _prior_block(num_poi, 5, [INIT_ZMIN, INIT_XYMIN, INIT_XYMIN, 0, 0],
                                        [INIT_ZMAX, INIT_XYMAX, INIT_XYMAX, 0, 0])
        self.prior_psf = {k: v[0] for k, v in _prior_block(1, NPAR_PSF, _PSF_MIN, _PSF_MAX).items()}
        self.prior_cosmo = dict(min=np.array(_COSMO_MIN, float), max=np.array(_COSMO_MAX, float),
                                med=np.array(self.settings.cosmo, float), sig=np.zeros(5))
        self.obs_point = [None] * num_poi
        return self

    def set_lens(self, i, model, *p):
        if model not in LENS_MODELS:
            raise ValueError(f"invalid lens model name '{model}'")
        self.lens_models[i - 1] = model
        self.lens_params[i - 1] = np.asarray(p, float)

    def set_extend(self, i, model, *p):
        if model not in EXTEND_MODELS:
            raise ValueError(f"invalid extend model name '{model}'")
        if model == "srcs":
            raise NotImplementedError("extend model 'srcs' is not supported by glafic_jax")
        self.ext_models[i - 1] = model
        self.ext_params[i - 1] = np.asarray(p, float)

    def set_point(self, i, zs, x, y):
        self.poi_params[i - 1] = [zs, x, y]

    def set_psf(self, *p):
        self.psf_params = np.asarray(p, float)
        self.flag_seeing = 1

    def setopt_lens(self, i, *f):
        self.flag_lens[i - 1] = np.asarray(f, int)

    def setopt_extend(self, i, *f):
        self.flag_ext[i - 1] = np.asarray(f, int)

    def setopt_point(self, i, *f):
        self.flag_poi[i - 1] = np.asarray(f, int)

    def setopt_psf(self, *f):
        self.flag_psf = np.asarray(f, int)

    def set_secondary(self, line):
        self.settings = self.settings.set_secondary(line)

    def set_cosmo(self, omega, lambda_, weos, hubble, wa=0.0):
        s = self.settings
        for k, v in (("omega", omega), ("lambda", lambda_), ("weos", weos), ("hubble", hubble), ("wa", wa)):
            s = s.set(k, v)
        self.settings = s

    def read_gals(self, fname):
        rows = []
        for line in open(fname):
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            v = [float(a) for a in t[:5]]
            if len(v) == 3:
                v += [0.0, 0.0]
            rows.append(v)
        self.gals = np.array(rows)

    # ------------------------------------------------------------------
    # data
    def readobs_point(self, fname):
        lines = [l for l in open(fname) if l.strip() and not l.lstrip().startswith("#")]
        k = 0
        while k < len(lines):
            t = lines[k].split()
            i, n = int(t[0]), int(t[1])
            zs = float(t[2]) if len(t) > 2 else -1.0
            zserr = float(t[3]) if len(t) > 3 else 0.0
            if zs >= 0.0:
                if abs(zs - self.poi_params[i - 1, 0]) > TOL_ZS and zserr == 0.0:
                    raise ValueError("wrong zs (readobs_point)")
            else:
                zserr = 0.0
            self.prior_point["med"][i - 1, 0] = zs
            self.prior_point["sig"][i - 1, 0] = zserr
            tab, par = [], []
            for j in range(n):
                v = lines[k + 1 + j].split("#")[0].split()
                tab.append([float(a) for a in v[:7]])
                par.append(int(v[7]))
            self.obs_point[i - 1] = dict(tab=np.array(tab).reshape(n, 7), parity=np.array(par, dtype=int))
            k += 1 + n

    def reset_obs_point(self, i, j, k, p):
        if k == 8:
            self.obs_point[i - 1]["parity"][j - 1] = int(p)
        else:
            self.obs_point[i - 1]["tab"][j - 1, k - 1] = p

    def readobs_extend(self, fname, fname_mask=None):
        from astropy.io import fits
        obs = np.asarray(fits.getdata(fname), dtype=np.float32).astype(np.float64)
        s = self.settings
        if obs.shape != (s.ny_ext, s.nx_ext):
            raise ValueError(f"invalid obs fits size {obs.shape} != {(s.ny_ext, s.nx_ext)}")
        mask = None
        if fname_mask is not None and fname_mask != "N":
            mask = np.asarray(fits.getdata(fname_mask)) > 0
        self.set_obs_extend(obs, mask)

    def set_obs_extend(self, obs, mask=None):
        """Set the observed image (ny, nx) and optional bool mask (True = masked)."""
        s = self.settings
        self.obs_image = np.asarray(obs, dtype=float)
        self.obs_mask = None if mask is None else np.asarray(mask, dtype=bool)
        med, sig, noise = ext_mod.calc_obsnoise(self.obs_image, self.obs_mask, s)
        if s.skyfix_value > 0.1 * 1.0e10:
            self.settings = s.set("skyfix_value", med)
        self.sky = self.settings.skyfix_value
        self.skysigma = sig
        self.obs_noise = noise if self.obs_noise_file is None else self.obs_noise_file

    def readnoise_extend(self, fname):
        from astropy.io import fits
        self.obs_noise_file = np.asarray(fits.getdata(fname), dtype=np.float32).astype(np.float64)
        self.obs_noise = self.obs_noise_file

    def readpsf(self, fname):
        from astropy.io import fits
        a = np.asarray(fits.getdata(fname), dtype=np.float32).astype(np.float64)
        if a.shape[0] != a.shape[1] or a.shape[0] % 2 == 0:
            raise ValueError("PSF fits must be square with an odd size")
        self.psf_image = a / a.sum()
        self.fpsf_nk = (a.shape[0] - 1) // 2
        self.flag_seeing = -1

    def parprior(self, fname):
        from .io import read_prior
        read_prior(self, fname)

    # ------------------------------------------------------------------
    # derived static structure
    def model_init(self):
        """Freeze the static structure (call after changing models/redshifts)."""
        s = self.settings
        self.lens_system = LensSystem(self.lens_models, self.lens_params[:, 0], s, self.gals)
        self.nlp_point = [self.lens_system.nlp_for(z) for z in self.poi_params[:, 0]]
        self.ext_zs_init = [float(z) for z in self.ext_params[:, 0]]
        self.prior_cosmo["med"] = np.where(self.prior_cosmo["sig"] != 0.0, self.prior_cosmo["med"],
                                           np.array(s.cosmo, float))
        self._finder_cache = {}
        self._point_batch = point_mod.batch_data(self)
        self._build_theta_map()
        return self

    @property
    def num_len(self):
        return len(self.lens_models)

    @property
    def num_ext(self):
        return len(self.ext_models)

    @property
    def num_poi(self):
        return self.poi_params.shape[0]

    # ------------------------------------------------------------------
    # parameters
    def params(self):
        """Current parameter pytree (numpy -> jnp)."""
        s = self.settings
        return {"lens": jnp.asarray(self.lens_params), "cosmo": jnp.asarray(s.cosmo, dtype=float),
                "ext": jnp.asarray(self.ext_params), "point": jnp.asarray(self.poi_params),
                "psf": jnp.asarray(self.psf_params), "sky": jnp.asarray(float(self.sky))}

    def _build_theta_map(self):
        s = self.settings
        names = []
        self._tm_lens = np.argwhere(self.flag_lens == 1)
        for i, j in self._tm_lens:
            names.append(f"lens{i + 1}_{self.lens_models[i]}_p{j + 1}")
        self._tm_cosmo = [k for k, f in enumerate((s.ovary, s.lvary, s.wvary, s.hvary, s.wavary)) if f != 0]
        for k in self._tm_cosmo:
            names.append(["omega", "lambda", "weos", "hubble", "wa"][k])
        self._tm_ext = np.argwhere(self.flag_ext == 1)
        for i, j in self._tm_ext:
            names.append(f"ext{i + 1}_{self.ext_models[i]}_p{j + 1}")
        self._tm_sky = (s.skyfix == 0) and len(self._tm_ext) > 0
        if self._tm_sky:
            names.append("sky")
        self._tm_poi_z = np.nonzero(self.flag_poi[:, 0] == 1)[0] if self.num_poi else np.zeros(0, int)
        for i in self._tm_poi_z:
            names.append(f"point{i + 1}_zs")
        self._tm_poi_xy = np.argwhere(self.flag_poi[:, 1:] == 1) if self.num_poi else np.zeros((0, 2), int)
        if self.source_mode != "sample":
            self._tm_poi_xy = np.zeros((0, 2), int)   # source positions are profiled / marginalised
        for i, j in self._tm_poi_xy:
            names.append(f"point{i + 1}_{'xy'[j]}")
        self._tm_psf = np.nonzero(self.flag_psf == 1)[0] if (len(self._tm_ext) > 0 and self.flag_seeing == 1) \
            else np.zeros(0, int)
        for j in self._tm_psf:
            names.append(f"psf_p{j + 1}")
        self.theta_names = names

    @property
    def ndim(self):
        return len(self.theta_names)

    def theta0(self):
        return self.flatten(self.params())

    def flatten(self, P):
        parts = []
        if len(self._tm_lens):
            parts.append(P["lens"][self._tm_lens[:, 0], self._tm_lens[:, 1]])
        if self._tm_cosmo:
            parts.append(P["cosmo"][np.array(self._tm_cosmo)])
        if len(self._tm_ext):
            parts.append(P["ext"][self._tm_ext[:, 0], self._tm_ext[:, 1]])
        if self._tm_sky:
            parts.append(jnp.atleast_1d(P["sky"]))
        if len(self._tm_poi_z):
            parts.append(P["point"][self._tm_poi_z, 0])
        if len(self._tm_poi_xy):
            parts.append(P["point"][self._tm_poi_xy[:, 0], self._tm_poi_xy[:, 1] + 1])
        if len(self._tm_psf):
            parts.append(P["psf"][self._tm_psf])
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def unflatten(self, theta, P0=None):
        """theta -> full parameter pytree (partopara + parmatch)."""
        P = dict(self.params() if P0 is None else P0)
        s = self.settings
        k = 0

        def take(n):
            nonlocal k
            v = theta[k:k + n]
            k += n
            return v

        if len(self._tm_lens):
            P["lens"] = P["lens"].at[self._tm_lens[:, 0], self._tm_lens[:, 1]].set(take(len(self._tm_lens)))
        if self._tm_cosmo:
            c = P["cosmo"]
            for idx in self._tm_cosmo:
                c = c.at[idx].set(take(1)[0])
                if s.flatfix != 0 and idx == 0:
                    c = c.at[1].set(1.0 - c[0])
                if s.flatfix != 0 and idx == 1:
                    c = c.at[0].set(1.0 - c[1])
            P["cosmo"] = c
        if len(self._tm_ext):
            P["ext"] = P["ext"].at[self._tm_ext[:, 0], self._tm_ext[:, 1]].set(take(len(self._tm_ext)))
        if self._tm_sky:
            P["sky"] = take(1)[0]
        if len(self._tm_poi_z):
            P["point"] = P["point"].at[self._tm_poi_z, 0].set(take(len(self._tm_poi_z)))
        if len(self._tm_poi_xy):
            P["point"] = P["point"].at[self._tm_poi_xy[:, 0], self._tm_poi_xy[:, 1] + 1].set(take(len(self._tm_poi_xy)))
        if len(self._tm_psf):
            P["psf"] = P["psf"].at[self._tm_psf].set(take(len(self._tm_psf)))
        return self.parmatch(P)

    def parmatch(self, P):
        """Apply 'match' priors with zero width (parmatch_lens/ext/poi)."""
        pl, pe, pp = self.prior_lens, self.prior_ext, self.prior_point
        L = P["lens"]
        for i in range(self.num_len):
            for j in range(NPAR_LEN):
                if (pl["rai"][i, j] != i or pl["raj"][i, j] != j) and pl["ras"][i, j] == 0.0:
                    L = L.at[i, j].set(pl["rat"][i, j] * L[pl["rai"][i, j], pl["raj"][i, j]])
        P["lens"] = L
        Q = P["point"]
        for i in range(self.num_poi):
            if pp["rai"][i, 0] != i and pp["ras"][i, 0] == 0.0:
                Q = Q.at[i, 0].set(pp["rat"][i, 0] * Q[pp["rai"][i, 0], 0])
            for j in range(NPAR_POI):
                if pp["lrai"][i, j] >= 0 and pp["lraj"][i, j] >= 0 and pp["lras"][i, j] == 0.0:
                    Q = Q.at[i, j].set(pp["lrat"][i, j] * L[pp["lrai"][i, j], pp["lraj"][i, j]])
        P["point"] = Q
        E = P["ext"]
        for i in range(self.num_ext):
            for j in range(NPAR_EXT):
                if (pe["rai"][i, j] != i or pe["raj"][i, j] != j) and pe["ras"][i, j] == 0.0:
                    E = E.at[i, j].set(pe["rat"][i, j] * E[pe["rai"][i, j], pe["raj"][i, j]])
                if pe["lrai"][i, j] >= 0 and pe["lraj"][i, j] >= 0 and pe["lras"][i, j] == 0.0:
                    E = E.at[i, j].set(pe["lrat"][i, j] * L[pe["lrai"][i, j], pe["lraj"][i, j]])
        P["ext"] = E
        ps = self.prior_psf
        F = P["psf"]
        for j in range(NPAR_PSF):
            if ps["raj"][j] != j and ps["ras"][j] == 0.0:
                F = F.at[j].set(ps["rat"][j] * F[ps["raj"][j]])
        P["psf"] = F
        return P

    # ------------------------------------------------------------------
    # priors and range checks
    @staticmethod
    def _block_range_bad(X, pr, rows, cols):
        bad = jnp.asarray(False)
        for i in rows:
            for j in cols:
                v = X[i, j]
                bad = bad | (v < pr["min"][i, j]) | (v > pr["max"][i, j])
                if pr["rerai"][i, j] != i or pr["reraj"][i, j] != j:
                    ref = X[pr["rerai"][i, j], pr["reraj"][i, j]]
                    bad = bad | (v < pr["reral"][i, j] * ref) | (v > pr["rerah"][i, j] * ref)
        return bad

    @staticmethod
    def _gauss_terms(v, med, sig):
        """gaussian (sig>0) or log10-gaussian (sig<0) prior term, and a 'bad' flag."""
        if sig > 0.0:
            return (v - med) ** 2 / (sig * sig), jnp.asarray(False)
        if sig < 0.0:
            bad = (v < 0.0) | (med < 0.0)
            lp = jnp.log10(jnp.where(v > 0, v, 1.0))
            return (lp - np.log10(max(med, 1e-300))) ** 2 / (sig * sig), bad
        return 0.0, jnp.asarray(False)

    def range_bad_lens(self, P):
        """check_para_lens_all (lens, ext zs, point zs, cosmo) + model validity."""
        bad = self._block_range_bad(P["lens"], self.prior_lens, range(self.num_len), range(NPAR_LEN))
        bad = bad | self._block_range_bad(P["ext"], self.prior_ext, range(self.num_ext), [0])
        bad = bad | self._block_range_bad(P["point"], self.prior_point, range(self.num_poi), [0])
        c = P["cosmo"]
        bad = bad | jnp.any((c < self.prior_cosmo["min"]) | (c > self.prior_cosmo["max"]))
        for i, m in enumerate(self.lens_models):
            bad = bad | ~param_valid(m, P["lens"][i])
        return bad

    def range_bad_ext(self, P):
        bad = self._block_range_bad(P["ext"], self.prior_ext, range(self.num_ext), range(1, NPAR_EXT))
        if self.flag_seeing == 1:
            ps = self.prior_psf
            F = P["psf"]
            for j in range(NPAR_PSF):
                bad = bad | (F[j] < ps["min"][j]) | (F[j] > ps["max"][j])
                if ps["reraj"][j] != j:
                    ref = F[ps["reraj"][j]]
                    bad = bad | (F[j] < ps["reral"][j] * ref) | (F[j] > ps["rerah"][j] * ref)
        return bad

    def chi2prior_lens(self, P):
        c2 = 0.0
        bad = jnp.asarray(False)
        pl = self.prior_lens
        L = P["lens"]
        for i in range(self.num_len):
            for j in range(NPAR_LEN):
                t, b = self._gauss_terms(L[i, j], pl["med"][i, j], pl["sig"][i, j])
                c2, bad = c2 + t, bad | b
                if pl["rai"][i, j] != i or pl["raj"][i, j] != j:
                    ref = pl["rat"][i, j] * L[pl["rai"][i, j], pl["raj"][i, j]]
                    ras = pl["ras"][i, j]
                    if ras > 0.0:
                        c2 = c2 + (L[i, j] - ref) ** 2 / ras ** 2
                    elif ras < 0.0:
                        bad = bad | (L[i, j] < 0) | (ref < 0)
                        c2 = c2 + (jnp.log10(jnp.abs(L[i, j])) - jnp.log10(jnp.abs(ref))) ** 2 / ras ** 2
        pc = self.prior_cosmo
        for k in range(5):
            t, b = self._gauss_terms(P["cosmo"][k], pc["med"][k], pc["sig"][k])
            c2, bad = c2 + t, bad | b
        for blk, X in ((self.prior_ext, P["ext"]), (self.prior_point, P["point"])):
            for i in range(X.shape[0]):
                t, b = self._gauss_terms(X[i, 0], blk["med"][i, 0], blk["sig"][i, 0])
                c2, bad = c2 + t, bad | b
                if blk["rai"][i, 0] != i:
                    ref = blk["rat"][i, 0] * X[blk["rai"][i, 0], 0]
                    ras = blk["ras"][i, 0]
                    if ras > 0.0:
                        c2 = c2 + (X[i, 0] - ref) ** 2 / ras ** 2
                    elif ras < 0.0:
                        c2 = c2 + (jnp.log10(jnp.abs(X[i, 0])) - jnp.log10(jnp.abs(ref))) ** 2 / ras ** 2
        return c2, bad

    def chi2prior_ext(self, P):
        c2 = 0.0
        bad = jnp.asarray(False)
        pe = self.prior_ext
        E = P["ext"]
        L = P["lens"]
        for i in range(self.num_ext):
            for j in range(1, NPAR_EXT):
                t, b = self._gauss_terms(E[i, j], pe["med"][i, j], pe["sig"][i, j])
                c2, bad = c2 + t, bad | b
                for (ri, rj, rt, rs, X) in ((pe["rai"][i, j], pe["raj"][i, j], pe["rat"][i, j], pe["ras"][i, j], E),
                                            (pe["lrai"][i, j], pe["lraj"][i, j], pe["lrat"][i, j], pe["lras"][i, j], L)):
                    is_ext = X is E
                    active = (ri != i or rj != j) if is_ext else (ri >= 0 or rj >= 0)
                    if not active:
                        continue
                    ref = rt * X[ri, rj]
                    if rs > 0.0:
                        c2 = c2 + (E[i, j] - ref) ** 2 / rs ** 2
                    elif rs < 0.0:
                        bad = bad | (E[i, j] < 0) | (ref < 0)
                        c2 = c2 + (jnp.log10(jnp.abs(E[i, j])) - jnp.log10(jnp.abs(ref))) ** 2 / rs ** 2
        if self.flag_seeing == 1:
            ps = self.prior_psf
            F = P["psf"]
            for j in range(NPAR_PSF):
                t, b = self._gauss_terms(F[j], ps["med"][j], ps["sig"][j])
                c2, bad = c2 + t, bad | b
                if ps["raj"][j] != j:
                    ref = ps["rat"][j] * F[ps["raj"][j]]
                    if ps["ras"][j] > 0.0:
                        c2 = c2 + (F[j] - ref) ** 2 / ps["ras"][j] ** 2
        return c2, bad

    # ------------------------------------------------------------------
    # lens model utilities
    def lensmodel(self, x, y, zs, P=None, lensid=0):
        """Equivalent of glafic.calcimage: returns a LensOut namedtuple."""
        P = self.params() if P is None else P
        if lensid:
            return self.lens_system.single_lens(lensid - 1, x, y, P["lens"], zs, P["cosmo"])
        return self.lens_system.ray_shoot(x, y, P["lens"], zs, P["cosmo"])

    def calcimage(self, zs, x, y, P=None):
        """Same output ordering as glafic.calcimage:
        (ax, ay, td, kappa, gamma1, gamma2, muinv, rotation)."""
        o = self.lensmodel(jnp.asarray(x, float), jnp.asarray(y, float), zs, P)
        return tuple(float(v) for v in (o.ax, o.ay, o.td, o.kap, o.gam1, o.gam2, o.muinv, o.rot))

    def _jit_lens(self, nlp, planes=False):
        key = (nlp, planes)
        if key not in self._finder_cache:
            ls = self.lens_system

            if planes:
                @jax.jit
                def f(x, y, lp, cp, zs):
                    _, pl = ls.ray_shoot(x, y, lp, zs, cp, nlp=nlp, return_planes=True)
                    return pl
            else:
                @jax.jit
                def f(x, y, lp, cp, zs):
                    o = ls.ray_shoot(x, y, lp, zs, cp, nlp=nlp)
                    return dict(ax=o.ax, ay=o.ay, kap=o.kap, gam1=o.gam1, gam2=o.gam2, rot=o.rot,
                                muinv=o.muinv, td=o.td)
            self._finder_cache[key] = f
        return self._finder_cache[key]

    def _lens_centers(self, lp, nlp):
        ls = self.lens_system
        c0 = []
        cp = [[] for _ in range(ls.num_planes)]
        for i, m in enumerate(self.lens_models):
            k = ls.plane_of_lens[i]
            pts = self.gals[:, :2] if m == "gals" else lp[i:i + 1, 2:4]
            (c0 if k == 0 else cp[k]).extend(list(np.asarray(pts)))
        planes = [None] + [np.array(cp[j]).reshape(-1, 2) for j in range(1, nlp)]
        return np.array(c0).reshape(-1, 2), planes

    def find_images_numeric(self, lp, cp, zs, xs, ys, point_index=None):
        """glafic findimg for concrete parameters; returns (n, 4) [x, y, mag, td]."""
        nlp = self.nlp_point[point_index] if point_index is not None else self.lens_system.nlp_for(zs)
        f = self._jit_lens(nlp)
        lpj = jnp.asarray(lp)
        cpj = jnp.asarray(cp)
        lens_fn = lambda x, y: f(x, y, lpj, cpj, zs)
        planes_fn = None
        c0, cpl = self._lens_centers(np.asarray(lp), nlp)
        if nlp > 1:
            g = self._jit_lens(nlp, planes=True)
            planes_fn = lambda x, y: [tuple(np.asarray(v) for v in pl) for pl in g(x, y, lpj, cpj, zs)]
        finder = ImageFinder(lens_fn, self.settings, c0, planes_fn, cpl[1:] if nlp > 1 else None)
        return finder.find(xs, ys)

    def findimg(self, i=None, P=None):
        """Images of point source i (1-based) or of all sources (list)."""
        P = self.params() if P is None else P
        lp = np.asarray(P["lens"])
        cp = np.asarray(P["cosmo"])
        ids = range(self.num_poi) if i is None else [i - 1]
        out = [self.find_images_numeric(lp, cp, float(P["point"][k, 0]), float(P["point"][k, 1]),
                                        float(P["point"][k, 2]), point_index=k) for k in ids]
        return out if i is None else out[0]

    def point_solve(self, zs, xs, ys, P=None):
        P = self.params() if P is None else P
        return self.find_images_numeric(np.asarray(P["lens"]), np.asarray(P["cosmo"]), zs, xs, ys)

    # ------------------------------------------------------------------
    # chi^2 and log-probability
    def chi2_components(self, P, point_mode="glafic", ext_mode="glafic", n_sub=3):
        s = self.settings
        out = {}
        bad_l = self.range_bad_lens(P)
        c2l, badl2 = self.chi2prior_lens(P)
        out["prior_lens"] = c2l
        tot = c2l
        bad = bad_l | badl2
        if self.obs_image is not None and self.num_ext > 0:
            c2e = ext_mod.chi2_extend_data(self, P, mode=ext_mode, n_sub=n_sub, point_mode=point_mode)
            c2pe, bade = self.chi2prior_ext(P)
            out["extend"] = c2e
            out["prior_ext"] = c2pe
            tot = tot + c2e + c2pe
            bad = bad | bade | self.range_bad_ext(P)
        batched = (self._point_batch is not None and len(self._point_batch["nlp"]) == 1 and
                   (s.chi2_splane == 1 or point_mode == "tracked"))
        if batched:
            mode = "splane" if s.chi2_splane == 1 else "tracked"
            terms = point_mod.chi2_point_batched(self, P, mode=mode, return_terms=True,
                                                 source_mode=self.source_mode)
            out["point_terms"] = terms
            out["point"] = jnp.sum(terms[0])
            tot = tot + out["point"]
        elif any(o is not None for o in self.obs_point):
            c2p = 0.0
            for i, ob in enumerate(self.obs_point):
                if ob is None or ob["tab"].shape[0] == 0:
                    continue
                if s.chi2_splane == 1:
                    r = point_mod.chi2_splane_single(self, P, i)
                else:
                    r = point_mod.chi2_iplane_single(self, P, i, mode=point_mode)
                out[f"point{i + 1}"] = r
                c2p = c2p + r[0]
            out["point"] = c2p
            tot = tot + c2p
        out["bad"] = bad
        out["total"] = jnp.where(bad, s.chi2pen_range, tot)
        return out

    def chi2(self, P=None, **kw):
        P = self.params() if P is None else P
        return self.chi2_components(P, **kw)["total"]

    def log_prob(self, theta, **kw):
        """-chi^2/2 with -inf for anything glafic would penalise (ranges,
        wrong number of images, wrong parity)."""
        P = self.unflatten(theta)
        c2 = self.chi2(P, **kw)
        pen = 0.5 * min(self.settings.chi2pen_range, self.settings.chi2pen_nimg, self.settings.chi2pen_parity)
        lp = -0.5 * c2
        if self.extra_log_prior is not None:
            lp = lp + self.extra_log_prior(P)
        return jnp.where(jnp.isfinite(c2) & (c2 < pen) & jnp.isfinite(lp), lp, -jnp.inf)

    def make_log_prob(self, jit=True, **kw):
        f = functools.partial(self.log_prob, **kw)
        return jax.jit(f) if jit else f

    # ------------------------------------------------------------------
    def model_image(self, P=None, compute_all=True, ext_mode="glafic", n_sub=3, point_mode="glafic",
                    flag_source=0, sky=0.0, per_source=False):
        """Model image (ny, nx), like glafic.writeimage(sky=0, noise=0)."""
        P = self.params() if P is None else P
        imgs = ext_mod.model_images(self, P, compute_all=compute_all, mode=ext_mode, n_sub=n_sub,
                                    point_mode=point_mode, flag_source=flag_source)
        return imgs + sky if per_source else jnp.sum(imgs, axis=0) + sky

    def set_source_mode(self, mode):
        """'sample', 'profile' or 'marginalize' (see point.chi2_point_batched)."""
        if mode not in ("sample", "profile", "marginalize"):
            raise ValueError(mode)
        self.source_mode = mode
        return self.model_init()

    def best_sources(self, P):
        """Profiled (weighted least-squares) source positions -> P with them filled in."""
        terms = point_mod.chi2_point_batched(self, P, mode="splane", return_terms=True, source_mode="profile")
        bd = self._point_batch
        P = dict(P)
        P["point"] = P["point"].at[bd["idx"], 1:3].set(terms[5])
        return P

    def update_from(self, P):
        """Write a parameter pytree back into the model (e.g. after optimisation)."""
        self.lens_params = np.asarray(P["lens"]).copy()
        self.ext_params = np.asarray(P["ext"]).copy()
        self.poi_params = np.asarray(P["point"]).copy()
        self.psf_params = np.asarray(P["psf"]).copy()
        self.sky = float(P["sky"])
        c = np.asarray(P["cosmo"])
        self.set_cosmo(*c)
        return self

    def copy(self):
        m = copy.copy(self)
        for k in ("lens_params", "ext_params", "poi_params", "psf_params"):
            setattr(m, k, getattr(self, k).copy())
        m._finder_cache = {}
        return m
