"""Primary and secondary glafic parameters.

All of these are *static* configuration (they are not sampled).  The names
follow the keywords used in glafic input files so that ``Settings.set``
accepts exactly the same strings as ``glafic.set_secondary``.
"""

from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class Settings:
    # primary parameters
    omega: float = 0.3
    lambda_: float = 0.7
    weos: float = -1.0
    hubble: float = 0.7
    prefix: str = "out"
    xmin: float = -60.0
    ymin: float = -60.0
    xmax: float = 60.0
    ymax: float = 60.0
    pix_ext: float = 0.2
    pix_poi: float = 3.0
    maxlev: int = 5

    # secondary parameters
    ovary: int = 0
    lvary: int = 0
    wvary: int = 0
    hvary: int = 0
    #: w0waCDM extension (not in C glafic): w(a) = weos + wa (1 - a)
    wa: float = 0.0
    wavary: int = 0
    galfile: str = "galfile.dat"
    srcfile: str = "srcfile.dat"
    ran_seed: int = -1234
    flag_hodensity: int = 0
    hodensity: float = 200.0
    gnfw_usetab: int = 1
    ein_usetab: int = 1
    nfw_users: int = 0
    nmax_poi_ite: int = 10
    max_poi_tol: float = 1.0e-10
    poi_imag_max: float = 5.0
    poi_imag_min: float = 0.001
    ang_step: int = 90
    imag_ceil: float = 1.0e-10
    smallcore: float = 1.0e-10
    outformat_exp: int = 0
    flatfix: int = 0
    flag_outpot: int = 0

    chi2_splane: int = 0
    chi2_checknimg: int = 1
    chi2_usemag: int = 0
    chi2_usedl: int = 0
    chi2_restart: int = 0
    chi2pen_range: float = 1.0e30
    chi2pen_nimg: float = 1.0e30
    chi2pen_parity: float = 1.0e30
    obs_gain: float = 3.0
    obs_ncomb: int = 1
    obs_readnoise: float = 10.0
    noise_clip: float = 3.0
    skyfix: int = 0
    skyfix_value: float = 1.0e10
    psfconv_size: float = 4.0
    seeing_sub: int = 1
    bicub_a: float = -0.5
    source_calcr0: float = 20.0
    flag_extref: int = 1
    source_refr0: float = 3.0
    num_pixint: int = 5
    flag_extnorm: int = 0
    nmax_fft: int = 20000000
    dr_lens_center: float = 0.9
    flag_pow_tm15: int = 1
    tol_pow_tm15: float = 1.0e-8

    # JAX-port specific numerical knobs (not in the C code)
    #: number of terms of the Tessore & Metcalf (2015) series for ``pow``
    n_tm15_terms: int = 300
    #: panels x nodes of Gauss-Legendre used for elliptical-density integrals
    ell_quad_panels: int = 8
    ell_quad_nodes: int = 16
    #: 1 -> use a continuous (more accurate) lower limit for the elliptical-density
    #: integrals instead of glafic's switch at xi^2 = 10 (removes a ~1e-3 jump)
    ell_accurate: int = 0
    #: >0: cap |mu| at this value in the source-plane chi^2 weights (smooths the
    #: chi^2 for images on critical curves; not in glafic)
    mag_cap: float = 0.0

    # derived quantities --------------------------------------------------
    @property
    def nx_ext(self):
        return int((self.xmax - self.xmin + 1.0e-10) / self.pix_ext)

    @property
    def ny_ext(self):
        return int((self.ymax - self.ymin + 1.0e-10) / self.pix_ext)

    @property
    def nx_poi(self):
        return int((self.xmax - self.xmin + 1.0e-10) / self.pix_poi)

    @property
    def ny_poi(self):
        return int((self.ymax - self.ymin + 1.0e-10) / self.pix_poi)

    def set(self, keyword, value):
        """Return a copy with one parameter changed (glafic keyword names)."""
        name = "lambda_" if keyword == "lambda" else keyword
        if name in ("chi2_point_splane",):
            name = "chi2_splane"
        names = {f.name: f for f in fields(self)}
        if name not in names:
            raise KeyError(f"unknown glafic parameter '{keyword}'")
        typ = names[name].type
        if typ in (int, "int"):
            value = int(float(value))
        elif typ in (float, "float"):
            value = float(value)
        else:
            value = str(value)
        return replace(self, **{name: value})

    def set_secondary(self, line):
        """Parse a line like ``'chi2_splane 1'`` (same as glafic.set_secondary)."""
        toks = line.split()
        return self.set(toks[0], toks[1])

    @property
    def cosmo(self):
        return (self.omega, self.lambda_, self.weos, self.hubble, self.wa)


def default_settings(**kw):
    s = Settings()
    for k, v in kw.items():
        s = s.set(k, v)
    return s
