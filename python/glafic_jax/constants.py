"""Physical and numerical constants (mirrors the ``#define``s in glafic.h)."""

ARCSEC2RADIAN = 0.00000484813681109536
COVERH_MPCH = 2997.92458
FAC_CRITDENS = 5.5467248e+14
FAC_TDELAY_DAY = 3.571386089689082e12
MPC2METER = 3.085677581e22
R_SCHWARZ = 2953.339382
C_LIGHT_KMS = 2.99792458e5
NFW_RS_NORM = 0.00009510361
NFW_B_NORM = 6.34482175e-8
FAC_SIE_CALCEIN = 2.08955538e19

NPAR_LEN = 8
NPAR_POI = 3
NPAR_EXT = 8
NPAR_PSF = 9

TOL_ZLPL = 1.0e-3
TOL_CURVATURE = 1.0e-6
OFFSET_TDELAY_FAC = 1.0e-300
OFFSET_LOG = 1.0e-300
NPIX_SMALL_OFFSET = 1.0e-10
TOL_ZS = 1.0e-6
ULIM_JHK = 1.0e-8
DIS2_MIN_SET = 1.0e60
SERSIC_N_MIN = 0.06
SERSIC_N_MAX = 20.0
NMAX_POIMG = 50

# initial ranges used for parameter checks (glafic.h INIT_*)
INIT_MMIN, INIT_MMAX = 0.0, 1.0e30
INIT_XYMIN, INIT_XYMAX = -1.0e30, 1.0e30
INIT_EMIN, INIT_EMAX = 0.0, 1.0
INIT_ANGMIN, INIT_ANGMAX = -360.0, 360.0
INIT_R0MIN, INIT_R0MAX = 0.0, 1.0e30
INIT_NMIN, INIT_NMAX = -1.0e30, 1.0e30
INIT_ZMIN, INIT_ZMAX = 1.0e-6, 1.0e4
INIT_BETAMIN, INIT_BETAMAX = 1.0, 1.0e30
INIT_FMIN, INIT_FMAX = 0.0, 1.0
INIT_OMMIN, INIT_OMMAX = -1.0, 3.0
INIT_WMIN, INIT_WMAX = -5.0, 1.0
INIT_HMIN, INIT_HMAX = 0.0, 3.0

LENS_MODELS = [
    "gals", "nfwpot", "sie", "jaffe", "point", "pert", "clus3", "mpole",
    "hernpot", "nfw", "hern", "powpot", "pow", "gnfwpot", "gnfw", "serspot",
    "sers", "tnfwpot", "tnfw", "einpot", "ein", "anfw", "ahern", "crline",
    "gaupot", "acnfw",
]

EXTEND_MODELS = ["gauss", "sersic", "tophat", "moffat", "jaffe", "srcs", "point"]
