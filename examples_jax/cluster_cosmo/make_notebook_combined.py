"""Generate cluster_cosmo_combined.ipynb: combine the per-cluster cosmology chains, add CMB and SN.

    python make_notebook_combined.py

The notebook reads the NUTS chains written by cluster_cosmo_models.ipynb in
results/<cluster>/ (flat ΛCDM, wCDM, w0waCDM), turns each cluster's
cosmological posterior into a density on a grid (the lens parameters are
marginalised by the chain itself), multiplies the clusters together as in
Caminha et al. (2021, Eq. 8), and combines the strong-lensing (SL) result with
Planck 2018 CMB distance priors and Pantheon+ supernovae.

Everything it writes goes to results/combined/ (results/combined_<a>_<b>/
for a subset of clusters).
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Cluster strong-lensing cosmography: combining clusters, CMB and supernovae

This notebook follows Caminha et al. (2021, A&A, *Galaxy cluster strong lensing
cosmography*): it combines the cosmological posteriors of several clusters
and then adds the classical probes.

1. **Per-cluster posteriors.** For every cluster in `CLUSTERS` it loads the
   NUTS chains of `cluster_cosmo_models.ipynb` (`results/<cluster>/nuts_*.npz`)
   for flat ΛCDM (Ω_m), wCDM (Ω_m, w) and w0waCDM (Ω_m, w0, wa). The lens
   parameters are already marginalised: only the cosmological columns are used.
2. **Strong lensing (SL) combined.** Each cluster's samples are turned into a
   density on a grid. Its product over clusters is the joint SL posterior
   (Caminha et al. Eq. 8). This is exact because every cluster used the same
   flat prior on the same range, so the prior is counted once.
3. **CMB and SN.** Planck 2018 distance priors (R, l_A, ω_b; Chen, Huang &
   Wang 2019) and the Pantheon+ supernovae (Brout et al. 2022) are sampled
   with NUTS, alone and multiplied with the SL grid, giving a table like
   Caminha et al. Table 2. That table includes the improvement factors δ: the
   68% width without SL divided by the width with SL.

| cosmology | parameters | per-cluster chain |
|---|---|---|
| flat ΛCDM | Ω_m | `nuts_lcdm_om.npz` |
| flat wCDM | Ω_m, w (w < 0) | `nuts_wcdm_desi_wlt0.npz` |
| flat w0waCDM (CPL) | Ω_m, w0, wa (w0 + wa < 0) | `nuts_w0wa_desi.npz` |

The curved ΛCDM model of Caminha et al. is not included: the cluster
notebooks do not sample Ω_k.

**Inputs.** Run `cluster_cosmo_models.ipynb` (or `run_caminha.py`) for each
cluster first. The Pantheon+ data (about 35 MB) are downloaded once into
`external_data/pantheonplus/`.

**Outputs.** Everything goes to `results/combined/`: figures as PNG, the
summary table, and the cached chains of every probe combination. The cache is
stamped with a fingerprint of the cluster chains and the settings. A
combination is re-run when either changes. `setup.json` records the clusters
and their model tags, and the notebook stops if they change.
""")

md(r"""
## 1. Configuration
""")

code(r"""
import os
import json
import hashlib
import time
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import numpyro
numpyro.set_host_device_count(4)          # parallel NUTS chains on the CPU (before JAX starts)
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary as np_summary
jax.config.update("jax_enable_x64", True)

HERE = os.getcwd()
RESULTS = os.path.join(HERE, "results")

ALL_CLUSTERS = ["rj2129", "as1063", "mj1931", "mj0329", "mj2129"]    # Caminha et al. (2021), by lens redshift
CLUSTERS = os.environ.get("GLAFIC_CLUSTERS", "").split(",") if os.environ.get("GLAFIC_CLUSTERS") else ALL_CLUSTERS
RUN_DIR = os.path.join(RESULTS, "combined" if CLUSTERS == ALL_CLUSTERS else "combined_" + "_".join(CLUSTERS))
os.makedirs(RUN_DIR, exist_ok=True)

# per-cluster chains, as named by cluster_cosmo_models.ipynb (W_PRIOR = "desi", W_MAX_WCDM = 0)
W_MAX_WCDM = 0.0
W0WA_CUT = 0.0                            # w0 + wa < W0WA_CUT in w0waCDM (COSMO_REPARAM "sum_below")
NUTS_FILE = {"lcdm": "nuts_lcdm_om.npz", "wcdm": f"nuts_wcdm_desi_wlt{W_MAX_WCDM:g}.npz",
             "w0wa": "nuts_w0wa_desi.npz"}
COSMO = {"lcdm": "flat ΛCDM", "wcdm": "flat wCDM", "w0wa": "flat w0waCDM"}
PARAMS = {"lcdm": ["omega"], "wcdm": ["omega", "weos"], "w0wa": ["omega", "weos", "wa"]}
LABEL = {"omega": r"$\Omega_m$", "weos": r"$w_0$", "wa": r"$w_a$"}
TXT = {"omega": "Ω_m", "weos": "w0 (w)", "wa": "wa"}
REF = {"omega": 0.3, "weos": -1.0, "wa": 0.0}

# grid density of each cluster's posterior
GRID_N = {"lcdm": 400, "wcdm": 160, "w0wa": 64}   # cells per axis
BW_SCALE = 1.0          # kernel width x Scott's rule (n = the chain's ESS)
FLOOR = 1e-6            # density floor (x the cluster's peak) before the product: unsampled tails are not zero

# probe combinations sampled with NUTS (SL alone comes straight from the grid)
PROBES = ["SN", "CMB", "SL+SN", "SL+CMB", "CMB+SN", "SL+CMB+SN"]
NUTS_KW = dict(num_warmup=1000, num_samples=2500, num_chains=4)
IS_MIN_ESS = 1500       # SL × X by reweighting X's chain when the weights keep this many effective samples
SMOKE_TEST = False
if SMOKE_TEST:
    NUTS_KW = dict(num_warmup=150, num_samples=200, num_chains=2)
    GRID_N = {"lcdm": 200, "wcdm": 80, "w0wa": 32}


def show(name):
    # save the current figure as results/combined/<name>.png and display it
    plt.savefig(os.path.join(RUN_DIR, f"{name}.png"), dpi=130, bbox_inches="tight")
    plt.show()


def save_txt(fname, txt):
    print(txt)
    open(os.path.join(RUN_DIR, fname), "w").write(txt + "\n")


print("clusters:", ", ".join(CLUSTERS))
print("output:  ", os.path.relpath(RUN_DIR, HERE))
""")

md(r"""
## 2. Per-cluster chains

Each cluster's `setup.json` gives its model tag. Every chain must carry the
same cluster name and tag (the same check as `check_owner` in the cluster
notebook), so a chain from another model setup is never mixed in. All
clusters must also share the prior ranges in their `<cluster>_prior.dat`.
These ranges and the wCDM/w0waCDM cuts fix the domain on which densities are
estimated.
""")

code(r"""
def prior_box(cluster):
    box = {}
    for ln in open(os.path.join(RESULTS, cluster, f"{cluster}_prior.dat")):
        t = ln.split()
        if len(t) == 4 and t[0] == "range" and t[1] in ("omega", "weos", "wa"):
            box[t[1]] = (float(t[2]), float(t[3]))
    return box


boxes = {c: prior_box(c) for c in CLUSTERS}
BOX = boxes[CLUSTERS[0]]
for c, b in boxes.items():
    if b != BOX:
        raise RuntimeError(f"{c}: prior ranges {b} differ from {CLUSTERS[0]}'s {BOX}; the product of posteriors "
                           "needs one common flat prior")
DOMAIN = {"lcdm": {"omega": BOX["omega"]},
          "wcdm": {"omega": BOX["omega"], "weos": (BOX["weos"][0], W_MAX_WCDM)},
          "w0wa": {"omega": BOX["omega"], "weos": BOX["weos"], "wa": BOX["wa"]}}

tags, chains, fp = {}, {}, hashlib.sha1()
for c in CLUSTERS:
    setup = json.load(open(os.path.join(RESULTS, c, "setup.json")))
    tags[c] = setup["tag"]
    for k, f in NUTS_FILE.items():
        path = os.path.join(RESULTS, c, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: run cluster_cosmo_models.ipynb for {c} first")
        d = np.load(path)
        if str(d["cluster"]) != c or str(d["tag"]) != tags[c]:
            raise RuntimeError(f"{path} belongs to {d['cluster']} / {d['tag']}, not {c} / {tags[c]}")
        names = list(d["names"]); idx = [names.index(n) for n in PARAMS[k]]
        chains.setdefault(k, {})[c] = dict(x=d["theta"].reshape(-1, len(names))[:, idx],
                                           ess=d["ess"][idx], rhat=d["rhat"][idx],
                                           div=int(d["diverging"].sum()), n=d["diverging"].size)
        fp.update(open(path, "rb").read())

# the combined folder holds one set of clusters/tags; stop if that changed (move or rename the folder)
setup_path = os.path.join(RUN_DIR, "setup.json")
setup = {"clusters": CLUSTERS, "tags": tags}
if os.path.exists(setup_path):
    old = json.load(open(setup_path))
    if old != setup:
        raise RuntimeError(f"{setup_path} was made for {old}; now {setup}. Move or rename {RUN_DIR} to keep it.")
else:
    json.dump(setup, open(setup_path, "w"), indent=1)

print(f"{'cluster':8s} {'model':8s} {'samples':>8s} {'div':>5s}   " + "   ".join(f"{TXT[p]:>14s}" for p in PARAMS["w0wa"]))
for k in NUTS_FILE:
    for c in CLUSTERS:
        r = chains[k][c]
        cells = [f"R̂ {rh:.3f} ESS {e:4.0f}" for rh, e in zip(r["rhat"], r["ess"])]
        print(f"{c:8s} {k:8s} {len(r['x']):8d} {r['div']:5d}   " + "   ".join(f"{s:>14s}" for s in cells))
print("\nmodel tags:", json.dumps(tags, indent=1))
""")

md(r"""
## 3. From samples to densities

Caminha et al. multiply each cluster's posterior density, marginalised over
its mass parameters (their Eq. 8). Here each density is estimated on a grid
over the prior domain:

* the samples are histogrammed and smoothed with a Gaussian kernel. Its width
  per axis follows Scott's rule, σ n^(−1/(d+4)), with n the chain's effective
  sample size (not the number of draws, since NUTS draws are correlated);
* the same kernel is applied to the domain indicator, and the smoothed
  histogram is divided by it. This removes the leak through the prior edges
  (Ω_m = 0.05 or 1, w = 0, w0 + wa = 0), where several posteriors pile up.
  A plain KDE would underestimate the density there by up to a factor of two.

Before multiplying, each density is floored at `FLOOR` × its peak: no chain
samples its far tails, but they are not zero. **If the combined peak sits
where one cluster has almost no samples, the clusters disagree.** The product
then depends on the floor and the kernel, not on the data. The leave-one-out
table below shows whether this happens.
""")

code(r"""
def grid_axes(k):
    edges = [np.linspace(*DOMAIN[k][p], GRID_N[k] + 1) for p in PARAMS[k]]
    centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
    return edges, centers


def domain_fraction(k, edges, sub=8):
    # fraction of each grid cell inside the prior domain: 1 in a box; the w0 + wa cut by sub-sampling
    shape = [len(e) - 1 for e in edges]
    if k != "w0wa":
        return np.ones(shape)
    ew, ea = edges[1], edges[2]
    t = (np.arange(sub) + 0.5) / sub
    w = (ew[:-1, None] + np.diff(ew)[:, None] * t)          # (n, sub)
    a = (ea[:-1, None] + np.diff(ea)[:, None] * t)
    inside = (w[:, None, :, None] + a[None, :, None, :]) < W0WA_CUT
    f2 = inside.mean(axis=(2, 3))                             # (n_w, n_a)
    return np.broadcast_to(f2[None], shape).copy()


def grid_density(k, x, ess):
    # boundary-corrected smoothed histogram, normalised over the domain; also defined just outside it
    edges, _ = grid_axes(k)
    dim = x.shape[1]
    width = np.array([e[1] - e[0] for e in edges])
    neff = max(float(np.min(ess)), 50.0)
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0)) / 1.349
    bw = BW_SCALE * np.minimum(x.std(0), iqr) * neff ** (-1.0 / (dim + 4))
    s = np.maximum(bw / width, 0.5)
    frac = domain_fraction(k, edges)
    num = ndimage.gaussian_filter(np.histogramdd(x, bins=edges)[0], s, mode="constant")
    den = ndimage.gaussian_filter(frac, s, mode="constant")
    p = np.where(den > 1e-3, num / np.maximum(den, 1e-300), 0.0)
    return p / ((p * frac).sum() * np.prod(width)), bw


GRID = {k: dict(zip(("edges", "centers"), grid_axes(k))) for k in NUTS_FILE}
for k in NUTS_FILE:
    GRID[k]["frac"] = domain_fraction(k, GRID[k]["edges"])
    GRID[k]["vol"] = np.prod([e[1] - e[0] for e in GRID[k]["edges"]])
DENS, BW = {}, {}
for k in NUTS_FILE:
    for c in CLUSTERS:
        DENS[(k, c)], BW[(k, c)] = grid_density(k, chains[k][c]["x"], chains[k][c]["ess"])
    print(f"{COSMO[k]:14s} kernel widths: " + "; ".join(
        f"{c} " + ", ".join(f"{b:.3f}" for b in BW[(k, c)]) for c in CLUSTERS))


def combine(k, clusters):
    # product of the clusters' densities (floored), normalised over the domain
    logp = sum(np.log(np.maximum(DENS[(k, c)], FLOOR * DENS[(k, c)].max())) for c in clusters)
    p = np.exp(logp - logp.max())
    return p / ((p * GRID[k]["frac"]).sum() * GRID[k]["vol"]), logp


def marginal(k, p, param):
    # 1-d marginal on the grid: (centers, pdf)
    i = PARAMS[k].index(param)
    other = tuple(j for j in range(p.ndim) if j != i)
    m = (p * GRID[k]["frac"]).sum(axis=other) if other else p * GRID[k]["frac"]
    e = GRID[k]["edges"][i]
    return GRID[k]["centers"][i], m / (m.sum() * (e[1] - e[0]))


def grid_quantiles(k, p, param, q=(0.16, 0.5, 0.84)):
    x, m = marginal(k, p, param)
    e = GRID[k]["edges"][PARAMS[k].index(param)]
    cdf = np.concatenate([[0.0], np.cumsum(m) * (e[1] - e[0])])
    return np.interp(q, cdf, e)


def grid_samples(k, p, n=200_000, seed=0):
    # draws from a grid density: a cell by its mass, then uniform inside it (rejecting the cut part of a cell)
    rng = np.random.default_rng(seed)
    w = (p * GRID[k]["frac"]).ravel(); w = w / w.sum()
    out = []
    while sum(len(o) for o in out) < n:
        cell = np.unravel_index(rng.choice(w.size, size=n, p=w), p.shape)
        x = np.stack([e[i] + rng.random(n) * (e[1] - e[0]) for e, i in zip(GRID[k]["edges"], cell)], axis=1)
        if k == "w0wa":
            x = x[x[:, 1] + x[:, 2] < W0WA_CUT]
        out.append(x)
    return np.concatenate(out)[:n]


SL = {k: combine(k, CLUSTERS)[0] for k in NUTS_FILE}
SL_LOGP = {}
for k in NUTS_FILE:
    lp = np.log(np.maximum(SL[k], 1e-300))
    SL_LOGP[k] = np.maximum(lp, lp.max() - 200.0)   # for the NUTS interpolation: finite everywhere
""")

md(r"""
### Check: grid density against the chain

The grid density's median and 68% interval should reproduce the chain's own
quantiles for every cluster. A large mismatch means the grid is too coarse
or the kernel too wide (`GRID_N`, `BW_SCALE`).
""")

code(r"""
def fmt(q):
    return f"{q[1]:6.2f} [{q[0]:5.2f}, {q[2]:5.2f}]"


lines = [f"{'':8s}{'':10s}" + "".join(f"{TXT[p]:>24s}" for p in PARAMS["w0wa"])]
worst = 0.0
for k in NUTS_FILE:
    for c in CLUSTERS:
        x = chains[k][c]["x"]
        qs = [np.percentile(x[:, i], [16, 50, 84]) for i in range(x.shape[1])]
        qg = [grid_quantiles(k, DENS[(k, c)], p) for p in PARAMS[k]]
        for a, b, p in zip(qs, qg, PARAMS[k]):
            worst = max(worst, np.max(np.abs(a - b)) / (a[2] - a[0]))
        lines.append(f"{c:8s}{k:10s}" + "".join(f"{fmt(a):>24s}" for a in qs) + "   chain")
        lines.append(f"{'':8s}{'':10s}" + "".join(f"{fmt(b):>24s}" for b in qg) + "   grid")
lines.append(f"\nlargest quantile shift: {worst:.2f} x the 68% width")
print("\n".join(lines))
""")

md(r"""
## 4. Strong lensing: clusters combined

As in Caminha et al. Figs. 1, 2 (left) and 4: each cluster's posterior and
the product. ΛCDM shows the 1-d PDFs with their 68% intervals shaded. wCDM and
w0waCDM show each cluster's 68% region and the combined 68% and 95% regions
(grey). The yellow dot marks the combined medians.
""")

code(r"""
CL_COLORS = dict(zip(ALL_CLUSTERS, ["tab:blue", "tab:pink", "orange", "tab:green", "tab:red"]))
ZL = {}
for c in CLUSTERS:
    for ln in open(os.path.join(RESULTS, c, f"{c}.input")):
        t = ln.split()
        if len(t) >= 3 and t[0] == "lens":          # lens <type> <zl> ...
            ZL[c] = float(t[2]); break
order = sorted(CLUSTERS, key=lambda c: ZL.get(c, 0))


def hpd_level(p, frac, mass):
    # density threshold enclosing `mass` of the probability (highest-density region)
    w = (p * frac).ravel(); v = p.ravel()
    o = np.argsort(v)[::-1]
    cum = np.cumsum(w[o]); cum /= cum[-1]
    return v[o][np.searchsorted(cum, mass)]


def contour2d(ax, k, p, i, j, color, levels=(0.68,), fill=False, lw=1.5, zorder=2):
    # 68%/95% regions of the (i, j) marginal of a grid density
    axes_other = tuple(a for a in range(p.ndim) if a not in (i, j))
    m = (p * GRID[k]["frac"]).sum(axis=axes_other) if axes_other else p * GRID[k]["frac"]
    f2 = GRID[k]["frac"].max(axis=axes_other) if axes_other else GRID[k]["frac"]
    m = m if i < j else m.T                   # rows: parameter i
    f2 = f2 if i < j else f2.T
    m = ndimage.gaussian_filter(m, 0.7)
    lev = sorted(hpd_level(m, np.ones_like(m), L) for L in levels)
    X, Y = GRID[k]["centers"][j], GRID[k]["centers"][i]
    if fill:
        cols = [color] * len(lev)
        alphas = np.linspace(0.35, 0.75, len(lev))
        for L, a in zip(lev, alphas):
            ax.contourf(X, Y, m, levels=[L, m.max() * 1.01], colors=[color], alpha=a, zorder=zorder)
    ax.contour(X, Y, m, levels=lev, colors=[color], linewidths=lw, zorder=zorder)


# ΛCDM: 1-d PDFs
k = "lcdm"
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for c in order:
    x, m = marginal(k, DENS[(k, c)], "omega")
    lo, _, hi = grid_quantiles(k, DENS[(k, c)], "omega")
    ax.plot(x, m, color=CL_COLORS[c], label=f"{c}  $z_L$={ZL.get(c, np.nan):.2f}")
    ax.fill_between(x, m, where=(x > lo) & (x < hi), color=CL_COLORS[c], alpha=0.15)
x, m = marginal(k, SL[k], "omega")
lo, _, hi = grid_quantiles(k, SL[k], "omega")
ax.plot(x, m, color="k", lw=1.5, label="combined")
ax.fill_between(x, m, where=(x > lo) & (x < hi), color="0.3", alpha=0.6)
ax.axvline(0.3, color="0.5", ls=":", lw=0.8)
ax.set_xlabel(LABEL["omega"], fontsize=13); ax.set_ylabel("PDF"); ax.set_xlim(DOMAIN[k]["omega"]); ax.set_ylim(0)
ax.legend(fontsize=9); ax.set_title("strong lensing, flat ΛCDM")
show("sl_lcdm")

# wCDM and w0waCDM: per-cluster 68% regions and the combined 68/95%
for k in ("wcdm", "w0wa"):
    pairs = [(1, 0)] if k == "wcdm" else [(1, 0), (2, 0), (2, 1)]
    fig, axs = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.8), squeeze=False)
    med = [grid_quantiles(k, SL[k], p)[1] for p in PARAMS[k]]
    for ax, (i, j) in zip(axs[0], pairs):
        for c in order:
            contour2d(ax, k, DENS[(k, c)], i, j, CL_COLORS[c], lw=1.2)
        contour2d(ax, k, SL[k], i, j, "0.25", levels=(0.68, 0.95), fill=True, zorder=3)
        ax.plot(med[j], med[i], "o", color="gold", mec="k", ms=7, zorder=4)
        ax.axvline(REF[PARAMS[k][j]], color="0.5", ls=":", lw=0.8); ax.axhline(REF[PARAMS[k][i]], color="0.5", ls=":", lw=0.8)
        ax.set_xlabel(LABEL[PARAMS[k][j]], fontsize=13); ax.set_ylabel(LABEL[PARAMS[k][i]], fontsize=13)
        ax.set_xlim(DOMAIN[k][PARAMS[k][j]]); ax.set_ylim(DOMAIN[k][PARAMS[k][i]])
        if k == "w0wa" and (i, j) == (2, 1):
            w = np.array(DOMAIN[k]["weos"]); ax.fill_between(w, W0WA_CUT - w, DOMAIN[k]["wa"][1] + 1, color="k", alpha=0.85)
            ax.text(0.97, 0.95, r"$w_0+w_a>0$", color="w", ha="right", va="top", transform=ax.transAxes)
    handles = [plt.Line2D([], [], color=CL_COLORS[c]) for c in order] + [plt.Rectangle((0, 0), 1, 1, color="0.25")]
    fig.legend(handles, [f"{c}  $z_L$={ZL.get(c, np.nan):.2f}" for c in order] + ["combined (68, 95%)"],
               **(dict(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6) if k == "w0wa"
                  else dict(loc="upper right", bbox_to_anchor=(0.98, 0.9))), fontsize=9)
    fig.suptitle(f"strong lensing, {COSMO[k]}: clusters 68%, combined 68/95%")
    plt.tight_layout(); show(f"sl_{k}")
""")

md(r"""
### Consistency between clusters: leave one out

The combined SL constraint is recomputed with each cluster left out. If
dropping one cluster moves the result by much more than its error bar, that
cluster disagrees with the rest. Its mass model (or its chain) then sets the
combination, and the product's peak lies in the other clusters' poorly
sampled tails. Caminha et al. saw a milder case with RX J2129 in ΛCDM.

The last column is the probability, under the product of the other clusters,
of the left-out cluster's median. Values ≪ 1% flag a tension.
""")

code(r"""
lines = [f"{'cosmology':10s} {'left out':10s}" + "".join(f"{TXT[p]:>24s}" for p in PARAMS["w0wa"])
         + f"{'tension':>12s}"]
for k in NUTS_FILE:
    full = [grid_quantiles(k, SL[k], p) for p in PARAMS[k]]
    lines.append(f"{k:10s} {'(none)':10s}" + "".join(f"{fmt(q):>24s}" for q in full))
    for c in CLUSTERS:
        rest = [o for o in CLUSTERS if o != c]
        if not rest:
            continue
        p, _ = combine(k, rest)
        q = [grid_quantiles(k, p, par) for par in PARAMS[k]]
        # how probable is the left-out cluster's median under the rest? mass of the rest's posterior
        # at densities lower than that at the median (a p-value-like number; 1 = at the peak)
        med = np.median(chains[k][c]["x"], axis=0)
        idx = tuple(int(np.clip(np.searchsorted(e, v) - 1, 0, len(e) - 2)) for e, v in zip(GRID[k]["edges"], med))
        w = (p * GRID[k]["frac"]).ravel(); w /= w.sum()
        pval = w[p.ravel() <= p[idx]].sum()
        lines.append(f"{'':10s} {c:10s}" + "".join(f"{fmt(v):>24s}" for v in q) + f"{100 * pval:11.2f}%")
save_txt("sl_leave_one_out.txt", "\n".join(lines))
""")

md(r"""
## 5. CMB and supernova likelihoods

**CMB: Planck 2018 distance priors** (Chen, Huang & Wang 2019, JCAP 02, 028;
TT,TE,EE+lowE, base ΛCDM). The CMB is compressed into the shift parameter
R = √(Ω_m H0²) D_M(z*)/c, the acoustic scale l_A = π D_M(z*)/r_s(z*) and
ω_b = Ω_b h². These depend on h and ω_b too, which are sampled
(flat, h ∈ [0.4, 1], ω_b ∈ [0.005, 0.04]) and marginalised. Details:

* z* from the Hu & Sugiyama (1996) fit. g₁'s coefficient is 0.0783: Chen et
  al.'s Eq. 9 prints 0.0738, a typo (their code uses 0.0783);
* Ω_r = Ω_m / (1 + z_eq), z_eq = 2.5×10⁴ Ω_m h² (T_CMB/2.7 K)⁻⁴ (their Eq. 6);
* r_s with the sound speed c/√(3(1 + 3ω_b a / 4ω_γ)), 3/(4ω_γ) = 31500 (T_CMB/2.7 K)⁻⁴;
* the inverse covariance of their ΛCDM fit, used for all three cosmologies
  as in their paper.

This is a compression, not the full Planck likelihood that Caminha et al.
use. It is accurate for dark-energy models that do not change the
pre-recombination physics, which holds for wCDM and for w0waCDM with
w0 + wa < 0.

**SN: Pantheon+** (Scolnic et al. 2022; Brout et al. 2022): `m_b_corr` with
the full statistical + systematic covariance. SNe with z_HD < 0.01 are left
out (peculiar velocities), and the SH0ES Cepheid distances are not used. The
absolute magnitude M, degenerate with H0, is marginalised analytically
(flat prior), so SNe constrain only the shape of D_L(z), with
d_L = (1 + z_hel) D_C(z_HD).

The two checks below should give R ≈ 1.750 and Ω_m ≈ 0.316, h ≈ 0.674
(Planck 2018 ΛCDM) from a fit to the CMB priors alone.
""")

code(r"""
# ---- background ----
THETA27 = 2.7255 / 2.7


def e2(a, om, orad, w0, wa):
    # H²/H0² for flat w0waCDM (CPL) with radiation
    fde = a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))
    return orad * a ** -4 + om * a ** -3 + (1.0 - om - orad) * fde


# ---- CMB distance priors (Chen, Huang & Wang 2019; Planck 2018 TT,TE,EE+lowE) ----
CMB_MEAN = jnp.array([1.750235, 301.4707, 0.02235976])           # R, l_A, ω_b
CMB_ICOV = jnp.array([[94392.3971, -1360.4913, 1664517.2916],
                      [-1360.4913, 161.4349, 3671.6180],
                      [1664517.2916, 3671.6180, 79719182.5162]])
_T = jnp.linspace(0.0, 1.0, 1025)            # l_A good to 0.001 (σ = 0.09)


def cmb_obs(om, h, obh2, w0, wa):
    omh2 = om * h * h
    orad = om / (1.0 + 2.5e4 * omh2 * THETA27 ** -4)
    g1 = 0.0783 * obh2 ** -0.238 / (1.0 + 39.5 * obh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * obh2 ** 1.81)
    zs = 1048.0 * (1.0 + 0.00124 * obh2 ** -0.738) * (1.0 + g1 * omh2 ** g2)
    las = -jnp.log1p(zs)
    la = las * _T; a = jnp.exp(la)                                   # ln a from 0 down to ln a*
    dm = -jnp.trapezoid(1.0 / (a * jnp.sqrt(e2(a, om, orad, w0, wa))), la)        # D_M(z*) H0/c
    lb = jnp.log(1e-8) + (las - jnp.log(1e-8)) * _T; b = jnp.exp(lb)
    rb = 31500.0 * THETA27 ** -4 * obh2                              # 3 ω_b / (4 ω_γ)
    cs = 1.0 / jnp.sqrt(3.0 * (1.0 + rb * b))
    rs = jnp.trapezoid(cs / (b * jnp.sqrt(e2(b, om, orad, w0, wa))), lb)            # r_s(z*) H0/c
    return jnp.array([jnp.sqrt(om) * dm, jnp.pi * dm / rs, obh2])


def chi2_cmb(om, h, obh2, w0, wa):
    d = cmb_obs(om, h, obh2, w0, wa) - CMB_MEAN
    return d @ CMB_ICOV @ d


# ---- Pantheon+ ----
PANTHEON_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
                "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
PANTHEON_DIR = os.path.join(HERE, "external_data", "pantheonplus")
os.makedirs(PANTHEON_DIR, exist_ok=True)
for f in ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"):
    path = os.path.join(PANTHEON_DIR, f)
    if not os.path.exists(path):
        print("downloading", f)
        urllib.request.urlretrieve(PANTHEON_URL + f.replace("+", "%2B"), path)

tab = np.genfromtxt(os.path.join(PANTHEON_DIR, "Pantheon+SH0ES.dat"), names=True, dtype=None, encoding="utf-8")
with open(os.path.join(PANTHEON_DIR, "Pantheon+SH0ES_STAT+SYS.cov")) as fh:
    nsn = int(fh.readline())
    cov = np.array(fh.read().split(), dtype=float).reshape(nsn, nsn)
assert nsn == len(tab)
sel = tab["zHD"] > 0.01
SN_ZHD = jnp.array(tab["zHD"][sel]); SN_ZHEL = jnp.array(tab["zHEL"][sel]); SN_M = jnp.array(tab["m_b_corr"][sel])
icov = np.linalg.inv(cov[np.ix_(sel, sel)])
SN_ICOV = jnp.array(icov)
SN_A = jnp.array(icov.sum(axis=1)); SN_B = float(icov.sum())
_ZG = jnp.linspace(0.0, float(SN_ZHD.max()) * 1.001, 1501)
print(f"Pantheon+: {int(sel.sum())} of {nsn} SNe (z_HD > 0.01), z up to {float(SN_ZHD.max()):.2f}")


def chi2_sn(om, w0, wa):
    # χ² with the absolute magnitude marginalised analytically (radiation negligible at z < 2.3)
    a = 1.0 / (1.0 + _ZG)
    inv_e = 1.0 / jnp.sqrt(e2(a, om, 0.0, w0, wa))
    dc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * jnp.diff(_ZG))])
    dl = (1.0 + SN_ZHEL) * jnp.interp(SN_ZHD, _ZG, dc)
    r = SN_M - 5.0 * jnp.log10(dl)
    return r @ SN_ICOV @ r - (SN_A @ r) ** 2 / SN_B


# ---- checks ----
from scipy.optimize import minimize

print("CMB at Planck 2018 ΛCDM (Ω_m 0.3153, h 0.6736, ω_b 0.02237):  R, l_A, ω_b =",
      np.round(np.array(cmb_obs(0.3153, 0.6736, 0.02237, -1.0, 0.0)), 4))
f = jax.jit(lambda p: chi2_cmb(p[0], p[1], p[2], -1.0, 0.0))
r = minimize(lambda p: float(f(jnp.array(p))), [0.3, 0.7, 0.0224], method="Nelder-Mead",
             bounds=[(0.1, 0.6), (0.5, 0.9), (0.018, 0.026)], options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
print(f"CMB-only ΛCDM best fit: Ω_m {r.x[0]:.4f}, h {r.x[1]:.4f}, ω_b {r.x[2]:.5f}  (χ² {r.fun:.2g})")
g = jax.jit(lambda om: chi2_sn(om, -1.0, 0.0))
oms = np.linspace(0.1, 0.6, 501); c2 = np.array([float(g(o)) for o in oms])
print(f"Pantheon+ ΛCDM best fit: Ω_m {oms[c2.argmin()]:.3f}  (χ²_min {c2.min():.1f} for {int(sel.sum())} SNe)")
""")

md(r"""
## 6. Combined sampling

For each cosmology and each combination in `PROBES`, NUTS samples Ω_m, the
dark-energy parameters and (with the CMB) h and ω_b. The priors are flat on
the same ranges as the cluster chains: Ω_m ∈ [0.05, 1], w ∈ [−3, 0], and
w0 ∈ [−3, 1], wa ∈ [−3, 2] with w0 + wa < 0. wa is sampled as a fraction of
its allowed range with the Jacobian included, as in the cluster notebook.
The SL term is the log of the combined grid density, interpolated linearly.

SL alone is not re-sampled: it comes straight from the grid (section 4).
A combination SL × X is first tried by **importance weighting**: X's chain
weighted by the SL density at each sample, then resampled. This targets the
same posterior and works whenever SL is broad compared with X (for example
SL × CMB × SN). NUTS on the interpolated grid is used only when the weights
keep fewer than `IS_MIN_ESS` effective samples. In that case X is so much
broader than SL that its chain covers SL's region poorly.
Every chain is cached in `results/combined/` under a fingerprint of the
cluster chains and these settings.
""")

code(r"""
def sl_logp_fn(k):
    lp = jnp.array(SL_LOGP[k])
    c0 = [float(c[0]) for c in GRID[k]["centers"]]
    dx = [float(c[1] - c[0]) for c in GRID[k]["centers"]]

    def f(*xs):
        coords = [(x - a) / d for x, a, d in zip(xs, c0, dx)]
        return jax.scipy.ndimage.map_coordinates(lp, coords, order=1, mode="nearest")
    return f


def make_model(k, probes):
    use = set(probes.split("+"))
    sl = sl_logp_fn(k) if "SL" in use else None

    def model():
        om = numpyro.sample("omega", dist.Uniform(*DOMAIN[k]["omega"]))
        if k == "lcdm":
            w0, wa = -1.0, 0.0
        else:
            w0 = numpyro.sample("weos", dist.Uniform(*DOMAIN[k]["weos"]))
            wa = 0.0
        if k == "w0wa":
            lo, hi = DOMAIN[k]["wa"]
            top = jnp.minimum(hi, W0WA_CUT - w0)
            u = numpyro.sample("u_wa", dist.Uniform(0.0, 1.0))
            wa = numpyro.deterministic("wa", lo + u * (top - lo))
            numpyro.factor("jacobian", jnp.log(top - lo))           # flat in (w0, wa)
        if "CMB" in use:
            h = numpyro.sample("h", dist.Uniform(0.4, 1.0))
            obh2 = numpyro.sample("ombh2", dist.Uniform(0.005, 0.04))
            numpyro.factor("cmb", -0.5 * chi2_cmb(om, h, obh2, w0, wa))
        if "SN" in use:
            numpyro.factor("sn", -0.5 * chi2_sn(om, w0, wa))
        if sl is not None:
            numpyro.factor("sl", sl(*[v for v, p in zip((om, w0, wa), PARAMS["w0wa"]) if p in PARAMS[k]]))
    return model


def init_values(k, probes):
    v = {"omega": 0.3}
    if k != "lcdm":
        v["weos"] = -1.0
    if k == "w0wa":
        v["u_wa"] = 3.0 / 5.0                                        # wa = 0 at w0 = -1
    if "CMB" in probes:
        v.update(h=0.68, ombh2=0.0224)
    return v


SETTINGS = dict(grid=GRID_N, bw=BW_SCALE, floor=FLOOR, nuts=NUTS_KW, cut=W0WA_CUT, wmax=W_MAX_WCDM,
                sn="pantheon+ zHD>0.01 m_b_corr STAT+SYS", cmb="chen2019 lcdm R lA ombh2 g1=0.0783 n1025", v=2)
FINGERPRINT = hashlib.sha1(fp.hexdigest().encode() + json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()
OUT_PARAMS = ["omega", "weos", "wa", "h", "ombh2"]


def run_combo(k, probes):
    path = os.path.join(RUN_DIR, f"chain_{k}_{probes.replace('+', 'x')}.npz")
    if os.path.exists(path):
        d = np.load(path)
        if str(d["fingerprint"]) == FINGERPRINT:
            return {n: d[n] for n in d.files if n in OUT_PARAMS}, str(d["diag"])
        print(f"  {os.path.basename(path)}: inputs changed, re-running")
    t0 = time.time()
    mcmc = MCMC(NUTS(make_model(k, probes), init_strategy=init_to_value(values=init_values(k, probes)),
                     target_accept_prob=0.9, dense_mass=True),
                chain_method="parallel", progress_bar=False, **NUTS_KW)
    mcmc.run(jax.random.PRNGKey(0))
    s = mcmc.get_samples(group_by_chain=True)
    div = int(mcmc.get_extra_fields()["diverging"].sum())
    summ = np_summary({n: np.asarray(v) for n, v in s.items() if n in OUT_PARAMS})
    diag = (f"{time.time() - t0:5.0f} s, divergences {div}, max R-hat "
            f"{max(float(v['r_hat']) for v in summ.values()):.3f}, min ESS {min(float(v['n_eff']) for v in summ.values()):.0f}")
    out = {n: np.asarray(v).reshape(-1) for n, v in s.items() if n in OUT_PARAMS}
    np.savez(path, fingerprint=FINGERPRINT, diag=diag, clusters=np.array(CLUSTERS), **out)
    return out, diag


from scipy.interpolate import RegularGridInterpolator


def reweight_by_sl(k, base, min_ess=IS_MIN_ESS, seed=0):
    # SL × X by importance weighting X's chain with the SL grid density: the same target as sampling SL × X,
    # without NUTS crossing the grid's linear-interpolation kinks.  Returns (samples, weight ESS) or (None, ESS)
    s = SAMPLES[(k, base)]
    f = RegularGridInterpolator(GRID[k]["centers"], SL_LOGP[k], bounds_error=False, fill_value=None)
    lw = f(np.stack([s[p] for p in PARAMS[k]], axis=1))
    w = np.exp(lw - lw.max()); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    if ess < min_ess:
        return None, ess
    idx = np.random.default_rng(seed).choice(len(w), size=len(w), p=w)
    return {n: v[idx] for n, v in s.items()}, ess


SAMPLES, HOW = {}, {}
for k in NUTS_FILE:
    SAMPLES[(k, "SL")] = dict(zip(PARAMS[k], grid_samples(k, SL[k]).T))
    for probes in PROBES:
        base = probes.replace("SL+", "")
        if probes.startswith("SL+") and (k, base) in SAMPLES:
            s, ess = reweight_by_sl(k, base)
            if s is not None:
                SAMPLES[(k, probes)] = s
                HOW[(k, probes)] = f"{base} chain reweighted by SL (weight ESS {ess:.0f})"
                print(f"{COSMO[k]:14s} {probes:10s} {HOW[(k, probes)]}")
                continue
            print(f"{COSMO[k]:14s} {probes:10s} reweighting {base} has ESS {ess:.0f} < {IS_MIN_ESS}: NUTS")
        SAMPLES[(k, probes)], diag = run_combo(k, probes)
        HOW[(k, probes)] = "NUTS: " + diag
        print(f"{COSMO[k]:14s} {probes:10s} {diag}")
""")

md(r"""
## 7. Results

The table follows Caminha et al. Table 2: medians and 68% intervals, and δ,
the 68% width without SL divided by the width with SL (δ > 1: SL helps). The
SL row comes from the grid. The list below the table says how each SL × X
row was obtained.
""")

code(r"""
def q68(v):
    return np.percentile(v, [16, 50, 84])


def cell(v):
    lo, m, hi = q68(v)
    return f"{m:7.3f} +{hi - m:.3f} -{m - lo:.3f}"


ROWS = ["SL", "SN", "CMB", "SL+SN", "SL+CMB", "CMB+SN", "SL+CMB+SN"]
BASE = {"SL+SN": "SN", "SL+CMB": "CMB", "SL+CMB+SN": "CMB+SN"}
lines = [f"Strong lensing: {len(CLUSTERS)} clusters ({', '.join(CLUSTERS)})",
         "SN: Pantheon+ (z_HD > 0.01, M marginalised); CMB: Planck 2018 distance priors (Chen+2019)", ""]
for k in NUTS_FILE:
    head = f"{COSMO[k]:14s}" + "".join(f"{TXT[p]:>24s}{'δ':>6s}" for p in PARAMS[k])
    lines += [head, "-" * len(head)]
    for row in ROWS:
        s = SAMPLES[(k, row)]
        txt = f"{row.replace('+', '×'):14s}"
        for p in PARAMS[k]:
            d = ""
            if row in BASE:
                w0 = np.subtract(*q68(SAMPLES[(k, BASE[row])][p])[[2, 0]])
                w1 = np.subtract(*q68(s[p])[[2, 0]])
                d = f"{w0 / w1:6.2f}"
            txt += f"{cell(s[p]):>24s}{d:>6s}"
        lines.append(txt)
    lines.append("")
lines.append("How each combination was sampled:")
lines += [f"  {COSMO[k]:14s} {pr:10s} {h}" for (k, pr), h in HOW.items()]
save_txt("cosmology_combined_summary.txt", "\n".join(lines))
""")

md(r"""
### Posteriors by probe

68% and 95% regions for each probe and combination, as in Caminha et al.
Figs. 2–4 (right panels). The left grid shows the full prior range; the right
grid zooms in on the combinations that include the CMB.
""")

code(r"""
import corner

PROBE_STYLE = {"SL": ("0.35", "Strong lensing"), "SN": ("tab:orange", "SN (Pantheon+)"),
               "CMB": ("tab:green", "CMB (Planck distance priors)"), "SL+SN": ("tab:brown", "SL × SN"),
               "SL+CMB": ("tab:blue", "SL × CMB"), "CMB+SN": ("tab:purple", "CMB × SN"),
               "SL+CMB+SN": ("tab:red", "SL × CMB × SN")}


def shade(color, a=(0.15, 0.4)):
    import matplotlib.colors as mc
    r, g, b, _ = mc.to_rgba(color)
    return [(r, g, b, 0.0), (r, g, b, a[0]), (r, g, b, a[1])]


def triangle(k, probes, ranges, fname, title):
    ps = PARAMS[k]; K = len(ps)
    if K == 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for pr in probes:
            col, lab = PROBE_STYLE[pr]
            ax.hist(SAMPLES[(k, pr)]["omega"], bins=120, range=ranges["omega"], density=True,
                    histtype="step", color=col, lw=1.6, label=lab)
        ax.axvline(REF["omega"], color="0.5", ls=":", lw=0.8)
        ax.set_xlabel(LABEL["omega"], fontsize=13); ax.set_yticks([]); ax.set_xlim(ranges["omega"])
        ax.legend(fontsize=9); ax.set_title(title)
        show(fname)
        return
    fig, axs = plt.subplots(K, K, figsize=(3.3 * K, 3.3 * K))
    for i, pi in enumerate(ps):
        for j, pj in enumerate(ps):
            ax = axs[i, j]
            if j > i:
                ax.axis("off"); continue
            for pr in probes:
                col, _ = PROBE_STYLE[pr]
                s = SAMPLES[(k, pr)]
                if i == j:
                    ax.hist(s[pi], bins=60, range=ranges[pi], density=True, histtype="step", color=col, lw=1.4)
                else:
                    corner.hist2d(s[pj], s[pi], ax=ax, range=[ranges[pj], ranges[pi]], bins=60, smooth=1.2,
                                  color=col, levels=(0.68, 0.95), plot_datapoints=False, plot_density=False,
                                  fill_contours=True, contourf_kwargs=dict(colors=shade(col)),
                                  contour_kwargs=dict(colors=[col], linewidths=1.2))
            ax.axvline(REF[pj], color="0.5", ls=":", lw=0.8)
            ax.set_xlim(ranges[pj])
            if i == j:
                ax.set_yticks([])
            else:
                ax.axhline(REF[pi], color="0.5", ls=":", lw=0.8); ax.set_ylim(ranges[pi])
            if i == K - 1:
                ax.set_xlabel(LABEL[pj], fontsize=13)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABEL[pi], fontsize=13)
            elif i != j:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], color=PROBE_STYLE[p][0], lw=2) for p in probes]
    fig.legend(handles, [PROBE_STYLE[p][1] for p in probes], loc="upper right", bbox_to_anchor=(0.97, 0.97),
               fontsize=10, title=title)
    plt.tight_layout(); show(fname)


def zoom_range(k, probes, pad=1.5):
    out = {}
    for p in PARAMS[k]:
        lo = min(np.percentile(SAMPLES[(k, pr)][p], 0.5) for pr in probes)
        hi = max(np.percentile(SAMPLES[(k, pr)][p], 99.5) for pr in probes)
        m = 0.5 * (lo + hi); h = 0.5 * (hi - lo) * pad
        out[p] = (max(m - h, DOMAIN[k][p][0]), min(m + h, DOMAIN[k][p][1]))
    return out


for k in NUTS_FILE:
    full = dict(DOMAIN[k])
    triangle(k, ["SL", "SN", "CMB", "SL+CMB", "SL+CMB+SN"], full, f"probes_{k}", COSMO[k])
    zprobes = ["CMB", "SL+CMB", "CMB+SN", "SL+CMB+SN"]
    triangle(k, zprobes, zoom_range(k, zprobes), f"probes_{k}_zoom", COSMO[k] + " (zoom)")
""")

md(r"""
## Notes

* **Reading the SL result.** The grid product is only as good as each
  cluster's posterior. If the leave-one-out table shows one cluster in
  tension, look at that cluster's mass model (`model_choice.json`, its
  `cosmology_corner.png`, the posterior-predictive check) before trusting the
  combination. A cosmology that moves with the mass model was set by the
  mass model.
* **Prior dependence.** Caminha et al. use Ω_m ∈ [0, 1], w ∈ [−3, 0] and
  wa ∈ [−3, 3], removing wa > −w0. The cluster chains here use DESI's ranges
  (Ω_m ∈ [0.05, 1], wa ∈ [−3, 2], w0 + wa < 0). SL-only intervals that reach
  a prior edge depend on these ranges.
* **Subsets.** Set `GLAFIC_CLUSTERS=as1063,mj0329` (comma-separated) to
  combine a subset. It writes to `results/combined_as1063_mj0329/`.
* **Kernel settings.** `GRID_N`, `BW_SCALE` and `FLOOR` control the grid
  densities. The check in section 3 compares them with the chains. If the
  combined SL result moves with `FLOOR`, the clusters disagree.
* **BAO.** Caminha et al. also add BAO. A BAO likelihood (e.g. DESI DR2
  D_M/r_d, D_H/r_d) can be added in section 5 in the same way; it needs r_d,
  which the CMB block's h and ω_b provide.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(HERE, "cluster_cosmo_combined.ipynb")
nbf.write(nb, out)
print("wrote", out)
