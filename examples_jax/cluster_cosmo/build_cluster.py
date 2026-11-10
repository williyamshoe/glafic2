#!/usr/bin/env python
"""Convert a cluster catalogue in example_data/<name>/ into glafic input files.

Expected files in ``example_data/<name>/`` (the glafic public-model format;
the highest version ``vN`` is used when several exist):

  glafic_<name>_vN_mulimage.txt   # ID RA Dec err_pos[arcsec] redshift err_redshift
                                  #   ID = "<system>.<image>"; err_redshift > 0 marks a
                                  #   photometric redshift, which is sampled
  glafic_<name>_vN_memgal.txt     # RA Dec L/L_* e PA[deg]  (the brightest member is the BCG)
  glafic_<name>_vN_readme.txt     # optional: "# zl = ...", cosmology, "# chi2 = .. , dof = .. , rms[arcsec] = .."

The lens redshift is read from the readme; pass ``zl=`` when there is none.

Writes (in ``outdir``, file names start with ``prefix``, default ``name``):
  <prefix>.input        glafic input file (lens model, cosmology flags, setopt)
  <prefix>_obs.dat      multiple images in glafic readobs_point format
  <prefix>_galfile.dat  cluster members (x, y, L/L*, e, PA) for the 'gals' model
  <prefix>_prior.dat    parameter ranges (parprior format)

Coordinates are arcsec relative to the BCG with glafic's convention
x = (RA0 - RA) cos(Dec0) * 3600 (positive to the west), y = (Dec - Dec0) * 3600.

    python build_cluster.py                 # list the available clusters
    python build_cluster.py mj0429 [outdir] # write the glafic files
"""
import glob
import math
import os
import re
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "example_data"))


def available(data=DATA):
    """Names of the clusters in ``data`` that have a multiple-image catalogue."""
    return sorted(d for d in os.listdir(data)
                  if glob.glob(os.path.join(data, d, f"glafic_{d}_*_mulimage.txt")))


def _version(path):
    m = re.search(r"_v(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_files(name, data=DATA):
    """{'mulimage', 'memgal', 'readme'} -> path (readme may be None) for the latest version."""
    d = name if os.path.isdir(name) else os.path.join(data, name)
    base = os.path.basename(os.path.normpath(d))
    out = {}
    for kind in ("mulimage", "memgal", "readme"):
        hits = sorted(glob.glob(os.path.join(d, f"glafic_{base}_*_{kind}.txt")), key=_version)
        if not hits and kind != "readme":
            have = ", ".join(available(data)) or "none"
            raise FileNotFoundError(f"no glafic_{base}_*_{kind}.txt in {d} (available clusters: {have})")
        out[kind] = hits[-1] if hits else None
    return out


def read_readme(path):
    """Lens redshift, reference cosmology and published fit from a readme (missing keys are absent)."""
    info = {}
    if path is None:
        return info
    txt = open(path).read()
    for key in ("omega", "lambda", "hubble", "weos", "zl", "chi2", "dof"):
        m = re.search(rf"\b{key}\s*=\s*([-+0-9.eE]+)", txt)
        if m:
            info[key] = float(m.group(1))
    m = re.search(r"rms\[arcsec\]\s*=\s*([-+0-9.eE]+)", txt)
    if m:
        info["rms"] = float(m.group(1))
    m = re.search(r"MODEL_REF\s+(\S+)", txt)
    if m:
        info["model_ref"] = m.group(1)
    return info


def radec_to_xy(ra, dec, ra0, dec0):
    return (ra0 - ra) * 3600.0 * np.cos(np.deg2rad(dec0)), (dec - dec0) * 3600.0


def _system_id(s):
    sid = s.split(".")[0]
    return int(sid) if sid.isdigit() else sid


def load(name, zl=None, data=DATA):
    """Read a cluster.  Returns a dict with
    name, zl, center (RA, Dec of the BCG), gals (N x 5: x, y, L, e, PA; BCG first),
    systems (OrderedDict id -> {zs, zerr, err, errs, ids, xy}), ref (readme info), files."""
    files = find_files(name, data)
    ref = read_readme(files["readme"])
    zl = ref.get("zl") if zl is None else zl
    if zl is None:
        raise ValueError(f"{name}: no 'zl = ...' in the readme; pass zl= explicitly")
    gal = np.atleast_2d(np.loadtxt(files["memgal"]))
    order = np.r_[np.argmax(gal[:, 2]), np.delete(np.arange(len(gal)), np.argmax(gal[:, 2]))]
    gal = gal[order]                                   # BCG (brightest member) first
    ra0, dec0 = gal[0, 0], gal[0, 1]
    gx, gy = radec_to_xy(gal[:, 0], gal[:, 1], ra0, dec0)
    gals = np.column_stack([gx, gy, gal[:, 2], gal[:, 3], gal[:, 4]])
    systems = OrderedDict()
    for line in open(files["mulimage"]):
        tok = line.split("#")[0].split()
        if not tok:
            continue
        if len(tok) < 6:
            raise ValueError(f"{files['mulimage']}: expected 'ID RA Dec err z zerr', got {line!r}")
        iid, (ra, dec, err, zs, zerr) = tok[0], map(float, tok[1:6])
        x, y = radec_to_xy(ra, dec, ra0, dec0)
        s = systems.setdefault(_system_id(iid), dict(zs=zs, zerr=zerr, err=err, errs=[], ids=[], xy=[]))
        if abs(s["zs"] - zs) > 1e-6 or abs(s["zerr"] - zerr) > 1e-6:
            raise ValueError(f"{name}: image {iid} has a different redshift from the rest of its system")
        s["errs"].append(err); s["ids"].append(iid); s["xy"].append((float(x), float(y)))
    for sid, s in systems.items():
        if len(s["xy"]) < 2:
            raise ValueError(f"{name}: system {sid} has a single image")
        if s["zs"] <= zl:
            raise ValueError(f"{name}: system {sid} has z_s = {s['zs']} <= z_l = {zl}")
    return dict(name=os.path.basename(os.path.normpath(name)), zl=float(zl), center=(ra0, dec0),
                gals=gals, systems=systems, ref=ref, files=files)


def summary(c):
    """One-paragraph description of a loaded cluster."""
    S = c["systems"].values()
    zs = [s["zs"] for s in S]
    nphot = sum(s["zerr"] > 0 for s in S)
    nimg = sum(len(s["xy"]) for s in S)
    r = np.hypot(*np.concatenate([np.array(s["xy"]) for s in S]).T)
    txt = (f"{c['name']}: z_l = {c['zl']:.3f}; {nimg} images of {len(zs)} systems, "
           f"{min(zs):.2f} < z_s < {max(zs):.2f} ({nphot} photometric); {len(c['gals'])} members; "
           f"images at {r.min():.1f}-{r.max():.1f}\" from the BCG")
    if "chi2" in c["ref"]:
        txt += f"; published fit chi2 = {c['ref']['chi2']:.2f} for {c['ref']['dof']:.0f} dof"
    return txt


def write_all(name, outdir=HERE, prefix=None, zl=None, bcg=False, mpoles=(), shear=True, eta_free=True,
              pos_err=None, parity=None, fov=None, data=DATA):
    """Write the glafic files for cluster ``name``.  Defaults follow the glafic
    recipe of Oguri et al. (VENUS, 2026, Sec. 2.3): elliptical NFW halo (all 6
    parameters free, 1 < c < 40, e < 0.8), pseudo-Jaffe members with
    sigma ~ L^1/4 and r_trun ~ L^eta (100 < sigma* < 400 km/s,
    1" < r_trun* < 100", 0.2 < eta < 1.5), external shear, and optional
    multipoles.  Returns the systems (see :func:`load`).

    pos_err: override the catalogue positional error [arcsec] (the paper
             rescales errors so that the best fit has reduced chi^2 ~ 1).
    bcg    : model the BCG separately (pseudo-Jaffe 'jaffe', e and PA from the
             catalogue, sigma and truncation radius free) and remove it from
             the 'gals' member catalogue.
    mpoles : sequence of (m, n) multipole perturbations ('mpole') centred on
             the BCG, with amplitude and angle free.
    shear  : include the external shear ('pert', amplitude and angle free).
             Lenses are numbered anfw, gals, [pert], [jaffe], [mpole...].
    parity : {image ID: +1 or -1} required parity (sign of 1/mu) of selected
             images, e.g. both members of a merging pair, which must straddle
             a critical curve.  glafic gives a wrong parity a chi^2 of
             chi2pen_parity (a hard wall); other images stay unconstrained.
    fov    : half-width of the glafic grid [arcsec]; default max(60, 2 x the
             largest image distance from the BCG), rounded up to 10".
    """
    c = load(name, zl=zl, data=data)
    parity = {str(k): int(v) for k, v in (parity or {}).items()}
    known = {iid for s in c["systems"].values() for iid in s["ids"]}
    if set(parity) - known:
        raise ValueError(f"parity: unknown image IDs {sorted(set(parity) - known)}")
    if any(v not in (-1, 1) for v in parity.values()):
        raise ValueError("parity values must be +1 or -1")
    prefix = prefix or c["name"]
    zl, gals, systems = c["zl"], c["gals"], c["systems"]
    os.makedirs(outdir, exist_ok=True)
    bcg_row = gals[0]
    if bcg:
        gals = gals[1:]
    if fov is None:
        rmax = max(np.hypot(*np.array(s["xy"]).T).max() for s in systems.values())
        fov = max(60.0, 10.0 * math.ceil(2.0 * rmax / 10.0))
    ra0, dec0 = c["center"]
    np.savetxt(os.path.join(outdir, f"{prefix}_galfile.dat"), gals, fmt="%12.6f %12.6f %10.5f %7.3f %8.2f",
               header="x[arcsec] y[arcsec] L/L* e PA[deg]  (BCG at RA=%.7f Dec=%.7f)" % (ra0, dec0))
    with open(os.path.join(outdir, f"{prefix}_obs.dat"), "w") as f:
        f.write("# id nimg zs zs_err ; then x y flux pos_err flux_err td td_err parity\n")
        for k, (sid, s) in enumerate(systems.items(), start=1):
            f.write(f"{k} {len(s['xy'])} {s['zs']:.4f} {s['zerr']:.4f}   # system {sid}\n")
            for (x, y), e, iid in zip(s["xy"], s["errs"], s["ids"]):
                f.write(f"{x:10.5f} {y:10.5f} 0.0 {pos_err or e:.3f} 0.0 0.0 0.0 {parity.get(iid, 0)}"
                        + (f"   # {iid}\n" if iid in parity else "\n"))
    lines = [f"# glafic input for {c['name']} (generated by build_cluster.py)",
             "omega     0.300000", "lambda    0.700000", "weos     -1.000000", "hubble    0.700000",
             f"prefix    {prefix}",
             f"xmin     {-fov:.1f}", f"ymin     {-fov:.1f}", f"xmax      {fov:.1f}", f"ymax      {fov:.1f}",
             "pix_ext   0.2", "pix_poi   1.0", "maxlev    6",
             f"galfile   {prefix}_galfile.dat",
             "chi2_splane    1", "chi2_checknimg 0",
             "flatfix   1", "ovary     1", "wvary     1", "wavary    1", "hvary     0",
             "", f"startup {2 + int(shear) + int(bcg) + len(mpoles)} 0 {len(systems)}",
             f"lens anfw  {zl} 8.0e14  0.0  0.0  0.4  -20.0  6.0  0.0",
             f"lens gals  {zl} 200.0   0.0  0.0  0.0    0.0 20.0  1.0"]
    if shear:
        lines.append(f"lens pert  {zl} 2.0     0.0  0.0  0.05  30.0  0.0  0.0")
    if bcg:
        lines.append(f"lens jaffe {zl} 300.0   0.0  0.0  {bcg_row[3]:.3f} {bcg_row[4]:.1f}  10.0  0.01")
    for (mm, nn) in mpoles:
        lines.append(f"lens mpole {zl} 2.0     0.0  0.0  0.01  0.0  {mm:.1f}  {nn:.1f}")  # (m, n)
    for s in systems.values():                         # initial source guess: image centroid
        cxy = np.mean(s["xy"], axis=0)
        lines.append(f"point {s['zs']:.4f} {cxy[0]:.4f} {cxy[1]:.4f}")
    lines += ["end_startup", "", "start_setopt",
              "0 1 1 1 1 1 1 0",      # anfw: M, x, y, e, PA, c
              "0 1 0 0 0 0 1 %d" % int(eta_free)]   # gals: sigma*, r_trunc*, eta
    if shear:
        lines.append("0 0 0 0 1 1 0 0")       # pert: gamma, theta
    if bcg:
        lines.append("0 1 0 0 0 0 1 0")       # BCG: sigma, truncation radius
    for _ in mpoles:
        lines.append("0 0 0 0 1 1 0 0")       # multipole: amplitude, angle
    for s in systems.values():
        lines.append(f"{1 if s['zerr'] > 0 else 0} 1 1")
    lines += ["end_setopt", "", "start_command",
              f"readobs_point {prefix}_obs.dat", f"parprior {prefix}_prior.dat", "optimize", "quit", ""]
    open(os.path.join(outdir, f"{prefix}.input"), "w").write("\n".join(lines))
    # Angles get a wide range here (no hard wall); the sampler restricts them to
    # one period around the best fit (period 180 deg for ellipses/shear,
    # 360/m for multipoles) to avoid the exact angle degeneracy.
    ib = 3 + int(shear)                                # BCG lens index (if any)
    nl = 2 + int(shear) + int(bcg)                     # lenses before the multipoles
    prior = ["# parameter ranges (flat priors inside); Oguri et al. VENUS recipe",
             "range lens 1 2 1.0e13 1.0e16", "range lens 1 3 -15.0 15.0", "range lens 1 4 -15.0 15.0",
             "range lens 1 5 0.0 0.8", "range lens 1 6 -720.0 720.0", "range lens 1 7 1.0 40.0",
             "range lens 2 2 100.0 400.0", "range lens 2 7 1.0 100.0", "range lens 2 8 0.2 1.5",
             *(["range lens 3 5 0.0 0.5", "range lens 3 6 -720.0 720.0"] if shear else []),
             *([f"range lens {ib} 2 50.0 800.0", f"range lens {ib} 7 0.1 300.0"] if bcg else []),
             *sum(([f"range lens {nl + 1 + q} 5 0.0 0.5", f"range lens {nl + 1 + q} 6 -720.0 720.0"]
                   for q in range(len(mpoles))), []),
             "range omega 0.05 1.0", "range weos -2.5 0.0", "range wa -10.0 10.0"]
    for k, s in enumerate(systems.values(), start=1):
        if s["zerr"] > 0:
            prior.append(f"range point {k} 1 {max(s['zs'] - 4 * s['zerr'], zl + 0.05):.3f} "
                         f"{s['zs'] + 4 * s['zerr']:.3f}")
    open(os.path.join(outdir, f"{prefix}_prior.dat"), "w").write("\n".join(prior) + "\n")
    return systems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        for n in available():
            print(summary(load(n)))
        sys.exit(0)
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, sys.argv[1])
    s = write_all(sys.argv[1], out)
    print(f"{len(s)} systems, {sum(len(v['xy']) for v in s.values())} images written to {out}")
