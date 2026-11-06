"""Readers for glafic input files and prior files (init.c)."""

import os
import shlex
import warnings

import numpy as np

from .settings import Settings
from .model import Model

PRIMARY = ["omega", "lambda", "weos", "hubble", "prefix", "xmin", "ymin", "xmax", "ymax",
           "pix_ext", "pix_poi", "maxlev"]

DATA_COMMANDS = {"readobs_point", "readobs_extend", "readnoise_extend", "readpsf", "parprior",
                 "reset_obs_point"}


def _tokens(line):
    line = line.split("#")[0]
    return line.split()


def read_input(fname, run_commands=True, verbose=False):
    """Build a :class:`Model` from a glafic input file.

    Data-loading commands in ``start_command`` (readobs_point, readobs_extend,
    readnoise_extend, readpsf, parprior, reset_obs_point) are executed; other
    commands (optimize, mcmc, writeimage, ...) are returned in
    ``model.commands`` so the caller can run the JAX equivalents.
    """
    base = os.path.dirname(os.path.abspath(fname))
    lines = open(fname).read().splitlines()
    s = Settings()
    k = 0
    # primary + secondary parameters up to startup
    while k < len(lines):
        t = _tokens(lines[k])
        if t and t[0] == "startup":
            break
        if t:
            if t[0] == "zl":
                raise ValueError("wrong input file (primary parameter zl not used in ver. 2)")
            try:
                s = s.set(t[0], t[1])
            except KeyError:
                warnings.warn(f"glafic_jax: ignoring unsupported parameter '{t[0]}'")
        k += 1
    if k >= len(lines):
        raise ValueError("no startup found")
    t = _tokens(lines[k])
    nl, ne, npo = int(t[1]), int(t[2]), int(t[3])
    m = Model(s)
    m.startup_setnum(nl, ne, npo)
    il = ie = ip = 0
    order = []
    k += 1
    while True:
        t = _tokens(lines[k])
        k += 1
        if not t:
            continue
        if t[0] == "end_startup":
            break
        if t[0] == "lens":
            if t[1] == "gals":
                m.read_gals(os.path.join(base, s.galfile))
            m.set_lens(il + 1, t[1], *[float(v) for v in t[2:10]])
            il += 1
            order.append("lens")
        elif t[0] == "extend":
            m.set_extend(ie + 1, t[1], *[float(v) for v in t[2:10]])
            ie += 1
            order.append("extend")
        elif t[0] == "point":
            m.set_point(ip + 1, *[float(v) for v in t[1:4]])
            ip += 1
            order.append("point")
        elif t[0] == "psf":
            m.set_psf(*[float(v) for v in t[1:10]])
            order.append("psf")
    if (il, ie, ip) != (nl, ne, npo):
        raise ValueError("startup failed (number mismatch)")
    # setopt
    while k < len(lines):
        t = _tokens(lines[k])
        k += 1
        if t and t[0] == "start_setopt":
            counters = {"lens": 0, "extend": 0, "point": 0}
            n = 0
            while True:
                t = _tokens(lines[k])
                k += 1
                if not t:
                    continue
                if t[0] == "end_setopt":
                    break
                if n >= len(order):   # glafic silently ignores extra lines
                    continue
                kind = order[n]
                n += 1
                f = [int(v) for v in t]
                if kind == "lens":
                    counters["lens"] += 1
                    m.setopt_lens(counters["lens"], *f)
                elif kind == "extend":
                    counters["extend"] += 1
                    m.setopt_extend(counters["extend"], *f)
                elif kind == "point":
                    counters["point"] += 1
                    m.setopt_point(counters["point"], *f)
                else:
                    m.setopt_psf(*f)
            break
    m.model_init()
    # commands
    cmds = []
    while k < len(lines):
        t = _tokens(lines[k])
        k += 1
        if t and t[0] == "start_command":
            break
    while k < len(lines):
        t = _tokens(lines[k])
        k += 1
        if not t:
            continue
        if t[0] == "quit":
            break
        if run_commands and t[0] in DATA_COMMANDS:
            run_command(m, t, base)
        else:
            cmds.append(t)
    m.commands = cmds
    m.model_init()
    return m


def run_command(m, t, base="."):
    p = lambda f: os.path.join(base, f)
    c = t[0]
    if c == "readobs_point":
        m.readobs_point(p(t[1]))
    elif c == "readobs_extend":
        m.readobs_extend(p(t[1]), p(t[2]) if len(t) > 2 else None)
    elif c == "readnoise_extend":
        m.readnoise_extend(p(t[1]))
    elif c == "readpsf":
        m.readpsf(p(t[1]))
    elif c == "parprior":
        read_prior(m, p(t[1]))
    elif c == "reset_obs_point":
        m.reset_obs_point(int(t[1]), int(t[2]), int(t[3]), float(t[4]))


_COSMO = {"omega": 0, "lambda": 1, "weos": 2, "w0": 2, "hubble": 3, "wa": 4}


def read_prior(m, fname):
    """parprior file: range / gauss / match / relrange / obsext lines."""
    pl, pe, pp, ps, pc = m.prior_lens, m.prior_ext, m.prior_point, m.prior_psf, m.prior_cosmo
    blocks = {"lens": pl, "extend": pe, "point": pp}
    for line in open(fname):
        t = _tokens(line)
        if len(t) < 2:
            continue
        typ, key = t[0], t[1]
        if typ == "range":
            if key in blocks:
                i, j, lo, hi = int(t[2]) - 1, int(t[3]) - 1, float(t[4]), float(t[5])
                blocks[key]["min"][i, j] = lo
                blocks[key]["max"][i, j] = hi
            elif key == "psf":
                j = int(t[2]) - 1
                ps["min"][j], ps["max"][j] = float(t[3]), float(t[4])
            elif key in _COSMO:
                c = _COSMO[key]
                pc["min"][c], pc["max"][c] = float(t[2]), float(t[3])
        elif typ == "gauss":
            if key in blocks:
                i, j = int(t[2]) - 1, int(t[3]) - 1
                blocks[key]["med"][i, j] = float(t[4])
                blocks[key]["sig"][i, j] = float(t[5])
            elif key == "psf":
                j = int(t[2]) - 1
                ps["med"][j], ps["sig"][j] = float(t[3]), float(t[4])
            elif key in _COSMO:
                c = _COSMO[key]
                pc["med"][c], pc["sig"][c] = float(t[2]), float(t[3])
        elif typ == "match":
            if key == "psf":
                j, jj = int(t[2]) - 1, int(t[3]) - 1
                ps["raj"][j], ps["rat"][j], ps["ras"][j] = jj, float(t[4]), float(t[5])
            else:
                i, j, ii, jj = (int(v) - 1 for v in t[2:6])
                rat, ras = float(t[6]), float(t[7])
                if key in ("lens", "extend", "point"):
                    b = blocks[key]
                    b["rai"][i, j], b["raj"][i, j], b["rat"][i, j], b["ras"][i, j] = ii, jj, rat, ras
                elif key == "extlens":
                    pe["lrai"][i, j], pe["lraj"][i, j], pe["lrat"][i, j], pe["lras"][i, j] = ii, jj, rat, ras
                elif key == "poilens":
                    pp["lrai"][i, j], pp["lraj"][i, j], pp["lrat"][i, j], pp["lras"][i, j] = ii, jj, rat, ras
        elif typ == "relrange":
            if key == "psf":
                j, jj = int(t[2]) - 1, int(t[3]) - 1
                ps["reraj"][j], ps["reral"][j], ps["rerah"][j] = jj, float(t[4]), float(t[5])
            elif key in blocks:
                i, j, ii, jj = (int(v) - 1 for v in t[2:6])
                b = blocks[key]
                b["rerai"][i, j], b["reraj"][i, j] = ii, jj
                b["reral"][i, j], b["rerah"][i, j] = float(t[6]), float(t[7])
        elif typ == "obsext":
            pass  # only an initial guess for glafic's optimiser
