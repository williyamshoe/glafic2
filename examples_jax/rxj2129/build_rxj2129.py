#!/usr/bin/env python
"""RXJ2129 wrapper around ../cluster_cosmo/build_cluster.py (kept for rxj2129_hmc.ipynb).

Writes rxj2129.input, rxj2129_obs.dat, rxj2129_galfile.dat and rxj2129_prior.dat
from example_data/rj2129/.  See build_cluster.write_all for the options.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cluster_cosmo"))
import build_cluster as _bc  # noqa: E402

NAME = "rj2129"
ZL = 0.234
radec_to_xy = _bc.radec_to_xy


def load():
    """((ra0, dec0), gals, systems) for RXJ2129."""
    c = _bc.load(NAME)
    return c["center"], c["gals"], c["systems"]


def write_all(outdir=HERE, **kw):
    return _bc.write_all(NAME, outdir, prefix="rxj2129", **kw)


if __name__ == "__main__":
    s = write_all()
    print(f"{len(s)} systems, {sum(len(v['xy']) for v in s.values())} images written to {HERE}")
