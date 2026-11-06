"""Extract the CSE coefficient tables from app_ell.c / app_cnfw.c into an npz.

Run from the repository root:  python python/glafic_jax/tools/extract_cse_tables.py
"""
import os
import re
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
out = os.path.join(os.path.dirname(__file__), "..", "data", "cse_tables.npz")


def grab(src, name):
    m = re.search(r"%s\s*(\[[^\]]*\])*\s*=\s*\{(.*?)\};" % re.escape(name), src, re.S)
    body = m.group(2).replace("{", " ").replace("}", " ")
    return np.array([float(v) for v in re.split(r"[,\s]+", body) if v.strip()])


ell = open(os.path.join(root, "app_ell.c")).read()
cnfw = open(os.path.join(root, "app_cnfw.c")).read()
nfw = grab(ell, "par_cse_nfw")
hern = grab(ell, "par_cse_hern")
cn = grab(cnfw, "par_cse_cnfw").reshape(201, 88)
assert nfw.size == 88 and hern.size == 82
np.savez(out, nfw=nfw.reshape(44, 2), hern=hern.reshape(41, 2), cnfw=cn.reshape(201, 44, 2))
print("wrote", out, nfw.shape, hern.shape, cn.shape)
