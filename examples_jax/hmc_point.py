#!/usr/bin/env python
"""HMC (NUTS) for the point-source sample (samples/point.input) with glafic_jax.

    python examples_jax/hmc_point.py            # from the repository root

Steps: read the glafic input file -> optimise (source plane, then image
plane, both with exact JAX gradients) -> sample with NUTS.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import glafic_jax as gj  # noqa: E402

model = gj.read_input(os.path.join(HERE, "..", "samples", "point.input"))
model.set_secondary("hvary 0")      # the sample file varies h, which point data barely constrain
model.model_init()
print("free parameters:", model.theta_names)

# 1) optimise: the source-plane chi^2 is a much smoother starting point
model.set_secondary("chi2_splane 1"); model.model_init()
theta, chi2, _ = gj.optimize(model)
print(f"source-plane best fit: chi2 = {chi2:.4f}")
model.set_secondary("chi2_splane 0"); model.model_init()
theta, chi2, _ = gj.optimize(model, theta)
print(f"image-plane  best fit: chi2 = {chi2:.4f}")

# 2) NUTS.  point_mode="glafic" finds images with glafic's algorithm on every
#    step (exact glafic chi^2, checks the image number); "tracked" is pure JAX.
res = gj.run_nuts(model, theta, num_warmup=500, num_samples=1000, point_mode="tracked")
print(gj.summary(res))
np.savez("hmc_point_chain.npz", theta=res["theta"], names=res["names"], log_prob=res["log_prob"])
