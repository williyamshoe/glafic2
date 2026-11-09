"""glafic_jax: a differentiable (JAX) port of the glafic gravitational lensing code.

Double precision is enabled on import because the lens equation solver and
cosmological distances need it.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .settings import Settings, default_settings  # noqa: E402
from . import cosmology, lens_models, lensing  # noqa: E402
from .lensing import LensSystem  # noqa: E402

__version__ = "2.1.15+jax0"
from .model import Model  # noqa: E402
from .io import read_input, read_prior  # noqa: E402
from . import point, extend, imagefinder  # noqa: E402
from .inference import optimize, run_nuts, summary  # noqa: E402
from .pipeline import Pipeline, diagnostics, angle_bounds  # noqa: E402
