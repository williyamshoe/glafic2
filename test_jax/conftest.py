import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
SAMPLES = os.path.join(HERE, "..", "samples")


def c_glafic():
    """Import the compiled C extension if GLAFIC_C_PATH points to it (the
    directory containing the 'glafic' package built with 'make python')."""
    p = os.environ.get("GLAFIC_C_PATH")
    if not p:
        return None
    sys.path.insert(0, p)
    try:
        import glafic
        glafic.version  # noqa
        return glafic
    except Exception:
        return None
    finally:
        sys.path.remove(p)
