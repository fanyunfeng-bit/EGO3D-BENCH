"""Pluggable visual-token compressors.

Register new methods here; the runner selects one by name via
`--compress_method`. `none` (or None) means no compression (baseline).
"""

from .base import TokenCompressor
from .vispruner import VisPrunerCompressor
from .random_prune import RandomPruneCompressor

_REGISTRY = {
    "vispruner": VisPrunerCompressor,
    "random": RandomPruneCompressor,
}


def available_methods():
    return ["none"] + sorted(_REGISTRY)


def build_compressor(name, **kwargs):
    if name in (None, "none"):
        return None
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown compress_method '{name}'. Available: {available_methods()}"
        )
    return _REGISTRY[name](**kwargs)
