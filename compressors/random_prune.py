"""Random visual-token pruning baseline.

Drops a fixed proportion of visual tokens from each view uniformly at random
(no visual cues) -- the natural ablation for VisPruner. The kept count per view
is identical to any other method at the same keep_ratio, so FLOPs/KV/storage
match; only WHICH tokens are kept differs. Any accuracy gap over this baseline
is the value of informed selection.

Determinism: the per-view RNG seed is supplied by the adapter, derived from
(experiment seed, sample index, view index), so the same experiment reproduces
the exact same pruning every run and is unaffected by resume / ordering.
"""

import torch

from .base import TokenCompressor


class RandomPruneCompressor(TokenCompressor):
    name = "random"
    needs_importance = False  # ignores ViT attention / features entirely

    def __init__(self, **kwargs):  # accept & ignore method-specific kwargs
        pass

    @torch.no_grad()
    def select(self, importance, features, keep, seed=None):
        N = features.shape[0]
        keep = max(1, min(int(keep), N))
        g = torch.Generator()  # CPU generator, isolated from global RNG state
        g.manual_seed(int(seed) if seed is not None else 0)
        idx = torch.randperm(N, generator=g)[:keep]
        return idx.sort().values.to(features.device)
