"""VisPruner: visual-cue token pruning (ICCV 2025, arXiv:2412.01818).

Faithful port of the selection logic in the official repo
(`llava/model/llava_arch.py::encode_images`), specialised to a single image:

  1. Important tokens  - top `keep * important_ratio` tokens by visual attention
     importance (here the ViT CLS->patch attention, supplied by the adapter).
  2. Diverse tokens    - from the remaining tokens, iteratively drop duplicates
     by ToMe-style bipartite soft matching on cosine similarity, until
     `keep - important_token_num` distinct tokens remain.

Retaining diverse tokens alongside the important ones preserves more of the
image's visual information than attention-ranking alone.
"""

import torch

from .base import TokenCompressor


class VisPrunerCompressor(TokenCompressor):
    name = "vispruner"

    def __init__(self, important_ratio: float = 0.5, merge_chunk: int = 8):
        # important_ratio r: fraction of the kept budget spent on important tokens
        # (the rest are diverse tokens). VisPruner default is 0.5.
        self.important_ratio = float(important_ratio)
        # max tokens removed per ToMe iteration (VisPruner uses 8).
        self.merge_chunk = int(merge_chunk)

    @torch.no_grad()
    def select(self, importance, features, keep, seed=None):  # seed unused (deterministic)
        N = importance.shape[0]
        keep = max(1, min(int(keep), N))

        n_important = int(keep * self.important_ratio)
        n_diverse = keep - n_important

        # --- important tokens: top-k by attention importance ---
        order = importance.argsort(descending=True)          # (N,)
        important_idx = order[:n_important]                   # (n_important,)
        residual_idx = order[n_important:]                    # (N - n_important,)

        # --- diverse tokens: ToMe bipartite matching on the residual set ---
        if n_diverse > 0 and residual_idx.numel() > n_diverse:
            feats = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            while True:
                R = residual_idx.shape[0]
                r = min(self.merge_chunk, R - n_diverse)
                if r <= 0:
                    break
                res = feats[residual_idx]                     # (R, C)
                a, b = res[0::2], res[1::2]                   # even / odd split
                scores = a @ b.transpose(0, 1)               # (|a|, |b|)
                scores = scores.max(dim=-1).values           # (|a|,) max sim to any b
                # drop the r most-similar (redundant) tokens of a, keep the rest
                distinct = scores.argsort(descending=True)[r:]
                residual_idx = torch.cat(
                    [residual_idx[0::2][distinct], residual_idx[1::2]]
                )
            diverse_idx = residual_idx
        elif n_diverse > 0:
            diverse_idx = residual_idx                        # fewer than budget: keep all
        else:
            diverse_idx = residual_idx[:0]                    # none

        selected = torch.cat([important_idx, diverse_idx])
        return selected.sort().values                         # preserve spatial order
