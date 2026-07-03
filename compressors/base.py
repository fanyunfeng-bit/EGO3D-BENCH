"""Model-agnostic visual-token compression interface.

A `TokenCompressor` decides, for ONE image, which visual tokens to keep. It
operates at the granularity of the tokens that actually enter the LLM (e.g. for
InternVL3 these are the 256 post-pixel-shuffle tokens per tile), so it is
independent of any particular VLM. Model-specific plumbing (extracting the
importance cue, mapping ViT patches to LLM tokens, rebuilding the prompt) lives
in the per-model adapter, not here.

Adding a new compression method = adding a subclass and registering it in
`compressors/__init__.py`; the runner and adapter do not change.
"""

from abc import ABC, abstractmethod

import torch


class TokenCompressor(ABC):
    name: str = "base"
    # Whether this method needs the ViT importance cue. Methods that don't
    # (e.g. random) let the adapter skip the attention extraction entirely.
    needs_importance: bool = True

    @abstractmethod
    def select(self, importance, features, keep, seed=None) -> torch.Tensor:
        """Return the indices of the tokens to KEEP for a single image.

        Args:
            importance: (N,) per-token importance score (higher = more important),
                        or None for methods with needs_importance = False.
            features:   (N, C) per-token features, used for similarity/diversity.
            keep:       number of tokens to retain (clamped to [1, N]).
            seed:       optional int for reproducible randomness. The adapter
                        derives it deterministically per (experiment, sample,
                        view) so repeats are identical and resume-safe.

        Returns:
            (keep,) LongTensor of indices into [0, N), sorted ascending so the
            original spatial/sequence order of the kept tokens is preserved.
        """
        raise NotImplementedError
