"""InternVL3-specific glue between the model and a model-agnostic TokenCompressor.

Responsibilities (the model-dependent parts VisPruner needs):
  1. Expose the InternViT CLS->patch attention as the importance cue. The ViT
     defaults to flash-attention which discards weights, so we force the feature
     layer (`select_layer`) onto the naive path and stash its softmax attention.
  2. Run the standard InternVL feature path (ViT -> drop CLS -> pixel_shuffle ->
     mlp1) to get the per-tile LLM tokens, and aggregate the patch-level
     importance onto those same tokens by reusing the model's own pixel_shuffle
     (guarantees index consistency with the embeddings).
  3. Per tile, ask the compressor which tokens to keep, and return the pruned
     visual embeddings ready to hand to `model.generate(visual_features=...)`.

This file knows about InternVL; `compressors/vispruner.py` stays model-agnostic.
"""

import numpy as np
import torch


class AttentionCapture:
    """Forces one InternViT layer onto the naive attention path and stashes its
    CLS->patch attention (averaged over heads) on every forward pass."""

    def __init__(self, model):
        vm = model.vision_model
        layers = vm.encoder.layers
        sel = model.select_layer
        self.layer_idx = sel if sel >= 0 else len(layers) + sel  # select_layer=-1 -> last
        self.attn = layers[self.layer_idx].attn
        self.attn.use_flash_attn = False  # naive path materialises attn weights
        self.cls_attn = None
        self.attn._naive_attn = self._naive_attn_with_capture

    def _naive_attn_with_capture(self, x):
        a = self.attn
        B, N, C = x.shape
        qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if a.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = a.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = a.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        attn = ((q * a.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        # CLS (index 0) attention to the patch tokens (1:), averaged over heads.
        self.cls_attn = attn[:, :, 0, 1:].mean(dim=1).detach()  # (B, N-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return a.proj_drop(a.proj(x))


@torch.no_grad()
def compute_visual_features(model, pixel_values, compressor=None, keep_ratio=0.5,
                            capture=None, base_seed=42, sample_id=0):
    """Return (visual_features, tokens_per_tile).

    Baseline (compressor is None): the full per-tile tokens from extract_feature.
    Compressed: each tile pruned to `round(num_image_token * keep_ratio)` tokens
    by the compressor. Methods with needs_importance=True get the CLS->patch
    attention as importance; methods without it (e.g. random) get importance=None
    and a deterministic per-view seed derived from (base_seed, sample_id, view),
    so the same experiment reproduces identical pruning and is resume-safe.
    `visual_features` is (sum_tokens, C), ready for model.generate(visual_features=).
    """
    vit_embeds = model.extract_feature(pixel_values)        # (T_tiles, num_image_token, C)
    n_tiles, n_tok, C = vit_embeds.shape

    if compressor is None:
        return vit_embeds.reshape(-1, C), n_tok

    keep = max(1, round(n_tok * keep_ratio))
    if keep >= n_tok:
        return vit_embeds.reshape(-1, C), n_tok

    need_imp = getattr(compressor, "needs_importance", True)
    imp = None
    if need_imp:
        # patch-level importance (T_tiles, P) -> token-level (T_tiles, num_image_token)
        cls_attn = capture.cls_attn                         # (T_tiles, P) e.g. P=1024
        hw = int(cls_attn.shape[1] ** 0.5)
        imp = cls_attn.reshape(n_tiles, hw, hw, 1).to(vit_embeds.dtype)
        imp = model.pixel_shuffle(imp, scale_factor=model.downsample_ratio)  # (T,h,w,r^-2)
        imp = imp.mean(dim=-1).reshape(n_tiles, -1)         # (T_tiles, num_image_token)

    kept = []
    for t in range(n_tiles):
        imp_t = imp[t] if imp is not None else None
        seed = None
        if not need_imp:  # deterministic, order/resume-independent
            seed = int(np.random.SeedSequence([int(base_seed), int(sample_id), int(t)]).generate_state(1)[0])
        idx = compressor.select(imp_t, vit_embeds[t], keep, seed=seed)  # (keep,)
        kept.append(vit_embeds[t][idx])
    return torch.cat(kept, dim=0), keep                     # (n_tiles*keep, C), keep
