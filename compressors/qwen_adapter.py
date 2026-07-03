"""Qwen2.5-VL glue for the model-agnostic TokenCompressor interface.

Qwen2.5-VL differs from InternVL in every way that matters for pruning:
  * ViT has no CLS token, uses windowed attention, and the attention classes
    discard weights -> we monkey-patch ONE full-attention block to stash the
    softmax attention (replicating the eager path), and use mean-attention-
    RECEIVED as the importance cue (the CLS-free analog).
  * a 2x2 patch merger + a window reordering sit between patches and the LLM
    tokens -> importance is aggregated patch->merged-token and un-permuted with
    the model's own window_index, so it aligns with `model.visual`'s output.
  * the LLM uses M-RoPE -> dropping tokens is handled in the runner (it keeps
    each retained token's original 2D position); this file only selects indices.

`needs_importance=False` compressors (random) skip the capture entirely.
"""

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision


class QwenAttentionCapture:
    """Force one full-attention ViT block onto an eager path that stashes the
    per-patch attention-received (mean over heads and over query positions)."""

    def __init__(self, model):
        vt = model.visual
        full = list(vt.fullatt_block_indexes)
        self.layer_idx = full[len(full) // 2]  # a middle full-attention block
        self.attn = vt.blocks[self.layer_idx].attn
        # flash/sdpa attn classes don't store head_dim; derive from qkv weight
        self.head_dim = self.attn.qkv.weight.shape[0] // (3 * self.attn.num_heads)
        self.received = None
        self.attn.forward = self._forward  # bound replacement

    def _forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        a = self.attn
        seq = hidden_states.shape[0]
        q, k, v = a.qkv(hidden_states).reshape(seq, 3, a.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)   # (heads, seq, d)
        # Attention is block-diagonal over the cu_seqlens segments (per-frame full attention),
        # so compute each segment separately: identical result, O(sum seg^2) mem instead of
        # O(seq^2) -> avoids a 16k x 16k (~17 GB) materialization at 16 frames.
        received = torch.zeros(seq, dtype=torch.float32, device=q.device)
        out = torch.empty_like(vh)                                           # (heads, seq, d)
        scale = self.head_dim ** -0.5
        cu = cu_seqlens.tolist()
        for i in range(1, len(cu)):
            lo, hi = cu[i - 1], cu[i]
            qs, ks, vs = qh[:, lo:hi], kh[:, lo:hi], vh[:, lo:hi]
            aw = torch.softmax((qs @ ks.transpose(1, 2)) * scale, dim=-1, dtype=torch.float32)  # (heads,s,s)
            received[lo:hi] = aw.mean(dim=1).mean(dim=0)                     # attn received (mean over q, heads)
            out[:, lo:hi] = (aw.to(vs.dtype) @ vs)
        self.received = received.to(hidden_states.dtype)                     # (seq,)
        return a.proj(out.transpose(0, 1).reshape(seq, -1))


@torch.no_grad()
def merged_importance(model, grid_thw, received):
    """patch-level received attention (window order) -> per merged-token
    importance in the ORIGINAL order that `model.visual` outputs."""
    vt = model.visual
    unit = vt.spatial_merge_unit                       # 4
    window_index, _ = vt.get_window_index(grid_thw)
    n_merged = received.shape[0] // unit
    imp_window = received.reshape(n_merged, unit).mean(dim=1)    # mean over the 2x2 block
    reverse = torch.argsort(window_index)
    return imp_window[reverse]                         # aligned with model.visual output


@torch.no_grad()
def qwen_select_per_image(model, image_embeds, importance, grid_thw, compressor,
                          keep_ratio, base_seed=42, sample_id=0):
    """Per image (frame), pick which merged tokens to keep. Returns a boolean
    keep-mask over the full visual-token sequence (model.visual output order)."""
    import numpy as np
    device = image_embeds.device
    keep_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=device)
    need_imp = getattr(compressor, "needs_importance", True)
    off = 0
    for f, (t, h, w) in enumerate(grid_thw.tolist()):
        n = t * (h // model.visual.spatial_merge_size) * (w // model.visual.spatial_merge_size)
        keep = max(1, round(n * keep_ratio))
        feats = image_embeds[off:off + n]
        imp = importance[off:off + n] if (need_imp and importance is not None) else None
        seed = None
        if not need_imp:
            seed = int(np.random.SeedSequence([int(base_seed), int(sample_id), int(f)]).generate_state(1)[0])
        idx = compressor.select(imp, feats, keep, seed=seed)
        keep_mask[off + idx] = True
        off += n
    return keep_mask
