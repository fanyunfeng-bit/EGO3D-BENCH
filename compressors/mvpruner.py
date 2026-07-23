"""MVPruner (Zhang et al., arXiv:2606.27660; github.com/Zizzzzzzz/MVPruner) — faithful port of
the two-stage in-LLM multi-view token pruner, re-targeted from DriveMM/LLaMA to this repo's
InternVL3 (Qwen2Model, 1D RoPE) and Qwen2.5-VL (M-RoPE) VSI runners, as a comparison baseline.

Original method (verbatim from modeling_llama_mvpruner.py):
  * Stage-1 @ layer L1 (paper 0) — DRA/CCTS: per-view budget from intra-view *diversity*
    (mean nearest-neighbour distance, 1-cos); select within a view by `task_guided_sampling`
    (greedy uniqueness x text-relevance) when instruction text exists, else farthest-point.
  * Stage-2 @ layer L2 (paper 16), attention captured @ L2-1 — IRA/ITS: view importance from
    text->vision attention reweights per-view budget; keep top-k by instruction attention.
  Kept tokens retain ORIGINAL position ids (PESP), so RoPE + causal order stay correct; layers
  L1..L2-1 carry N1 vision tokens, L2..L-1 carry N2 -> layer-average vision fraction = keep_ratio.

Budget: the paper anchors `stage_keep_ratio` to a target FLOP (layer-average) budget — the SAME
convention as this repo's scmpruner_qa. `stage_keep_ratio_for()` solves that quadratic so
MVPruner's keep25/10/5 are FLOP-matched to random/vispruner/fastv/SCM (verified: paper's 0.169
for keep10 falls out at their L=32,L2=16).

Model-specific forward patches: `make_mvpruner_forward_qwen` (M-RoPE) and
`make_mvpruner_forward_internvl` (1D RoPE) — they mirror compressors/fastv.py + qstage_llm.py,
adding a SECOND prune point and MVPruner's selection. Selection core below is model-agnostic.
"""
import math
import os
import torch
import torch.nn.functional as F

_DBG = os.environ.get("MV_DEBUG")


# ============================ budget calibration ============================
def stage_keep_ratio_for(keep_ratio, L, L1=0, L2=16):
    """Return the per-stage keep ratio s such that the layer-average vision fraction equals
    keep_ratio: L1 layers at 1.0, (L2-L1) at s, (L-L2) at s^2  ->  L1 + (L2-L1)s + (L-L2)s^2 = L*keep.
    (Matches the paper's own calibration: L=32,L2=16,keep=0.1 -> s~=0.169.)"""
    a = (L - L2)
    b = (L2 - L1)
    c = (L1 - L * float(keep_ratio))
    if a <= 0:                       # L2 == L: single-stage linear
        s = -c / b
    else:
        s = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return float(min(1.0, max(1e-4, s)))


# ============================ selection primitives (verbatim MVPruner) ============================
def _softmax(x):
    e = torch.exp(x)
    return e / (e.sum() + 1e-8)


def _dmat(states):
    """(n,C) -> (n,n) distance = 1 - cosine, diagonal = +inf (self excluded)."""
    s = F.normalize(states, p=2, dim=1)
    d = 1.0 - s @ s.T
    d.fill_diagonal_(float("inf"))
    return d


def _keep_count(n, ratio):
    return max(1, min(int(n * float(ratio)), n))


def _fps(k, dmat):
    """Farthest-point sampling on a precomputed distance matrix; returns k indices."""
    n = dmat.shape[0]
    k = min(k, n)
    sel = torch.zeros(n, dtype=torch.bool, device=dmat.device)
    res = []
    for _ in range(k):
        if sel.sum() == 0:
            avail = dmat.clone()
            avail[:, sel] = float("-inf")
            row_mins, _ = avail.min(dim=1)
            row_mins[sel] = float("-inf")
            nxt = row_mins.argmax()
        else:
            seld = dmat[sel, :]
            md, _ = seld.min(dim=0)
            md[sel] = float("-inf")
            nxt = md.argmax()
        sel[nxt] = True
        res.append(int(nxt.item()))
    return torch.tensor(res[-k:], device=dmat.device)


def _task_guided(states, k, relevance):
    """Greedy uniqueness (min-dist to selected) x task relevance; returns k indices."""
    dmat = _dmat(states)
    n = states.shape[0]
    k = min(k, n)
    rel = relevance.reshape(-1)
    sel = torch.zeros(n, dtype=torch.bool, device=states.device)
    res = []
    for _ in range(k):
        if sel.sum() == 0:
            sc = rel.clone()
        else:
            seld = dmat[sel, :]
            md, _ = seld.min(dim=0)
            sc = md * rel
        sc[sel] = float("-inf")
        nxt = sc.argmax()
        sel[nxt] = True
        res.append(int(nxt.item()))
    return torch.tensor(res[-k:], device=states.device)


def stage1_keep(hidden, view_idx_list, text_idx, skr):
    """Stage-1 DRA/CCTS. hidden: (S,C). view_idx_list: list of global vision-index LongTensors
    (one per view). text_idx: LongTensor of instruction-token indices (may be empty). skr:
    stage_keep_ratio. Returns list[LongTensor] of GLOBAL kept-vision indices per view."""
    nviews = len(view_idx_list)
    has_text = text_idx is not None and text_idx.numel() > 0
    tstates = hidden[text_idx] if has_text else None
    dmats, div, relev = [], [], []
    for vi in view_idx_list:
        vs = hidden[vi]
        d = _dmat(vs)
        dmats.append(d)
        mind, _ = d.min(dim=-1)
        div.append(mind.mean())
        relev.append((vs @ tstates.T).mean(dim=1) if has_text else None)
    div = torch.stack(div)
    div = (div - div.max()) / (div.max() - div.min() + 1e-5)
    ratios = skr * nviews * _softmax(div)
    out = []
    for j, vi in enumerate(view_idx_list):
        kc = _keep_count(vi.numel(), ratios[j])
        ki = _task_guided(hidden[vi], kc, relev[j]) if has_text else _fps(kc, dmats[j])
        out.append(vi[ki])
    return out


def stage2_keep(hidden, view_idx_list, text_slice, attn_mean, skr):
    """Stage-2 IRA/ITS over the (already stage-1-pruned) sequence. hidden: (S1,C). view_idx_list:
    list of COMPACT vision-index LongTensors per view. text_slice: (t0,t1) compact instruction
    span. attn_mean: (S1,S1) attention averaged over heads (from layer L2-1), or None. Returns
    list[LongTensor] of COMPACT kept-vision indices per view."""
    nviews = len(view_idx_list)
    t0, t1 = text_slice
    has_text = attn_mean is not None and t1 > t0
    scores = None
    if has_text:
        scores = [attn_mean[t0:t1][:, vi].mean(dim=0) for vi in view_idx_list]   # (Lv,) per view
        vimp = torch.stack([s.mean() for s in scores])
        vimp = (vimp - vimp.max()) / (vimp.max() - vimp.min() + 1e-5)
        vimp = _softmax(vimp)
        lens = torch.tensor([float(vi.numel()) for vi in view_idx_list], device=hidden.device)
        share = lens / lens.sum()
        ratios = skr * (1.0 + vimp - share)
    else:
        ratios = [skr] * nviews
    out = []
    for j, vi in enumerate(view_idx_list):
        r = ratios[j].item() if torch.is_tensor(ratios[j]) else ratios[j]
        kc = _keep_count(vi.numel(), r)
        if scores is not None:
            ki = torch.topk(scores[j], k=kc, largest=True, sorted=False).indices
        else:
            ki = _fps(kc, _dmat(hidden[vi]))
        out.append(vi[ki])
    return out


# ============================ controller ============================
class MVPruner:
    """Attach once after model load; call configure() per sample, generate, then off().
    Holds the two stage layers, the FLOP-matched stage_keep_ratio, and per-sample view/text
    layout. Prefill (S>1) computes both keep-sets and builds the pruned KV cache; decode reuses."""

    def __init__(self, stage1_layer=0, stage2_layer=16):
        self.L1 = int(stage1_layer)
        self.L2 = int(stage2_layer)
        self.attn_layer = int(stage2_layer) - 1
        self.active = False
        self.stage_keep_ratio = None
        self.vis_pos = None          # LongTensor: global vision indices in the prompt (sorted)
        self.view_lengths = None     # list[int]: vision tokens per view (sum == len(vis_pos))
        self.text_start = None       # int: instruction span start (global)
        self.text_end = None         # int: instruction span end (exclusive)
        # scratch (rebuilt each prefill):
        self._attn_mean = None

    def configure(self, vis_pos, view_lengths, text_start, text_end, keep_ratio, num_layers):
        self.vis_pos = vis_pos
        self.view_lengths = [int(x) for x in view_lengths]
        self.text_start = int(text_start)
        self.text_end = int(text_end)
        self.stage_keep_ratio = stage_keep_ratio_for(keep_ratio, num_layers, self.L1, self.L2)
        self.active = True
        self._attn_mean = None

    def off(self):
        self.active = False

    # -- shared bookkeeping helpers used by both forward patches --
    def view_index_tensors(self, device):
        """Split the global vis_pos into one LongTensor per view (original global indices)."""
        out, off = [], 0
        for L in self.view_lengths:
            out.append(self.vis_pos[off:off + L].to(device))
            off += L
        return out

    def compact_view_tensors(self, keep_vis_per_view, keep1_sorted):
        """Given per-view GLOBAL kept-vision indices and the sorted stage-1 keep set, return each
        view's indices in COMPACT (post-stage-1) coordinates, plus the compact instruction span."""
        pos = {int(g): i for i, g in enumerate(keep1_sorted.tolist())}
        comp = [torch.tensor(sorted(pos[int(g)] for g in vv.tolist()), device=keep1_sorted.device)
                for vv in keep_vis_per_view]
        # instruction tokens are never pruned -> map the contiguous global span into compact coords
        t0 = min(pos[i] for i in range(self.text_start, self.text_end) if i in pos)
        t1 = max(pos[i] for i in range(self.text_start, self.text_end) if i in pos) + 1
        return comp, (t0, t1)


def _keep1_indices(ctrl, S, device):
    """Prefill stage-1: return (keep1_sorted, compact_view_tensors, compact_text_slice) given the
    already-computed per-view global kept vision (stored transiently). Non-vision tokens all kept."""
    return None  # (unused placeholder; logic inlined in the forward patches for clarity)


# ============================ Qwen2.5-VL (M-RoPE) forward patch ============================
class MVPrunerQwenCapture:
    """Force ONE Qwen2.5-VL LLM layer onto an eager path that stashes the full mean-over-heads
    attention (q,kv) at layer L2-1 (MVPruner reads text->vision rows from it)."""

    def __init__(self, self_attn):
        self.attn = self_attn
        self.mat = None
        self.enabled = False
        self._orig = self_attn.forward
        self_attn.forward = self._forward

    def _forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                 output_attentions=False, use_cache=False, cache_position=None,
                 position_embeddings=None, **kw):
        a = self.attn
        if not self.enabled:
            return self._orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                              past_key_value=past_key_value, output_attentions=output_attentions,
                              use_cache=use_cache, cache_position=cache_position,
                              position_embeddings=position_embeddings, **kw)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb, repeat_kv)
        bsz, q_len, _ = hidden_states.size()
        q = a.q_proj(hidden_states).view(bsz, q_len, -1, a.head_dim).transpose(1, 2)
        k = a.k_proj(hidden_states).view(bsz, q_len, -1, a.head_dim).transpose(1, 2)
        v = a.v_proj(hidden_states).view(bsz, q_len, -1, a.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, a.rope_scaling["mrope_section"])
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, a.layer_idx,
                                         {"sin": sin, "cos": cos, "cache_position": cache_position})
        kk, vv = repeat_kv(k, a.num_key_value_groups), repeat_kv(v, a.num_key_value_groups)
        scores = torch.matmul(q, kk.transpose(2, 3)) / (a.head_dim ** 0.5)
        kv_len = scores.shape[-1]
        causal = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=scores.device,
                                       dtype=scores.dtype), diagonal=kv_len - q_len + 1)
        aw = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
        self.mat = aw[0].mean(0)                       # (q_len, kv_len) mean over heads
        out = torch.matmul(aw.to(vv.dtype), vv).transpose(1, 2).reshape(bsz, q_len, -1)
        return (a.o_proj(out), None, past_key_value)


def _prune_reindex_qwen(hidden_states, position_ids, position_embeddings, cache_position,
                        causal_mask, ki):
    """Slice all M-RoPE state to the kept indices ki (sorted); kept tokens keep original 3D pos."""
    hidden_states = hidden_states[:, ki, :]
    position_ids = position_ids[:, :, ki]                                   # (3,B,S): seq = dim 2
    position_embeddings = (position_embeddings[0].index_select(-2, ki),
                           position_embeddings[1].index_select(-2, ki))     # seq = dim -2
    cache_position = cache_position[ki]
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, ki][:, :, :, ki]
    return hidden_states, position_ids, position_embeddings, cache_position, causal_mask


def make_mvpruner_forward_qwen(text_model, ctrl):
    """Patch a Qwen2.5-VL text model forward for MVPruner two-stage in-LLM pruning (prefill only)."""
    import types
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    ctrl.capture = MVPrunerQwenCapture(text_model.layers[ctrl.attn_layer].self_attn)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                cache_position=None, **kw):
        use_cache = True if use_cache is None else use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        S = inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + S, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position,
                                               past_key_values, False)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        do_prune = ctrl.active and S > 1
        comp_views, comp_text = None, None
        for idx, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if do_prune and idx == ctrl.L1:                                 # --- Stage-1 ---
                dev = hidden_states.device
                views = ctrl.view_index_tensors(dev)
                tidx = torch.arange(ctrl.text_start, ctrl.text_end, device=dev)
                kept_vis = stage1_keep(hidden_states[0], views, tidx, ctrl.stage_keep_ratio)
                vis_set = set(ctrl.vis_pos.tolist())
                kept_set = set(int(x) for vv in kept_vis for x in vv.tolist())
                keep1 = torch.tensor(sorted(i for i in range(hidden_states.shape[1])
                                            if i not in vis_set or i in kept_set), device=dev)
                comp_views, comp_text = ctrl.compact_view_tensors(kept_vis, keep1)
                if _DBG:
                    print(f"[MV] stage1 @L{idx}: S {hidden_states.shape[1]} -> {keep1.numel()} "
                          f"(vis {ctrl.vis_pos.numel()} -> {sum(v.numel() for v in kept_vis)}, skr={ctrl.stage_keep_ratio:.3f})", flush=True)
                (hidden_states, position_ids, position_embeddings, cache_position,
                 causal_mask) = _prune_reindex_qwen(hidden_states, position_ids,
                                                    position_embeddings, cache_position, causal_mask, keep1)

            elif do_prune and idx == ctrl.L2 and comp_views is not None:    # --- Stage-2 ---
                dev = hidden_states.device
                kept_vis2 = stage2_keep(hidden_states[0], comp_views, comp_text, ctrl._attn_mean,
                                        ctrl.stage_keep_ratio)
                vis_set = set(int(x) for vv in comp_views for x in vv.tolist())
                kept_set = set(int(x) for vv in kept_vis2 for x in vv.tolist())
                keep2 = torch.tensor(sorted(i for i in range(hidden_states.shape[1])
                                            if i not in vis_set or i in kept_set), device=dev)
                if _DBG:
                    print(f"[MV] stage2 @L{idx}: S {hidden_states.shape[1]} -> {keep2.numel()} "
                          f"(vis {sum(v.numel() for v in comp_views)} -> {sum(v.numel() for v in kept_vis2)})", flush=True)
                (hidden_states, position_ids, position_embeddings, cache_position,
                 causal_mask) = _prune_reindex_qwen(hidden_states, position_ids,
                                                    position_embeddings, cache_position, causal_mask, keep2)

            cap = do_prune and idx == ctrl.attn_layer
            if cap:
                ctrl.capture.enabled = True
            lo = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                       past_key_value=past_key_values, use_cache=use_cache,
                       cache_position=cache_position, position_embeddings=position_embeddings)
            hidden_states = lo[0]
            if cap:
                ctrl.capture.enabled = False
                ctrl._attn_mean = ctrl.capture.mat

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None)

    text_model.forward = types.MethodType(forward, text_model)
    return text_model


# ============================ InternVL3 (Qwen2Model, 1D RoPE) forward patch ============================
def _stage1_compute(ctrl, hidden, dev):
    views = ctrl.view_index_tensors(dev)
    tidx = torch.arange(ctrl.text_start, ctrl.text_end, device=dev)
    kept_vis = stage1_keep(hidden[0], views, tidx, ctrl.stage_keep_ratio)
    vis_set = set(ctrl.vis_pos.tolist())
    kept_set = set(int(x) for vv in kept_vis for x in vv.tolist())
    keep1 = torch.tensor(sorted(i for i in range(hidden.shape[1]) if i not in vis_set or i in kept_set),
                         device=dev)
    comp_views, comp_text = ctrl.compact_view_tensors(kept_vis, keep1)
    return keep1, comp_views, comp_text, sum(v.numel() for v in kept_vis)


def _stage2_compute(ctrl, hidden, comp_views, comp_text, dev):
    kept2 = stage2_keep(hidden[0], comp_views, comp_text, ctrl._attn_mean, ctrl.stage_keep_ratio)
    vis_set = set(int(x) for vv in comp_views for x in vv.tolist())
    kept_set = set(int(x) for vv in kept2 for x in vv.tolist())
    keep2 = torch.tensor(sorted(i for i in range(hidden.shape[1]) if i not in vis_set or i in kept_set),
                         device=dev)
    return keep2, sum(v.numel() for v in kept2)


def _reindex_1d(hidden, position_ids, position_embeddings, cache_position, causal_mask, ki):
    hidden = hidden[:, ki, :]
    position_ids = position_ids[:, ki]
    position_embeddings = (position_embeddings[0][:, ki, :], position_embeddings[1][:, ki, :])
    cache_position = cache_position[ki]
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, ki][:, :, :, ki]
    return hidden, position_ids, position_embeddings, cache_position, causal_mask


def make_mvpruner_forward_internvl(model, ctrl):
    """Patch InternVL3's Qwen2Model.forward for MVPruner two-stage in-LLM pruning. 1D RoPE; the
    L2-1 attention is captured by forcing that one layer onto eager (config toggle). Decode RoPE
    positions are taken from a counter (stage-1 @L0 shrinks layer-0 cache, so get_seq_length is
    not the original length)."""
    import types
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, cache_position=None, **kw):
        use_cache = True if use_cache is None else use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        S = inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + S, device=inputs_embeds.device)
        if ctrl.active and S == 1:                            # decode: RoPE position from counter
            position_ids = torch.full((1, 1), ctrl._orig_next + ctrl._dstep,
                                      device=inputs_embeds.device, dtype=torch.long)
            ctrl._dstep += 1
        elif position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position,
                                               past_key_values, False)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        do_prune = ctrl.active and S > 1
        if do_prune:
            ctrl._orig_next = S
            ctrl._dstep = 0
            ctrl._attn_mean = None
        comp_views, comp_text = None, None
        for idx, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if do_prune and idx == ctrl.attn_layer:                     # --- eager attention capture ---
                saved = self.config._attn_implementation
                self.config._attn_implementation = "eager"
                emask = self._update_causal_mask(attention_mask, hidden_states, cache_position,
                                                 past_key_values, True)
                lo = layer(hidden_states, attention_mask=emask, position_ids=position_ids,
                           past_key_value=past_key_values, output_attentions=True, use_cache=use_cache,
                           cache_position=cache_position, position_embeddings=position_embeddings)
                self.config._attn_implementation = saved
                hidden_states = lo[0]
                ctrl._attn_mean = lo[1].mean(dim=1).squeeze(0)          # (S1,S1) mean over heads
                continue

            if do_prune and idx == ctrl.L1:                             # --- Stage-1 ---
                dev = hidden_states.device
                keep1, comp_views, comp_text, nvis = _stage1_compute(ctrl, hidden_states, dev)
                if _DBG:
                    print(f"[MV] stage1 @L{idx}: S {hidden_states.shape[1]} -> {keep1.numel()} "
                          f"(vis {ctrl.vis_pos.numel()} -> {nvis}, skr={ctrl.stage_keep_ratio:.3f})", flush=True)
                (hidden_states, position_ids, position_embeddings, cache_position,
                 causal_mask) = _reindex_1d(hidden_states, position_ids, position_embeddings,
                                            cache_position, causal_mask, keep1)
            elif do_prune and idx == ctrl.L2 and comp_views is not None:  # --- Stage-2 ---
                dev = hidden_states.device
                keep2, nvis2 = _stage2_compute(ctrl, hidden_states, comp_views, comp_text, dev)
                if _DBG:
                    print(f"[MV] stage2 @L{idx}: S {hidden_states.shape[1]} -> {keep2.numel()} "
                          f"(vis -> {nvis2})", flush=True)
                (hidden_states, position_ids, position_embeddings, cache_position,
                 causal_mask) = _reindex_1d(hidden_states, position_ids, position_embeddings,
                                            cache_position, causal_mask, keep2)

            lo = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                       past_key_value=past_key_values, output_attentions=False, use_cache=use_cache,
                       cache_position=cache_position, position_embeddings=position_embeddings)
            hidden_states = lo[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None,
                                       hidden_states=None, attentions=None)

    model.forward = types.MethodType(forward, model)
    return model
