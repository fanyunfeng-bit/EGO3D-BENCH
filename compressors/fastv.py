"""FastV (Chen et al., ECCV 2024; github.com/pkunlp-icler/FastV) — in-LLM visual-token
pruning at decoder layer K.

FastV keeps ALL visual tokens for layers 0..K-1, then at layer K ranks visual tokens by
the attention they receive from the last prompt token and keeps the top round(keep*M),
dropping the rest for layers K..L-1 (and their KV cache). Default K=2 (paper).

InternVL3 (LM = Qwen2Model, 2D RoPE): reuses the qstage in-LLM patch (`compressors/qstage_llm`)
with signal='attn' — the identical mechanism (prune at layer K by last-token attention, PESP
positions). Qwen2.5-VL (M-RoPE) gets its own patch in `make_fastv_forward_qwen` below.
"""
import torch
import torch.nn.functional as F
from compressors.qstage_llm import QStage, make_qstage_forward


class FastVInternVL:
    """FastV for InternVL3: thin wrapper over the qstage 'attn' path. Attach once after
    model load; call configure() per sample, then model.generate(), then off()."""

    def __init__(self, lm_model, K=2):
        self.qs = QStage(K=K, signal="attn")
        lm_model._qs = self.qs
        make_qstage_forward(lm_model)

    def configure(self, input_ids, img_token_id, keep_ratio, n_views):
        ids = input_ids.reshape(-1)
        vis_pos = torch.where(ids == img_token_id)[0]
        last_vis = int(vis_pos[-1].item())
        self.qs.vis_pos = vis_pos
        self.qs.query_pos = torch.arange(last_vis + 1, ids.shape[0], device=ids.device)
        n_tok = vis_pos.numel() // n_views
        self.qs.keep_pv = max(1, round(keep_ratio * n_tok))   # per-view budget (matches baselines)
        self.qs.n_views = n_views
        self.qs.per_view = True                                # FastV = rank/keep within each view
        self.qs.N2 = self.qs.keep_pv * n_views                 # bookkeeping (total kept)
        self.qs.kept_vis = None
        self.qs.active = True

    def off(self):
        self.qs.active = False


class QwenLLMAttentionCapture:
    """Force ONE Qwen2.5-VL LLM decoder layer's self-attn onto an eager path that stashes the
    attention each token RECEIVES from the last query position (FastV's ranking signal).
    Qwen2.5-VL attention is hardcoded to flash (ignores config._attn_implementation), so we
    monkey-patch the specific layer's self_attn.forward (same trick as the ViT capture)."""

    def __init__(self, self_attn):
        self.attn = self_attn
        self.last_row = None            # (kv_len,) attention from last query to each key, mean heads
        self.enabled = False
        self._orig = self_attn.forward
        self_attn.forward = self._forward

    def _forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                 output_attentions=False, use_cache=False, cache_position=None,
                 position_embeddings=None, **kw):
        a = self.attn
        if not self.enabled:            # decode steps / non-prune: use the stock (flash) path
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
        scores = torch.matmul(q, kk.transpose(2, 3)) / (a.head_dim ** 0.5)   # (1, H, q_len, kv_len)
        kv_len = scores.shape[-1]
        causal = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=scores.device,
                                       dtype=scores.dtype), diagonal=kv_len - q_len + 1)
        aw = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
        self.last_row = aw[0, :, -1, :].mean(0)                             # (kv_len,)
        out = torch.matmul(aw.to(vv.dtype), vv).transpose(1, 2).reshape(bsz, q_len, -1)
        return (a.o_proj(out), None, past_key_value)


class QwenFastV:
    """Controller for the Qwen2.5-VL M-RoPE FastV patch (see make_fastv_forward_qwen).
    kept is computed once at prefill and reused so decode steps read the reduced cache."""

    def __init__(self, K=2):
        self.K = int(K)
        self.active = False
        self.N2 = None            # tokens to keep after layer K (global bookkeeping)
        self.vis_pos = None       # LongTensor of visual token indices in the prompt
        self.last_q = None        # index of the last prompt token (unused; last row = -1)
        self.kept = None          # stashed keep indices (prompt positions), reused across decode
        self.capture = None       # QwenLLMAttentionCapture on layer K-1
        self.per_view = True      # FastV = per-view: rank/keep top keep_pv WITHIN each view
        self.n_views = None
        self.keep_pv = None       # tokens to keep per view


def make_fastv_forward_qwen(text_model, ctrl):
    """Patch a Qwen2.5-VL text model's forward so that, on the prefill pass (S>1) with
    ctrl.active, it prunes visual tokens at layer K by the last prompt token's attention.

    M-RoPE aware: position_ids is (3, B, S); kept tokens keep their ORIGINAL 3D positions
    (PESP), sorted. Non-capture layers use stock flash (causal_mask is None under FA2, so no
    mask is threaded); layer K-1 uses the bespoke eager capture to get the ranking signal.
    """
    import types
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    ctrl.capture = QwenLLMAttentionCapture(text_model.layers[ctrl.K - 1].self_attn)

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
                                               past_key_values, False)   # None under flash
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        do_prune = ctrl.active and S > 1
        for idx, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if do_prune and idx == ctrl.K:                               # prune BEFORE layer K
                vis_set = set(ctrl.vis_pos.tolist()); kept = set(ctrl.kept.tolist())
                keep = [i for i in range(hidden_states.shape[1]) if (i not in vis_set) or (i in kept)]
                ki = torch.tensor(sorted(keep), device=hidden_states.device)
                hidden_states = hidden_states[:, ki, :]
                position_ids = position_ids[:, :, ki]                        # (3,B,S): seq = dim 2
                position_embeddings = (position_embeddings[0].index_select(-2, ki),
                                       position_embeddings[1].index_select(-2, ki))  # seq = dim -2
                cache_position = cache_position[ki]
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, ki][:, :, :, ki]

            cap = do_prune and idx == ctrl.K - 1
            if cap:
                ctrl.capture.enabled = True
            lo = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                       past_key_value=past_key_values, use_cache=use_cache,
                       cache_position=cache_position, position_embeddings=position_embeddings)
            hidden_states = lo[0]
            if cap:
                ctrl.capture.enabled = False
                if ctrl.kept is None:
                    vis = ctrl.vis_pos.to(hidden_states.device)
                    score = ctrl.capture.last_row[vis]                   # last query -> each visual token
                    if ctrl.per_view and ctrl.n_views:
                        nv = int(ctrl.n_views); ntok = vis.numel() // nv; kp = min(int(ctrl.keep_pv), ntok)
                        loc = score.view(nv, ntok).topk(kp, dim=1).indices   # (nv, kp) within-view
                        off = (torch.arange(nv, device=vis.device) * ntok).unsqueeze(1)
                        ctrl.kept = vis[(loc + off).reshape(-1)].sort().values
                    else:
                        n2 = min(int(ctrl.N2), vis.numel())
                        ctrl.kept = vis[torch.topk(score, n2).indices].sort().values

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None)

    text_model.forward = types.MethodType(forward, text_model)
    return text_model
