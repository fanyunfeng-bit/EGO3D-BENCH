"""Two-stage query-aware stage-2: in-LLM vision-token prune at layer K (Notes/CVSP-Method.md §13).

Patches InternVL3's Qwen2Model.forward so that, during a prefill forward (seq_len > 1), at
decoder layer K it keeps only the top-N2 vision tokens by a query signal, dropping the rest with
Position Embedding Sparse Preservation (PESP: kept tokens keep their ORIGINAL position_ids/rotary,
sorted, so RoPE + FA causal order stay correct). Layers 0..K-1 see N1 vision tokens, layers
K..L-1 see N2 -> layer-average budget = (N1*K + N2*(L-K))/L.

Signals (qs.signal):
  'cosine' (paper version, FA-preserving): q_bar = mean over query tokens of h_K; r = cos(h_K[vis], q_bar).
  'attn'   (Nuwa code version): last query token's attention to vision at layer K-1 (forces that
           one layer onto eager to materialize weights; rest keep FlashAttention).

The keep-set is computed once (first forward = prefill) and stashed in qs.kept_vis (original vision
indices), then reused on every later forward so a custom greedy decode loop with use_cache=False
prunes the SAME vision tokens each step.
"""
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast


class QStage:
    """Controller; attach as qwen2_model._qs and flip .active per generation."""
    def __init__(self, K, signal="cosine"):
        self.K = int(K)
        self.signal = signal              # 'cosine' | 'attn'
        self.active = False
        self.N2 = None                    # how many vision tokens to keep after layer K
        self.vis_pos = None               # LongTensor: vision token indices in the prompt
        self.query_pos = None             # LongTensor: question token indices (for cosine q_bar)
        self.kept_vis = None              # stashed kept vision indices (orig), reused across decode
        self.last_avg_tokens = None       # bookkeeping: layer-average vision-token count
        self.per_view = False             # FastV per-view variant: rank/keep within each view
        self.n_views = None               # # views (per_view); vision tokens are grouped per view
        self.keep_pv = None               # tokens to keep per view (per_view)
        self.query_reduce = "last"        # 'last' (FastV) | 'mean' (ITS-style over query tokens)


def _select_keep(qs, hidden_states, attn_K_minus_1):
    """Return sorted LongTensor of vision indices to KEEP, computed at prefill. Global top-N2
    by default; if qs.per_view, keep top qs.keep_pv WITHIN each view (vision tokens are grouped
    per view) -> per-view budget, matching the single-view baselines."""
    from compressors.scm import cosine_relevance
    vis = qs.vis_pos.to(hidden_states.device)
    if qs.signal == "attn":
        a = attn_K_minus_1.mean(dim=1).squeeze(0)            # (S,S) avg over heads, batch=1
        if getattr(qs, "query_reduce", "last") == "mean" and qs.query_pos is not None:
            qp = qs.query_pos.to(a.device)
            score = a[qp][:, vis].mean(0)                    # mean over query tokens (ITS)
        else:
            last_q = qs.query_pos[-1].item() if qs.query_pos is not None else hidden_states.shape[1] - 1
            score = a[last_q, vis]                           # last query token (FastV default)
    else:  # cosine
        score = cosine_relevance(hidden_states[0], vis, qs.query_pos.to(hidden_states.device))
    if getattr(qs, "per_view", False) and qs.n_views:
        nv = int(qs.n_views); ntok = vis.numel() // nv; kp = min(int(qs.keep_pv), ntok)
        loc = score.view(nv, ntok).topk(kp, dim=1).indices   # (nv, kp) within-view ranking
        off = (torch.arange(nv, device=vis.device) * ntok).unsqueeze(1)
        return vis[(loc + off).reshape(-1)].sort().values
    n2 = min(int(qs.N2), vis.numel())
    top = torch.topk(score, n2).indices
    return vis[top].sort().values


def make_qstage_forward(model):
    """Bind a prune-enabled forward to `model` (a Qwen2Model). Reads state from model._qs."""

    def forward(self,
                input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                cache_position=None, **kw):
        from transformers.cache_utils import DynamicCache
        qs = self._qs
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
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, False)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        do_prune = qs.active and S > 1                        # prefill only (decode S==1 just reads cache)
        attn_cap = None

        for idx, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # --- capture attention at layer K-1 for the 'attn' signal (forces eager, just this layer) ---
            if do_prune and qs.signal == "attn" and idx == qs.K - 1:
                saved_impl = self.config._attn_implementation
                self.config._attn_implementation = "eager"
                eager_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position,
                                                      past_key_values, True)
                lo = layer(hidden_states, attention_mask=eager_mask, position_ids=position_ids,
                           past_key_value=past_key_values, output_attentions=True, use_cache=use_cache,
                           cache_position=cache_position, position_embeddings=position_embeddings)
                self.config._attn_implementation = saved_impl
                hidden_states = lo[0]
                attn_cap = lo[1]
                continue

            # --- prune hidden states before layer K (layers <K KEEP full-N1 KV in cache; layers >=K
            #     will append only N2 KV -> exact two-stage, FA2 needs no explicit mask) ---
            if do_prune and idx == qs.K:
                if qs.kept_vis is None:
                    qs.kept_vis = _select_keep(qs, hidden_states, attn_cap)
                vis_set = set(qs.vis_pos.tolist())
                kept_vis = set(qs.kept_vis.tolist())
                keep = [i for i in range(hidden_states.shape[1]) if (i not in vis_set) or (i in kept_vis)]
                keep_idx = torch.tensor(sorted(keep), device=hidden_states.device)
                hidden_states = hidden_states[:, keep_idx, :]
                position_ids = position_ids[:, keep_idx]                       # PESP: original positions
                position_embeddings = (position_embeddings[0][:, keep_idx, :],
                                       position_embeddings[1][:, keep_idx, :])
                cache_position = cache_position[keep_idx]
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, keep_idx][:, :, :, keep_idx]

            lo = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                       past_key_value=past_key_values, output_attentions=False, use_cache=use_cache,
                       cache_position=cache_position, position_embeddings=position_embeddings)
            hidden_states = lo[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None,
                                       hidden_states=None, attentions=None)

    import types
    model.forward = types.MethodType(forward, model)
    return model
