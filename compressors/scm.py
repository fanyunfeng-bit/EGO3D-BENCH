"""SCMPruner core (Notes/SCMPruner-Method.md) — model-agnostic cross-view selection.

Single source of truth for the anchor score + three-bucket selector, imported by
BOTH Harness B (scripts/cvsp_curve.py) and the VSI runners (models/*_vsibench.py) so
InternVL3 and Qwen2.5-VL run byte-identical selection logic. Operates purely on a
token Gram matrix G (cosine) over the concatenated per-view visual tokens; the caller
supplies G, per-token saliency s, and the view layout.
"""
import torch
import torch.nn.functional as F


def scmpruner_keep_indices(feats2d, saliency, n_views, n_tok, keep_ratio,
                           anc_tau=0.6, anc_m=0.12, rho_a=1.0 / 3, rho_s=1.0 / 3):
    """End-to-end SCMPruner selection for one sample, shared by both VSI runners.
    feats2d   : (M, C) concatenated per-view visual features (M = n_views*n_tok).
    saliency  : (M,) per-token foreground cue (InternViT CLS attention / Qwen attn-received).
    Returns a sorted python list of kept GLOBAL token indices (length keep_pv*n_views),
    with xview coverage propagation ON (match_idx passed). Deterministic."""
    Fn = F.normalize(feats2d.float(), dim=-1)
    G = Fn @ Fn.t()
    view_id = torch.arange(feats2d.shape[0], device=feats2d.device) // n_tok
    a, support, match_idx = anchor_scores(G, view_id, n_views, anc_tau, anc_m)
    keep_pv = max(1, round(n_tok * keep_ratio))
    K = keep_pv * n_views
    gi = sel_scmpruner(G, a, saliency, support, view_id, n_views, n_tok, K,
                       rho_a=rho_a, rho_s=rho_s, anc_tau=anc_tau, match_idx=match_idx)
    return sorted(gi)


def _facility_pick(G, cov, avail, vmask=None):
    """One facility-location greedy pick: token maximizing extra coverage of the
    pool, i.e. argmax_t sum_u relu(G[t,u] - cov[u]). vmask restricts candidates."""
    gains = torch.relu(G - cov.unsqueeze(0)).sum(dim=1)      # (M,)
    cand = avail if vmask is None else (avail & vmask)
    gains[~cand] = -1e18
    j = int(gains.argmax())
    return j if bool(cand[j]) else -1


def anchor_scores(G, view_id, n_views, anc_tau, anc_m):
    """SCMPruner anchor = support x sharpness (Notes/SCMPruner-Method.md §2-3).
    For token t (view v), over every OTHER view u take its top-2 cross-view cosines
    s1>=s2; the match is 'sharp' if s1 > anc_tau AND Lowe margin (s1-s2) > anc_m.
      support(t)   = # of other views with a sharp match  (how many views re-identify t)
      sharpness(t) = mean margin over those matched views  (how unique the match is)
      a(t)         = support(t) * sharpness(t)             (quantity x quality landmark)
    Returns (a, support, match_idx):
      support labels 'sharp' (>=1) vs 'fuzzy' (==0) tokens for the saliency de-redundancy;
      match_idx[t,u] = global index of t's best match in view u IF that match is sharp,
      else -1 (own view = -1). match_idx are the sharp cross-view correspondence edges the
      xview coverage propagation walks (§10 safe variant: only sharp edges propagate)."""
    M = G.shape[0]
    support = torch.zeros(M, device=G.device)
    msum = torch.zeros(M, device=G.device)
    match_idx = torch.full((M, n_views), -1, dtype=torch.long, device=G.device)
    for v in range(n_views):
        rows = torch.where(view_id == v)[0]
        if rows.numel() == 0:
            continue
        for u in range(n_views):
            if u == v:
                continue
            cols = torch.where(view_id == u)[0]
            if cols.numel() < 2:
                continue
            tk = G[rows][:, cols].topk(2, dim=1)
            s1 = tk.values[:, 0]; margin = (s1 - tk.values[:, 1]).clamp(min=0)
            matched = (s1 > anc_tau) & (margin > anc_m)
            support[rows] += matched.float()
            msum[rows] += torch.where(matched, margin, torch.zeros_like(margin))
            best = cols[tk.indices[:, 0]]                    # global idx of top-1 match
            match_idx[rows, u] = torch.where(matched, best, torch.full_like(best, -1))
    a = support * (msum / support.clamp(min=1))
    return a, support, match_idx


def sel_scmpruner(G, a, s, support, view_id, n_views, n_tok, K,
                  rho_a, rho_s, anc_tau, match_idx=None):
    """SCMPruner selector (Notes/SCMPruner-Method.md §4). Three buckets over budget K:
      1) anchor   : top round(rho_a*K) by a=support*sharpness, NO dedup -> keep every
                    cross-view copy of a landmark (the geometric signal).
      2) saliency : top-s up to round(rho_s*K). A candidate is skipped ONLY if it is a
                    'fuzzy' token (support==0) that ALSO collides cross-view with an
                    already-kept token (max cos > anc_tau) -> a repeated-texture copy.
                    'sharp' tokens (support>=1) are never dropped (they carry geometry).
      3) coverage : facility-location greedy on G, fills to K. If match_idx is given
                    (xview propagation, §10 safe variant, ON by default) a selected token
                    marks its SHARP cross-view matches as fully covered, so the same 3D
                    point is not re-selected in another view (soft 3D-aware diversity).
    Fully deterministic (no RNG). Budgets are per-bucket caps; any under-fill of buckets
    1-2 flows to coverage so the total is always K."""
    dev = G.device
    B_a, B_s = round(rho_a * K), round(rho_s * K)
    fuzzy = (support == 0)
    sel, selset = [], set()

    # 1) anchor: top-a, NO dedup (keep cross-view multiplicity)
    for i in torch.argsort(a, descending=True).tolist():
        if len(sel) >= B_a or len(sel) >= K:
            break
        if i not in selset:
            sel.append(i); selset.add(i)

    # 2) saliency: margin-aware dedup (drop only fuzzy repeats that collide cross-view)
    cnt = 0
    for i in torch.argsort(s, descending=True).tolist():
        if cnt >= B_s or len(sel) >= K:
            break
        if i in selset:
            continue
        if sel and bool(fuzzy[i]):
            kept = torch.tensor(sel, device=dev)
            other = kept[view_id[kept] != view_id[i]]
            if other.numel() and G[i, other].max() > anc_tau:
                continue                                     # redundant repeated-texture
        sel.append(i); selset.add(i); cnt += 1

    # 3) coverage: facility-location, fill to K
    M = G.shape[0]
    avail = torch.ones(M, dtype=torch.bool, device=dev)
    cov = torch.zeros(M, device=dev)
    if sel:
        kept = torch.tensor(sel, device=dev)
        avail[kept] = False
        cov = G[kept].max(dim=0).values

    def propagate(t):                                        # xview: same 3D point across
        if match_idx is None:                                # views only claims one slot
            return
        mm = match_idx[t]; mm = mm[mm >= 0]
        if mm.numel():
            cov[mm] = 1.0

    for t in sel:                                            # seed from anchor+saliency picks
        propagate(t)
    while len(sel) < K:
        j = _facility_pick(G, cov, avail)
        if j < 0:
            break
        sel.append(j); avail[j] = False; cov = torch.maximum(cov, G[j])
        propagate(j)
    return sel[:K]
