"""CVSP main curve (§8): ACC vs keep_ratio for the five methods, on the SAME
frozen InternVL3 harness (reuses scripts/four_way_extreme.py helpers, so the
prompt/feature/generate plumbing is identical to the §I/J runs).

Methods (all keep the SAME total budget K = keep_pv * n_views, scored by ACC):
  baseline      : all tokens (ratio-independent reference).
  plain_random  : K tokens drawn uniformly from the global pool (per-view varies).
  strat_random  : per-view uniform over 2x2 spatial blocks (strong baseline).
  vispruner     : per-view  top-saliency + ToMe-diverse  (the per-view method).
  cvsp          : three quotas (anchor / saliency / coverage) + per-view floor,
                  cross-view dedup, facility-location coverage. THE method.

CVSP selector (Notes/CVSP-Method.md §5):
  a(t) = cornerness(t) * lowe_max(t)          # cross-view geometric anchor
  s(t) = InternViT CLS->patch attention       # foreground saliency
  1) anchor bucket : top-a, skip if cross-view sim > tau, until B_a = rho_a*K
  2) saliency bucket: top-s, same dedup, until B_s = rho_s*K
  3) coverage      : facility-location greedy (max coverage of the pool), first
                     topping up any view below floor phi, then filling to K.

Hypothesis: curves coincide at keep25/10, separate at keep5/3 (cvsp > random).
Resumable per (ds, task, ratio, method). Logs -> logs/cvsp/.
"""
import argparse, json, math, os, sys

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS_DIR)

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from utils.eval import extract_number_from_answer_tag_mult_choice, extract_number_from_answer_tag_exact_num
from utils.internvl3_utils import prepare_images_internvl, split_model
import models.internvl3_compress as RUN
from compressors.internvl_adapter import AttentionCapture
from compressors.vispruner import VisPrunerCompressor
import four_way_extreme as FW   # cornerness, lowe_max, sel_strat_random, to_perview, build_prompt_var, collect

METHODS = ["baseline", "plain_random", "strat_random", "vispruner", "cvsp", "nuwa", "block_cvsp"]
CONFIGS = [
    ("ego3d", "Object_Centric_Absolute_Distance_MultiChoice"),
    ("ego3d", "Ego_Centric_Absolute_Distance_MultiChoice"),
]


# ---------- saliency cue (InternViT CLS->patch attention -> per-token) ----------
def compute_imp(model, capture, vit):
    """Aggregate captured CLS->patch attention to the LLM-token grid (matches
    compressors/internvl_adapter.compute_visual_features). Returns (M,) float."""
    n_tiles, n_tok, _ = vit.shape
    cls_attn = capture.cls_attn                              # (n_tiles, P)
    hw = int(cls_attn.shape[1] ** 0.5)
    imp = cls_attn.reshape(n_tiles, hw, hw, 1).to(vit.dtype)
    imp = model.pixel_shuffle(imp, scale_factor=model.downsample_ratio)
    imp = imp.mean(dim=-1).reshape(n_tiles, -1)             # (n_tiles, n_tok)
    return imp.reshape(-1).float()                          # (M,)


# ---------- selectors (return list of global indices) ----------
def sel_plain_random(M, K, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(M, generator=g)[:K].tolist()


def sel_vispruner(vit, imp, n_views, n_tok, keep_pv):
    vp = VisPrunerCompressor(important_ratio=0.5)
    out = []
    for v in range(n_views):
        base = v * n_tok
        idx = vp.select(imp[base:base + n_tok], vit[v], keep_pv)   # local indices
        out += (base + idx.to(imp.device)).tolist()
    return out


def _facility_pick(G, cov, avail, vmask=None):
    """One facility-location greedy pick: token maximizing extra coverage of the
    pool, i.e. argmax_t sum_u relu(G[t,u] - cov[u]). vmask restricts candidates."""
    gains = torch.relu(G - cov.unsqueeze(0)).sum(dim=1)      # (M,)
    cand = avail if vmask is None else (avail & vmask)
    gains[~cand] = -1e18
    j = int(gains.argmax())
    return j if bool(cand[j]) else -1


def sel_cvsp(G, a, s, view_id, n_views, n_tok, K, rho_a, rho_s, tau, phi,
             L=None, kappa=0.0):
    """Three-bucket cross-view selection (Notes/CVSP-Method.md §5).
    a, s only used by their within-bucket ranking, so raw scores == rank_norm.
    ★2 anchor reach (budget-relative): if kappa>0, a token is anchor-eligible only
    when its RAW cross-view support L(t) ranks in the top ceil(kappa*B_a). The
    anchor bucket then under-fills when fewer than B_a real anchors exist; the freed
    budget flows to COVERAGE (saliency keeps its own cap B_s). Budget-relative (not a
    fixed quantile) so the SAME setting bites consistently across keep ratios — L has
    no zeros, median ~0.03, so a 0.5 quantile would be inert since B_a/M~rho_a*r."""
    dev = G.device
    B_a, B_s = round(rho_a * K), round(rho_s * K)
    sel, selset = [], set()

    # ★2: anchor eligibility = top ceil(kappa*B_a) tokens by raw cross-view support L
    elig = None
    if L is not None and kappa > 0.0 and B_a > 0:
        pool = min(int(math.ceil(kappa * B_a)), int(L.numel()))
        idx = torch.topk(L.float(), pool).indices
        elig = torch.zeros(int(L.numel()), dtype=torch.bool, device=dev)
        elig[idx] = True

    def red_ok(i):
        if not sel:
            return True
        kept = torch.tensor(sel, device=dev)
        other = kept[view_id[kept] != view_id[i]]
        return not (other.numel() and G[i, other].max() > tau)

    def bucket(score, budget, eligible=None):
        cnt = 0
        for i in torch.argsort(score, descending=True).tolist():
            if cnt >= budget or len(sel) >= K:
                break
            if i in selset or not red_ok(i):
                continue
            if eligible is not None and not bool(eligible[i]):
                continue
            sel.append(i); selset.add(i); cnt += 1

    bucket(a, B_a, elig)                                     # 1) anchor (★2 thresholded)
    bucket(s, B_s)                                           # 2) saliency

    # 3) coverage (facility-location): per-view floor top-up, then fill to K
    M = G.shape[0]
    avail = torch.ones(M, dtype=torch.bool, device=dev)
    cov = torch.zeros(M, device=dev)
    if sel:
        kept = torch.tensor(sel, device=dev)
        avail[kept] = False
        cov = G[kept].max(dim=0).values
    counts = torch.bincount(view_id[torch.tensor(sel, device=dev)], minlength=n_views) \
        if sel else torch.zeros(n_views, dtype=torch.long, device=dev)
    for v in range(n_views):                                # 3a) floor
        vmask = (view_id == v)
        for _ in range(max(0, phi - int(counts[v]))):
            if len(sel) >= K:
                break
            j = _facility_pick(G, cov, avail, vmask)
            if j < 0:
                break
            sel.append(j); avail[j] = False; cov = torch.maximum(cov, G[j])
    while len(sel) < K:                                      # 3b) fill
        j = _facility_pick(G, cov, avail)
        if j < 0:
            break
        sel.append(j); avail[j] = False; cov = torch.maximum(cov, G[j])
    return sel[:K]


def sel_block_cvsp(G, a, s, view_id, n_views, n_tok, K, rho_a, tau, brows, bcols,
                   return_anchor=False):
    """Block-CVSP (Notes/CVSP-Method.md §12).
    Layer 1 (anchors): top round(rho_a*K) by a(t), NO dedup -> preserves cross-view
      anchor pairs. rho_a=0 -> no-anchor ablation.
    Layer 2 (block-stratified saliency, cross-view dedup): each per-view bxb spatial
      block first contributes its top-s dedup-ok token (spatial coverage), then global
      top-s water-fill, then a no-dedup fallback to hit the budget K.
    return_anchor=True -> also return the Layer-1 anchor global indices (subset of sel),
      so a downstream merge can leave anchors pure (§12.9)."""
    dev = G.device
    hw = int(round(n_tok ** 0.5))
    M = n_views * n_tok
    local = torch.arange(M, device=dev) % n_tok
    r, c = local // hw, local % hw
    rb = r // max(1, hw // brows)
    cb = c // max(1, hw // bcols)
    blk = view_id * (brows * bcols) + rb * bcols + cb        # global block id per token

    sel, selset = [], set()

    def red_ok(i):
        if not sel:
            return True
        kept = torch.tensor(sel, device=dev)
        other = kept[view_id[kept] != view_id[i]]
        return not (other.numel() and G[i, other].max() > tau)

    # Layer 1: anchors by a(t), ratio rho_a, NO dedup (keep cross-view pairs)
    anchors = []
    B_anc = round(rho_a * K)
    if B_anc > 0:
        for i in torch.argsort(a, descending=True).tolist():
            if len(sel) >= B_anc or len(sel) >= K:
                break
            if i not in selset:
                sel.append(i); selset.add(i); anchors.append(i)
    covered = set(int(blk[i]) for i in sel)

    # tokens grouped per block, saliency-descending
    order_s = torch.argsort(s, descending=True).tolist()
    blk_tok = {}
    for i in order_s:
        blk_tok.setdefault(int(blk[i]), []).append(i)

    # Layer 2a: round 1 -- one per uncovered block, blocks ordered by their top saliency
    uncovered = sorted((b for b in blk_tok if b not in covered),
                       key=lambda b: -float(s[blk_tok[b][0]]))
    for b in uncovered:
        if len(sel) >= K:
            break
        for i in blk_tok[b]:
            if i in selset or not red_ok(i):
                continue
            sel.append(i); selset.add(i)
            break
    # Layer 2b: rounds 2+ -- global top-s with dedup
    if len(sel) < K:
        for i in order_s:
            if len(sel) >= K:
                break
            if i in selset or not red_ok(i):
                continue
            sel.append(i); selset.add(i)
    # Layer 2c: fallback -- fill to K ignoring dedup (equal-budget comparability)
    if len(sel) < K:
        for i in order_s:
            if len(sel) >= K:
                break
            if i not in selset:
                sel.append(i); selset.add(i)
    sel = sel[:K]
    if return_anchor:
        return sel, [i for i in anchors if i in set(sel)]
    return sel


def apply_saliency_merge(vit, G, view_id, sel, anchors, n_tok, dist_thr):
    """Nüwa-style intra-view local merge (Notes/CVSP-Method.md §12.9 B). Each kept
    SALIENCY token (sel minus anchors) gathers same-view DROPPED tokens weighted by
    relu(cos) * spatial-distance-decay; anchors are left pure (not targets). Returns a
    merged (n_views, n_tok, C) tensor in vit.dtype; only saliency-token features change.
    cos reuses G (vit-feature cosine); no extra forward pass."""
    dev = G.device
    n_views, _, C = vit.shape
    M = n_views * n_tok
    flat = vit.reshape(M, C).float()
    merged = flat.clone()
    hw = int(round(n_tok ** 0.5))
    rr = (torch.arange(n_tok, device=dev) // hw).float()
    cc = (torch.arange(n_tok, device=dev) % hw).float()
    pos = torch.stack([rr, cc], dim=-1)
    distpen = (1.0 - torch.cdist(pos, pos) / dist_thr).clamp(min=0.0)   # (n_tok, n_tok)
    dropped = torch.ones(M, dtype=torch.bool, device=dev)
    if sel:
        dropped[torch.tensor(sel, device=dev)] = False
    ancset = set(anchors)
    loc = torch.arange(n_tok, device=dev)
    for i in sel:
        if i in ancset:                                    # leave anchors pure
            continue
        v = int(view_id[i]); base = v * n_tok; li = i - base
        cand = loc[dropped[base:base + n_tok]]             # same-view dropped (local idx)
        if cand.numel() == 0:
            continue
        w = torch.relu(G[i, base + cand]) * distpen[li, cand]
        sw = float(w.sum())
        if sw <= 0:
            continue
        merged[i] = (flat[i] + (w.unsqueeze(1) * flat[base + cand]).sum(0)) / (1.0 + sw)
    return merged.reshape(vit.shape).to(vit.dtype)


def sel_nuwa(imp, view_id, n_views, n_tok, keep_pv, lam, dist_thr):
    """Nuwa-lite (Notes/CVSP-Method.md §12.5): per-view greedy selection maximizing
    CLS-attention importance minus a spatial proximity penalty to already-picked tokens
    (encourages spatially spread-out salient tokens). Captures Nuwa stage-1's essence
    (attention + distance penalty); omits its merge/percentile refinements."""
    dev = imp.device
    hw = int(round(n_tok ** 0.5))
    rr = (torch.arange(n_tok, device=dev) // hw).float()
    cc = (torch.arange(n_tok, device=dev) % hw).float()
    pos = torch.stack([rr, cc], dim=-1)                      # (n_tok, 2)
    out = []
    for v in range(n_views):
        base = v * n_tok
        impv = imp[base:base + n_tok].clone().float()
        penalty = torch.zeros(n_tok, device=dev)
        chosen = []
        for _ in range(min(keep_pv, n_tok)):
            score = impv - lam * penalty
            if chosen:
                score[torch.tensor(chosen, device=dev)] = -1e30
            j = int(score.argmax())
            chosen.append(j)
            d = torch.cdist(pos, pos[j:j + 1]).squeeze(1)
            penalty = torch.maximum(penalty, torch.clamp(1.0 - d / dist_thr, min=0.0))
        out += [base + j for j in chosen]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--ratios", default="0.25,0.1,0.05,0.03")
    ap.add_argument("--methods", default="")               # csv subset; "" = all
    ap.add_argument("--configs", default="")               # csv ds:task; "" = default
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    ap.add_argument("--vsi_items", default="data/vsibench/vsibench_items.json")
    ap.add_argument("--rho_a", type=float, default=0.4)
    ap.add_argument("--rho_s", type=float, default=0.3)     # rho_c = 1 - rho_a - rho_s
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--phi", type=int, default=1)
    ap.add_argument("--kappa", type=float, default=0.0)     # ★2 anchor reach: pool=top ceil(kappa*B_a) by L (0 = off)
    ap.add_argument("--tag", default="")                    # filename suffix for cvsp/block_cvsp variant runs
    ap.add_argument("--brows", type=int, default=2)         # block_cvsp: spatial block rows per view
    ap.add_argument("--bcols", type=int, default=2)         # block_cvsp: spatial block cols per view
    ap.add_argument("--nuwa_lam", type=float, default=1.0)  # nuwa-lite: distance-penalty weight
    ap.add_argument("--nuwa_dist", type=float, default=11.0)# nuwa-lite: distance threshold (16x16 grid)
    ap.add_argument("--merge", default="none", choices=["none", "sal"])  # §12.9: merge dropped->saliency tokens
    ap.add_argument("--merge_dist", type=float, default=11.0)# merge spatial distance threshold (16x16 grid)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    methods = args.methods.split(",") if args.methods else METHODS
    configs = [tuple(c.split(":")) for c in args.configs.split(",")] if args.configs else CONFIGS

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(RUN.IMG_CTX)
    capture = AttentionCapture(model)                       # saliency cue
    ego_ds = load_dataset("vbdai/Ego3D-Bench")["test"]
    os.makedirs("logs/cvsp", exist_ok=True)

    def select(method, vit, imp, G, view_id, n_views, n_tok, ratio, seen):
        M = n_views * n_tok
        anc = []                                            # Layer-1 anchors (block_cvsp only)
        if method == "baseline":
            gi = list(range(M))
        else:
            keep_pv = max(1, round(n_tok * ratio)); K = keep_pv * n_views
            if method == "plain_random":
                gi = sel_plain_random(M, K, args.seed + seen)
            elif method == "strat_random":
                gi = FW.sel_strat_random(view_id, n_views, n_tok, keep_pv, args.seed + seen)
            elif method == "vispruner":
                gi = sel_vispruner(vit, imp, n_views, n_tok, keep_pv)
            elif method == "cvsp":
                L = FW.lowe_max(G, view_id, n_views)
                a = FW.cornerness(G, view_id, n_views) * L
                gi = sel_cvsp(G, a, imp, view_id, n_views, n_tok, K,
                              args.rho_a, args.rho_s, args.tau, args.phi,
                              L=L, kappa=args.kappa)
            elif method == "nuwa":
                gi = sel_nuwa(imp, view_id, n_views, n_tok, keep_pv, args.nuwa_lam, args.nuwa_dist)
            elif method == "block_cvsp":
                a = FW.cornerness(G, view_id, n_views) * FW.lowe_max(G, view_id, n_views)
                gi, anc = sel_block_cvsp(G, a, imp, view_id, n_views, n_tok, K,
                                         args.rho_a, args.tau, args.brows, args.bcols,
                                         return_anchor=True)
            else:
                raise ValueError(method)
        return FW.to_perview(gi, view_id, n_views, n_tok), gi, anc

    def run(method, ratio, save, samples):
        done = sum(1 for _ in open(save)) if os.path.exists(save) else 0
        for seen, (q, paths, gt) in enumerate(samples, 1):
            if seen <= done:
                continue
            pv, _ = prepare_images_internvl(paths)
            vit = model.extract_feature(pv)
            n_views, n_tok, C = vit.shape
            imp = compute_imp(model, capture, vit)
            Fn = torch.nn.functional.normalize(vit.reshape(-1, C).float(), dim=-1)
            G = Fn @ Fn.t()
            view_id = torch.arange(Fn.shape[0], device=Fn.device) // n_tok
            per_view, sel_g, anc_g = select(method, vit, imp, G, view_id, n_views, n_tok, ratio or 1.0, seen)
            mvit = vit
            if args.merge == "sal" and sel_g:                # §12.9: merge dropped -> saliency tokens
                mvit = apply_saliency_merge(vit, G, view_id, sel_g, anc_g, n_tok, args.merge_dist)
            feats = torch.cat([mvit[t][per_view[t]] for t in range(n_views)], dim=0)
            counts = [int(len(per_view[t])) for t in range(n_views)]
            mi, eos, tmpl = FW.build_prompt_var(model, tokenizer, q, counts)
            ids = mi["input_ids"].cuda()
            assert int((ids == model.img_context_token_id).sum()) == feats.shape[0]
            gen = model.generate(pixel_values=pv, input_ids=ids,
                                 attention_mask=mi["attention_mask"].cuda(), visual_features=feats,
                                 max_new_tokens=1024, do_sample=False, eos_token_id=eos)
            resp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].split(tmpl.sep.strip())[0].strip()
            proc = resp.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()
            with open(save, "a") as f:
                f.write(json.dumps({"Processed_Pred": proc, "GT": gt, "n_kept": int(feats.shape[0])}) + "\n")

    def summarize(dsname, task):
        is_num = task in FW.NUM_CATS
        def metric_of(path):
            rows = [json.loads(l) for l in open(path)]
            tok = sum(r["n_kept"] for r in rows) / len(rows)
            if is_num:
                yt, yp = [], []
                for r in rows:
                    pr = extract_number_from_answer_tag_exact_num(r["Processed_Pred"])
                    if pr is not None:
                        yt.append(float(r["GT"])); yp.append(min(float(pr), 100.0))
                v = (sum((a - b) ** 2 for a, b in zip(yt, yp)) / len(yt)) ** 0.5 if yt else float("nan")
            else:
                v = sum(1 for r in rows if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower()) / len(rows)
            return v, tok, len(rows)
        lab = "RMSE" if is_num else "ACC"
        print(f"\n=== {dsname}/{task} ({lab}) ===", flush=True)
        bl = f"logs/cvsp/{dsname}.{task}.baseline.jsonl"
        if os.path.exists(bl):
            a, t, n = metric_of(bl); print(f"  baseline                {lab}={a:.4f}  tok={t:.0f}  n={n}")
        for ratio in ratios:
            pct = int(round(ratio * 100)); print(f"  -- keep{pct}% --")
            for method in methods:
                if method == "baseline":
                    continue
                p = f"logs/cvsp/{dsname}.{task}.keep{pct}.{method}{args.tag if method in ('cvsp', 'block_cvsp') else ''}.jsonl"
                if os.path.exists(p):
                    a, t, n = metric_of(p); print(f"  {method:<15} keep{pct:<3} {lab}={a:.4f}  tok={t:.0f}  n={n}", flush=True)

    for dsname, task in configs:
        samples = FW.collect(dsname, task, args.n, args.image_root, args.vsi_items, ego_ds)
        for method in methods:
            for ratio in ([None] if method == "baseline" else ratios):
                mlabel = method + args.tag if method in ("cvsp", "block_cvsp") else method  # tag suffixes cvsp/block_cvsp variants
                seg = "baseline" if method == "baseline" else f"keep{int(round(ratio * 100))}.{mlabel}"
                save = f"logs/cvsp/{dsname}.{task}.{seg}.jsonl"
                run(method, ratio, save, samples)
                print(f"[{dsname}/{task} {seg}] done", flush=True)
        summarize(dsname, task)   # readable ACC table as soon as a config finishes


if __name__ == "__main__":
    main()
