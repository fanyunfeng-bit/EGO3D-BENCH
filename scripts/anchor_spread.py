"""Measure how concentrated Block-CVSP Layer-1 anchors are ACROSS VIEWS.

No LLM generate -- only ViT features + G + a(t)=cornerness*lowe_max, then replicate
sel_block_cvsp Layer 1 (global top-B_anc by a, rho_a=0.2). For each sample/ratio report,
averaged over samples, vs a random-same-count null:
  cover   = (#views with >=1 anchor)/n_views          (1.0 = every view represented)
  maxfrac = (max anchors in one view)/B_anc           (1/n_views = perfectly even)
  ent     = view-distribution entropy / log(n_views)   (1.0 = uniform, 0 = all one view)
  pair    = frac of anchors whose best cross-view partner is ALSO selected (implicit pairing)
Run under ego3d env. Fast (~few min): no generation.
"""
import argparse, json, math, os, sys

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, SCRIPTS_DIR)

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from utils.internvl3_utils import prepare_images_internvl, split_model
import four_way_extreme as FW


def stats(view_id, idx, n_views, B, G):
    """cover / maxfrac / entropy / pair for a selected index set idx (1-D long tensor)."""
    vc = torch.bincount(view_id[idx], minlength=n_views).float()
    cover = float((vc > 0).sum()) / n_views
    maxfrac = float(vc.max()) / B
    p = vc / B; nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum() / math.log(n_views)) if n_views > 1 else 1.0
    # pairing: best partner of each anchor in OTHER views also selected?
    sel = set(idx.tolist()); paired = 0
    for t in idx.tolist():
        row = G[t].clone(); row[view_id == view_id[t]] = -2.0
        if int(row.argmax()) in sel:
            paired += 1
    return cover, maxfrac, ent, paired / B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--configs", default="vsi:object_rel_direction_hard,vsi:object_rel_distance,ego3d:Localization")
    ap.add_argument("--ratios", default="0.1,0.05,0.03")
    ap.add_argument("--rho_a", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    ap.add_argument("--vsi_items", default="data/vsibench/vsibench_items.json")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    configs = [tuple(c.split(":")) for c in args.configs.split(",")]

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    ego_ds = load_dataset("vbdai/Ego3D-Bench")["test"]

    for dsname, task in configs:
        samples = FW.collect(dsname, task, args.n, args.image_root, args.vsi_items, ego_ds)
        # acc[ratio] = [B, nv, cover_a, maxfrac_a, ent_a, pair_a, cover_r, maxfrac_r, ent_r], summed
        agg = {r: [0.0] * 9 + [0] for r in ratios}  # last = sample count
        for q, paths, gt in samples:
            pv, _ = prepare_images_internvl(paths)
            vit = model.extract_feature(pv)
            n_views, n_tok, C = vit.shape
            if n_views < 2:
                continue
            Fn = torch.nn.functional.normalize(vit.reshape(-1, C).float(), dim=-1)
            G = Fn @ Fn.t()
            M = n_views * n_tok
            view_id = torch.arange(M, device=G.device) // n_tok
            a = FW.cornerness(G, view_id, n_views) * FW.lowe_max(G, view_id, n_views)
            for r in ratios:
                keep_pv = max(1, round(n_tok * r)); K = keep_pv * n_views
                B = round(args.rho_a * K)
                if B < 1:
                    continue
                anc = torch.topk(a, B).indices
                ca, ma, ea, pa = stats(view_id, anc, n_views, B, G)
                g = torch.Generator(device=G.device).manual_seed(args.seed + agg[r][9])
                rnd = torch.randperm(M, generator=g, device=G.device)[:B]
                cr, mr, er, _ = stats(view_id, rnd, n_views, B, G)
                v = agg[r]
                v[0] += B; v[1] += n_views; v[2] += ca; v[3] += ma; v[4] += ea
                v[5] += pa; v[6] += cr; v[7] += mr; v[8] += er; v[9] += 1

        print(f"\n=== {dsname}/{task}  (rho_a={args.rho_a}) ===", flush=True)
        print(f"  {'ratio':<7}{'B':>4}{'nv':>5} | {'cover':>6}{'maxfr':>7}{'ent':>6}{'pair':>6} |"
              f" {'cover':>7}{'maxfr':>7}{'ent':>6}  (random null)")
        for r in ratios:
            v = agg[r]; c = v[9]
            if c == 0:
                continue
            print(f"  keep{int(round(r*100)):<3}{v[0]/c:>4.0f}{v[1]/c:>5.1f} | "
                  f"{v[2]/c:>6.2f}{v[3]/c:>7.2f}{v[4]/c:>6.2f}{v[5]/c:>6.2f} | "
                  f"{v[6]/c:>7.2f}{v[7]/c:>7.2f}{v[8]/c:>6.2f}  (n={c})", flush=True)


if __name__ == "__main__":
    main()
