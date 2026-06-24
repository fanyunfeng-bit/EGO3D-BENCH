"""The "体检" (go/no-go diagnostic): does the multi-view token pool have
EXPLOITABLE non-uniform structure ("a few important tokens"), or is it flat
(in which case uniform random is provably near-optimal and no training-free
selector can beat it)?

For each scene we take the post-encoder LLM tokens F (M x d), unit-normalize,
form the token-token cosine Gram G = Fn Fn^T, and measure the SHAPE of its
spectrum / leverage distribution:
  erank     : effective rank = exp(spectral entropy of G's eigenvalues).
              Low (<< M) = tokens live in a low-dim subspace = redundant/structured.
  gini      : Gini of ridge-leverage scores (token "uniqueness"). High = spiky =
              a few tokens carry most of the structure = exploitable.
  coherence : max(leverage)/mean(leverage). High = spiky.
Three references per scene: REAL pool, a same-shape random-Gaussian NULL
("no structure" yardstick), and a SINGLE view (to see if the structure is
cross-view). Bonus: correlation(leverage, anchor-score) — are high-leverage
tokens the geometric anchors? And fraction of tokens with non-zero anchor.

No LLM generation. Ego3D + VSI.
"""
import argparse, json, os, sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from utils.internvl3_utils import prepare_images_internvl, split_model
import models.internvl3_compress as RUN


def gini(x):
    x = x.clamp(min=0).sort().values
    n = x.shape[0]
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return (((2 * idx - n - 1) * x).sum() / (n * x.sum() + 1e-12)).item()


@torch.no_grad()
def spectral_metrics(Fn, keep_frac=0.10):
    G = Fn @ Fn.t()
    M = G.shape[0]
    eig = torch.linalg.eigvalsh(G).clamp(min=0)
    p = eig / (eig.sum() + 1e-12)
    erank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    K = max(1, round(keep_frac * M))
    lam = eig.sort(descending=True).values[K - 1].item() + 1e-6
    inv = torch.linalg.inv(G + lam * torch.eye(M, device=G.device, dtype=G.dtype))
    lev = (1 - lam * torch.diagonal(inv)).clamp(min=0)
    deff = (eig / (eig + lam)).sum().item()
    return dict(M=M, erank=erank, erank_frac=erank / M,
                gini=gini(lev), coherence=(lev.max() / (lev.mean() + 1e-12)).item(),
                deff=deff), lev


@torch.no_grad()
def anchor_scores(G, view_id, n_views):
    M = G.shape[0]
    anchor = torch.zeros(M, device=G.device)
    for v in range(n_views):
        rows = torch.where(view_id == v)[0]
        if rows.numel() == 0:
            continue
        best = torch.zeros(rows.numel(), device=G.device)
        for u in range(n_views):
            if u == v:
                continue
            cols = torch.where(view_id == u)[0]
            if cols.numel() < 2:
                continue
            top2 = G[rows][:, cols].topk(2, dim=1).values
            best = torch.maximum(best, (top2[:, 0] - top2[:, 1]).clamp(min=0) * top2[:, 0])
        anchor[rows] = best
    return anchor


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def ego3d_samples(n, image_root):
    ds = load_dataset("vbdai/Ego3D-Bench")["test"]
    out = []
    for s in ds:
        order = RUN.IMAGE_ORDER.get(s["source"], list(s["images"].keys()))
        out.append([os.path.join(image_root, s["images"][v]) for v in order])
        if len(out) >= n:
            break
    return out


def vsi_samples(n, items_path):
    items = json.load(open(items_path))
    root = os.path.dirname(items_path)
    out = []
    seen = set()
    for it in items:
        key = tuple(it["frames"])
        if key in seen:
            continue
        seen.add(key)
        out.append([os.path.join(root, f) for f in it["frames"]])
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    ap.add_argument("--vsi_items", default="data/vsibench/vsibench_items.json")
    ap.add_argument("--out", default="logs/leverage_diagnostic.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    datasets = {"ego3d": ego3d_samples(args.n, args.image_root),
                "vsi": vsi_samples(args.n, args.vsi_items)}
    results = {}
    for name, samples in datasets.items():
        agg = {"real": [], "null": [], "single": [], "corr_lev_anchor": [], "anchor_nonzero_frac": []}
        for paths in samples:
            pv, _ = prepare_images_internvl(paths)
            vit = model.extract_feature(pv)
            n_views, n_tok, C = vit.shape
            Fn = torch.nn.functional.normalize(vit.reshape(-1, C).float(), dim=-1)
            view_id = torch.arange(Fn.shape[0], device=Fn.device) // n_tok
            real, lev = spectral_metrics(Fn)
            null_F = torch.nn.functional.normalize(torch.randn_like(Fn), dim=-1)
            null, _ = spectral_metrics(null_F)
            single, _ = spectral_metrics(Fn[view_id == 0])
            G = Fn @ Fn.t()
            anc = anchor_scores(G, view_id, n_views)
            agg["real"].append(real); agg["null"].append(null); agg["single"].append(single)
            agg["corr_lev_anchor"].append(pearson(lev, anc))
            agg["anchor_nonzero_frac"].append((anc > 1e-4).float().mean().item())
        def mean_of(key, sub):
            return sum(d[sub] for d in agg[key]) / len(agg[key])
        results[name] = {
            "n": len(samples),
            "real":   {k: mean_of("real", k) for k in ["M", "erank", "erank_frac", "gini", "coherence", "deff"]},
            "null":   {k: mean_of("null", k) for k in ["erank", "erank_frac", "gini", "coherence"]},
            "single": {k: mean_of("single", k) for k in ["M", "erank", "erank_frac", "gini", "coherence"]},
            "corr_lev_anchor": sum(agg["corr_lev_anchor"]) / len(agg["corr_lev_anchor"]),
            "anchor_nonzero_frac": sum(agg["anchor_nonzero_frac"]) / len(agg["anchor_nonzero_frac"]),
        }
        print(f"[{name}] done {len(samples)} samples", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print("\n==== DIAGNOSTIC (real vs null vs single-view) ====")
    for name, r in results.items():
        print(f"\n### {name} (n={r['n']}, M≈{r['real']['M']:.0f})")
        print(f"  erank_frac  real={r['real']['erank_frac']:.3f}  null={r['null']['erank_frac']:.3f}  single={r['single']['erank_frac']:.3f}   (low real & << null = redundant/structured)")
        print(f"  gini(lev)   real={r['real']['gini']:.3f}  null={r['null']['gini']:.3f}  single={r['single']['gini']:.3f}   (real >> null = spiky = exploitable)")
        print(f"  coherence   real={r['real']['coherence']:.2f}  null={r['null']['coherence']:.2f}")
        print(f"  corr(leverage, anchor) = {r['corr_lev_anchor']:.3f}   (high = high-leverage tokens ARE anchors)")
        print(f"  anchor_nonzero_frac    = {r['anchor_nonzero_frac']:.3f}   (how many tokens get a landmark score)")


if __name__ == "__main__":
    main()
