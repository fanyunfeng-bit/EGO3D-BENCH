"""P1-P3 补证: do the tokens VisPruner keeps have higher cross-view redundancy
and worse coverage than the tokens random keeps?

For each sample we encode the views, get the post-encoder LLM tokens (the space
the LLM actually sees), then for each method compute its kept index set and two
threshold-free metrics:

  R (cross-view redundancy) = mean over kept tokens of (max cosine sim to kept
     tokens from OTHER views). High = the kept set is full of cross-view copies.
  C (coverage of discarded) = mean over DISCARDED tokens of (max cosine sim to
     any kept token). High = the kept set still represents what was thrown away.

No LLM generation here (cheap). Accuracy is read from the existing keep10 runs.
"""
import argparse, json, os, sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from utils.internvl3_utils import prepare_images_internvl, split_model
from compressors import build_compressor
from compressors.internvl_adapter import AttentionCapture
import models.internvl3_compress as RUN


def importance_per_tile(model, capture, n_tiles):
    cls_attn = capture.cls_attn                         # (T, P)
    hw = int(cls_attn.shape[1] ** 0.5)
    imp = cls_attn.reshape(n_tiles, hw, hw, 1).float()
    imp = model.pixel_shuffle(imp, scale_factor=model.downsample_ratio)
    return imp.mean(dim=-1).reshape(n_tiles, -1)         # (T, num_image_token)


@torch.no_grad()
def kept_global_indices(method, comp, vit_embeds, imp, keep, n_tok, base_seed, sample_id):
    import numpy as np
    glob = []
    for t in range(vit_embeds.shape[0]):
        if method == "random":
            seed = int(np.random.SeedSequence([base_seed, sample_id, t]).generate_state(1)[0])
            idx = comp.select(None, vit_embeds[t], keep, seed=seed)
        else:
            idx = comp.select(imp[t], vit_embeds[t], keep, seed=None)
        idx = torch.as_tensor(idx, device=vit_embeds.device, dtype=torch.long)
        glob.append(t * n_tok + idx)
    return torch.cat(glob)


@torch.no_grad()
def metrics(Fn, kept_idx, view_id):
    K = Fn[kept_idx]                                     # (k, C)
    vk = view_id[kept_idx]
    sim = K @ K.T                                        # cosine (normalized)
    same = vk[:, None] == vk[None, :]
    R = sim.masked_fill(same, -2.0).max(dim=1).values.clamp(-1, 1).mean().item()
    keep_mask = torch.zeros(Fn.shape[0], dtype=torch.bool, device=Fn.device)
    keep_mask[kept_idx] = True
    disc = Fn[~keep_mask]
    C = 1.0 if disc.shape[0] == 0 else (disc @ K.T).max(dim=1).values.mean().item()
    return R, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--tasks", default="Object_Centric_Absolute_Distance_MultiChoice,"
                                       "Ego_Centric_Absolute_Distance_MultiChoice,"
                                       "Travel_Time,Localization")
    ap.add_argument("--keep_ratio", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    ap.add_argument("--out", default="logs/redundancy_analysis.jsonl")
    args = ap.parse_args()
    tasks = args.tasks.split(",")

    torch.set_grad_enabled(False)  # CRITICAL: .eval() alone keeps grad -> ViT activations OOM
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    capture = AttentionCapture(model)
    visp = build_compressor("vispruner", important_ratio=0.5)
    rand = build_compressor("random")
    dataset = load_dataset("vbdai/Ego3D-Bench")["test"]

    rows = []
    for task in tasks:
        seen = 0
        for sample in dataset:
            if sample["category"] != task:
                continue
            seen += 1
            if seen > args.n:
                break
            order = RUN.IMAGE_ORDER.get(sample["source"], list(sample["images"].keys()))
            paths = [os.path.join(args.image_root, sample["images"][v]) for v in order]
            pv, _ = prepare_images_internvl(paths)
            vit = model.extract_feature(pv)                 # (T, n_tok, C)
            n_tiles, n_tok, C = vit.shape
            imp = importance_per_tile(model, capture, n_tiles)
            keep = max(1, round(n_tok * args.keep_ratio))
            F = vit.reshape(-1, C).float()
            Fn = torch.nn.functional.normalize(F, dim=-1)
            view_id = (torch.arange(Fn.shape[0], device=Fn.device) // n_tok)
            for method, comp in [("vispruner", visp), ("random", rand)]:
                ki = kept_global_indices(method, comp, vit, imp, keep, n_tok, args.seed, seen)
                Rv, Cv = metrics(Fn, ki, view_id)
                rows.append({"task": task, "method": method, "sample": seen,
                             "R": Rv, "C": Cv, "n_kept": int(ki.numel()), "n_tot": int(Fn.shape[0])})
        print(f"[{task}] done {min(seen, args.n)} samples", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
