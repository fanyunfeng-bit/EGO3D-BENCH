"""P4 (causal): does an EXPLICIT low-redundancy / high-coverage kept set actually
beat random on accuracy at the same token budget?

Adds a third method, global FPS (farthest-point sampling over the pooled
multi-view tokens, with a >=1-token-per-view floor), which by construction
minimizes cross-view redundancy and maximizes coverage. We run vispruner /
random / fps on the SAME samples at the same total budget (= keep10), measure
each method's R & C, and score accuracy. If fps (lowest R) > random > vispruner
on accuracy, the chain "lower cross-view redundancy -> higher accuracy" is causal
and there is headroom below random (so a method CAN beat random).

Resumable: per (task, method) jsonl, skips written lines.
"""
import argparse, json, os, sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from utils.eval import extract_number_from_answer_tag_mult_choice
from utils.internvl3_utils import prepare_images_internvl, split_model
from compressors import build_compressor
from compressors.internvl_adapter import AttentionCapture
import models.internvl3_compress as RUN

METHODS = ["vispruner", "random", "fps"]


def importance_per_tile(model, capture, n_tiles):
    cls_attn = capture.cls_attn
    hw = int(cls_attn.shape[1] ** 0.5)
    imp = cls_attn.reshape(n_tiles, hw, hw, 1).float()
    imp = model.pixel_shuffle(imp, scale_factor=model.downsample_ratio)
    return imp.mean(dim=-1).reshape(n_tiles, -1)


def global_fps_perview(Fn, view_id, n_tok, n_views, total_budget):
    """FPS over the pooled tokens, seeded with one token per view (>=1/view floor).
    Returns per-view sorted local indices + the global selected index tensor."""
    N = Fn.shape[0]
    dev = Fn.device
    selected = [v * n_tok for v in range(n_views)]          # first token of each view
    sel_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    sel_mask[selected] = True
    S = Fn[selected]
    mind = (1 - Fn @ S.T).min(dim=1).values                 # min dist to seed set
    mind[sel_mask] = -1
    for _ in range(total_budget - n_views):
        cur = int(mind.argmax())
        selected.append(cur); sel_mask[cur] = True
        mind = torch.minimum(mind, 1 - Fn @ Fn[cur]); mind[sel_mask] = -1
    sel = torch.tensor(selected, device=dev)
    vid = view_id[sel]
    per_view = [(sel[vid == v] % n_tok).sort().values for v in range(n_views)]
    return per_view, sel


def build_prompt_var(model, tokenizer, question, per_view_counts):
    """Like RUN.build_prompt but with a per-view (variable) IMG_CONTEXT count.
    Relies on the question carrying one <image> marker per view (Ego3D convention)."""
    get_conv = sys.modules[type(model).__module__].get_conv_template
    if "<image>" not in question:
        question = "<image>\n" + question
    tmpl = get_conv(model.template); tmpl.system_message = model.system_message
    eos = tokenizer.convert_tokens_to_ids(tmpl.sep.strip())
    tmpl.append_message(tmpl.roles[0], question); tmpl.append_message(tmpl.roles[1], None)
    query = tmpl.get_prompt()
    for cnt in per_view_counts:
        query = query.replace("<image>", RUN.IMG_START + RUN.IMG_CTX * int(cnt) + RUN.IMG_END, 1)
    return tokenizer(query, return_tensors="pt"), eos, tmpl


def metrics(Fn, kept_global, view_id):
    K = Fn[kept_global]; vk = view_id[kept_global]
    sim = K @ K.T
    R = sim.masked_fill(vk[:, None] == vk[None, :], -2.0).max(dim=1).values.clamp(-1, 1).mean().item()
    km = torch.zeros(Fn.shape[0], dtype=torch.bool, device=Fn.device); km[kept_global] = True
    disc = Fn[~km]
    C = 1.0 if disc.shape[0] == 0 else (disc @ K.T).max(dim=1).values.mean().item()
    return R, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--tasks", default="Object_Centric_Absolute_Distance_MultiChoice,"
                                       "Ego_Centric_Absolute_Distance_MultiChoice")
    ap.add_argument("--keep_ratio", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    args = ap.parse_args()
    tasks = args.tasks.split(",")

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(RUN.IMG_CTX)
    capture = AttentionCapture(model)
    visp = build_compressor("vispruner", important_ratio=0.5)
    rand = build_compressor("random")
    import numpy as np
    dataset = load_dataset("vbdai/Ego3D-Bench")["test"]
    os.makedirs("logs/p4", exist_ok=True)

    def kept_perview(method, vit, imp, keep, n_tok, n_views, view_id, Fn, seen):
        if method == "fps":
            pv, sel = global_fps_perview(Fn, view_id, n_tok, n_views, keep * n_views)
            return pv, sel
        locs = []
        for t in range(n_views):
            if method == "random":
                s = int(np.random.SeedSequence([args.seed, seen, t]).generate_state(1)[0])
                idx = rand.select(None, vit[t], keep, seed=s)
            else:
                idx = visp.select(imp[t], vit[t], keep, seed=None)
            locs.append(torch.as_tensor(idx, device=vit.device, dtype=torch.long).sort().values)
        glob = torch.cat([t * n_tok + locs[t] for t in range(n_views)])
        return locs, glob

    for task in tasks:
        for method in METHODS:
            save = f"logs/p4/{task}.{method}.jsonl"
            done = sum(1 for _ in open(save)) if os.path.exists(save) else 0
            seen = 0
            for sample in dataset:
                if sample["category"] != task:
                    continue
                seen += 1
                if seen <= done:
                    continue
                if seen > args.n:
                    break
                q = sample["question"]
                for opt in (sample["options"] or []):
                    q += "\n" + opt
                q += ("\nOutput the thinking process in <think> </think> and final answer "
                      "(only the letter of the choice) in <answer> </answer> tags.")
                order = RUN.IMAGE_ORDER.get(sample["source"], list(sample["images"].keys()))
                paths = [os.path.join(args.image_root, sample["images"][v]) for v in order]
                pv, _ = prepare_images_internvl(paths)
                vit = model.extract_feature(pv)
                n_tiles, n_tok, C = vit.shape
                imp = importance_per_tile(model, capture, n_tiles)
                keep = max(1, round(n_tok * args.keep_ratio))
                Fn = torch.nn.functional.normalize(vit.reshape(-1, C).float(), dim=-1)
                view_id = torch.arange(Fn.shape[0], device=Fn.device) // n_tok

                per_view, glob = kept_perview(method, vit, imp, keep, n_tok, n_tiles, view_id, Fn, seen)
                R, Cc = metrics(Fn, glob, view_id)
                feats = torch.cat([vit[t][per_view[t]] for t in range(n_tiles)], dim=0)
                counts = [int(len(per_view[t])) for t in range(n_tiles)]
                mi, eos, tmpl = build_prompt_var(model, tokenizer, q, counts)
                ids = mi["input_ids"].cuda()
                assert int((ids == model.img_context_token_id).sum()) == feats.shape[0]
                gen = model.generate(pixel_values=pv, input_ids=ids,
                                     attention_mask=mi["attention_mask"].cuda(), visual_features=feats,
                                     max_new_tokens=1024, do_sample=False, eos_token_id=eos)
                resp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].split(tmpl.sep.strip())[0].strip()
                proc = resp.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()
                with open(save, "a") as f:
                    f.write(json.dumps({"Processed_Pred": proc, "GT": sample["answer"],
                                        "R": R, "C": Cc, "n_kept": int(feats.shape[0])}) + "\n")
            print(f"[{task}/{method}] done", flush=True)

    print("\n=== P4 RESULTS (ACC, chance=0.25) + mean R/C ===")
    for task in tasks:
        print(f"\n### {task}")
        for method in METHODS:
            save = f"logs/p4/{task}.{method}.jsonl"
            if not os.path.exists(save):
                continue
            rows = [json.loads(l) for l in open(save)]
            acc = sum(1 for r in rows if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower()) / len(rows)
            mR = sum(r["R"] for r in rows) / len(rows); mC = sum(r["C"] for r in rows) / len(rows)
            print(f"  {method:<10} ACC={acc:.4f}  R={mR:.3f} C={mC:.3f}  n={len(rows)}")


if __name__ == "__main__":
    main()
