"""InternVL3 + visual-token compression on VSI-Bench.

Same harness/methods as the Ego3D runner (baseline | vispruner | random), but on
VSI-Bench: each video is represented by `--frames` uniformly-sampled frames
(prepped by scripts/prep_vsibench.py), presented to the model as `View N:` images
with our R1-style CoT prompt. Loads the model ONCE and loops over `--category`
(a comma list, or "all") so the small-sample sweep over all 10 types is cheap.

Metrics (VSI-Bench): multiple-choice -> Accuracy; numeric -> Mean Relative
Accuracy (MRA = mean over thresholds {0.5..0.95} of 1[|pred-gt|/gt < 1-theta]).
Performance + efficiency are recorded per task:
  logs/<model>-<tag>-vsibench/<question_type>.{jsonl,result.json}
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import (extract_number_from_answer_tag_mult_choice,
                        extract_number_from_answer_tag_exact_num)
from utils.internvl3_utils import prepare_images_internvl, split_model
from utils import efficiency
from compressors import build_compressor, scm
from compressors.internvl_adapter import AttentionCapture, compute_visual_features

# methods not in the per-image registry: cross-view SCMPruner (pre-LLM) + in-LLM FastV
SPECIAL_METHODS = {"scmpruner", "fastv"}

torch.manual_seed(42)

# numeric (Mean-Relative-Accuracy) tasks; everything else is multiple-choice (Accuracy)
NA_TYPES = {"object_size_estimation", "object_abs_distance",
            "object_counting", "room_size_estimation"}
ALL_TYPES = sorted(NA_TYPES | {
    "object_rel_distance", "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "obj_appearance_order", "route_planning"})
MRA_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]  # 0.50 .. 0.95
IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"


def build_vsi_question(item):
    n = len(item["frames"])
    q = f"These are {n} frames sampled in temporal order from a video scanning a scene.\n\n"
    for i in range(n):
        q += f"View {i + 1}: <image>\n"
    q += "\n" + item["question"]
    if item["options"]:
        for opt in item["options"]:
            q += "\n" + opt
    if item["question_type"] in NA_TYPES:
        q += "\nOutput only the final answer (number only) in <answer> </answer> tags. Do not include any reasoning."
    else:
        q += "\nOutput only the final answer (the letter of the choice) in <answer> </answer> tags. Do not include any reasoning."
    return q


def build_prompt(model, tokenizer, question, num_patches_list, num_image_token):
    """Replicate InternVLChatModel.chat() prompt construction with a controllable
    per-tile token count (so the pruned path can use a reduced count)."""
    get_conv_template = sys.modules[type(model).__module__].get_conv_template
    if "<image>" not in question:
        question = "<image>\n" + question
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for i, num_patches in enumerate(num_patches_list):
        # num_image_token: int (uniform per tile) OR per-<image> list of counts (SCMPruner)
        nt = num_image_token[i] if isinstance(num_image_token, (list, tuple)) else num_image_token * num_patches
        image_tokens = IMG_START + IMG_CTX * nt + IMG_END
        query = query.replace("<image>", image_tokens, 1)
    return tokenizer(query, return_tensors="pt"), eos_token_id, template


@torch.no_grad()
def scmpruner_features(model, pixel_values, capture, keep_ratio):
    """Cross-view SCMPruner selection for InternVL (VSI: 1 tile/frame -> n_views=n_frames,
    n_tok=256). extract_feature populates capture.cls_attn; we aggregate it to the LLM-token
    grid (as compute_visual_features does), run the shared selector, and return the kept
    features (concat per view, sorted) plus the per-view kept counts for build_prompt."""
    vit = model.extract_feature(pixel_values)               # (n_views, n_tok, C)
    n_views, n_tok, C = vit.shape
    cls_attn = capture.cls_attn                             # (n_views, P)
    hw = int(cls_attn.shape[1] ** 0.5)
    imp = cls_attn.reshape(n_views, hw, hw, 1).to(vit.dtype)
    imp = model.pixel_shuffle(imp, scale_factor=model.downsample_ratio)
    imp = imp.mean(dim=-1).reshape(n_views, -1).reshape(-1).float()   # (M,)
    keep = scm.scmpruner_keep_indices(vit.reshape(-1, C), imp, n_views, n_tok, keep_ratio)
    keep_t = torch.tensor(sorted(keep), device=vit.device)
    counts, feats = [], []
    for v in range(n_views):
        local = (keep_t[keep_t // n_tok == v] % n_tok).sort().values
        counts.append(int(local.numel()))
        feats.append(vit[v][local])
    return torch.cat(feats, dim=0), counts


def mra(pred, gt):
    if pred is None or gt == 0:
        return 0.0
    rel = abs(float(pred) - gt) / abs(gt)
    return sum(1.0 for th in MRA_THRESHOLDS if rel < 1 - th) / len(MRA_THRESHOLDS)


def mc_answer_letter(pred):
    """Robustly pull the A-D choice letter from a model answer. No-think outputs often
    drop the <answer> tags or add punctuation (e.g. 'B.'), which an exact-match scorer
    would miss; \\b[a-d]\\b picks the standalone letter and ignores letters inside words
    ('the answer is c.' -> 'c', not the 'a' in 'answer'). Falls back to the raw tag-
    extracted string. Backward-compatible with a clean '<answer>c</answer>' -> 'c'."""
    s = extract_number_from_answer_tag_mult_choice(pred)   # tag-aware + lowercased
    m = re.search(r"\b([a-d])\b", s)
    return m.group(1) if m else s


def compute_metric(save_path, question_type):
    rows = [json.loads(l) for l in open(save_path)]
    if not rows:
        return ("MRA" if question_type in NA_TYPES else "ACC"), None, 0
    if question_type in NA_TYPES:
        vals = []
        for r in rows:
            pred = extract_number_from_answer_tag_exact_num(r["Processed_Pred"])
            try:
                gt = float(r["GT"])
            except ValueError:
                continue
            vals.append(mra(pred, gt))
        return "MRA", (sum(vals) / len(vals) if vals else None), len(rows)
    correct = sum(1 for r in rows if mc_answer_letter(r["Processed_Pred"]) == r["GT"].lower())
    return "ACC", correct / len(rows), len(rows)


def run_category(model, tokenizer, items, category, args, compressor, capture, tag, out_dir, llm_cfg, fastv=None):
    save_path = f"{out_dir}/{category}.jsonl"
    processed = sum(1 for _ in open(save_path)) if os.path.exists(save_path) else 0
    cat_items = [it for it in items if it["question_type"] == category]
    if args.limit is not None:
        cat_items = cat_items[:args.limit]
    if processed:
        print(f"  resuming {category}: {processed} done")

    eff_records = []
    for seen, item in enumerate(tqdm(cat_items, desc=category), start=1):
        if seen <= processed:
            continue
        frames = [os.path.join(args.data_root, f) for f in item["frames"]]
        pixel_values, num_patches_list = prepare_images_internvl(frames)
        n_frames = sum(num_patches_list)
        question = build_vsi_question(item)

        with efficiency.GpuProfile() as enc:
            if args.compress_method == "scmpruner":
                visual_features, img_tokens = scmpruner_features(
                    model, pixel_values, capture, args.keep_ratio)   # img_tokens = per-view counts
            else:
                visual_features, img_tokens = compute_visual_features(
                    model, pixel_values, compressor, args.keep_ratio, capture,
                    base_seed=args.seed, sample_id=seen)             # img_tokens = uniform int
        model_inputs, eos_token_id, template = build_prompt(
            model, tokenizer, question, num_patches_list, img_tokens)
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        seq_len = input_ids.shape[1]
        n_visual = sum(img_tokens) if isinstance(img_tokens, list) else img_tokens * n_frames

        if fastv is not None:      # FastV: prune in-LLM at layer K by last-token attention
            fastv.configure(input_ids, model.img_context_token_id, args.keep_ratio, n_visual)
        with efficiency.GpuProfile() as prof:
            generated = model.generate(
                pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, visual_features=visual_features,
                max_new_tokens=16, do_sample=False, eos_token_id=eos_token_id)
        if fastv is not None:
            fastv.off()
        response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        response_processed = response.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()

        with open(save_path, "a") as f:
            f.write(json.dumps({
                "Question": question, "Question_Type": item["question_type"],
                "Pred": response, "Processed_Pred": response_processed,
                "GT": item["ground_truth"]}) + "\n")
        eff_records.append({
            "visual_tokens": n_visual, "seq_len": seq_len,
            "flops": efficiency.prefill_flops(seq_len, llm_cfg),
            "kv_bytes": efficiency.kv_cache_bytes(seq_len, llm_cfg),
            "cuda_time_ms": prof.cuda_time_ms, "peak_mem_bytes": prof.peak_mem_bytes,
            "encode_ms": enc.cuda_time_ms})

    metric_name, value, n_eval = compute_metric(save_path, category)
    result = {"benchmark": "vsibench", "category": category, "method": tag,
              "keep_ratio": args.keep_ratio if compressor is not None else 1.0,
              "seed": args.seed, "n_frames": args.frames,
              "metric": metric_name, "value": value, "n_eval": n_eval}
    if eff_records:
        result["efficiency"] = efficiency.summarize(eff_records)
        result["efficiency"]["encode_ms"] = sum(r["encode_ms"] for r in eff_records) / len(eff_records)
    json.dump(result, open(f"{out_dir}/{category}.result.json", "w"), indent=2)
    v = "—" if value is None else f"{value:.4f}"
    print(f"[result:{tag}:vsibench:{category}] {metric_name}={v} n={n_eval}"
          + (f" tokens={result['efficiency']['visual_tokens']:.0f} TFLOPs={result['efficiency']['tflops']:.2f}"
             if eff_records else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--model_name", default="InternVL3-8B")
    ap.add_argument("--data_root", default="data/vsibench")
    ap.add_argument("--items", default="data/vsibench/vsibench_items.json")
    ap.add_argument("--category", default="all", help="'all', one type, or comma-separated types")
    ap.add_argument("--compress_method", default="none", help="none | vispruner | random")
    ap.add_argument("--keep_ratio", type=float, default=0.25)
    ap.add_argument("--important_ratio", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=6, help="frames/video (must match prep)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fastv_k", type=int, default=2, help="FastV: prune layer K (default 2)")
    ap.add_argument("--limit", type=int, default=None, help="first N items per category (smoke)")
    args = ap.parse_args()

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path)).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)
    llm_cfg = model.language_model.config

    method = args.compress_method
    compressor = (build_compressor(method, important_ratio=args.important_ratio)
                  if method not in ({"none"} | SPECIAL_METHODS) else None)
    need_capture = (compressor is not None and getattr(compressor, "needs_importance", True)) \
        or method == "scmpruner"                             # scmpruner needs ViT saliency; fastv doesn't
    capture = AttentionCapture(model) if need_capture else None
    tag = "baseline" if method == "none" else f"{method}-keep{int(round(args.keep_ratio*100))}"

    fastv = None
    if method == "fastv":
        from compressors.fastv import FastVInternVL
        fastv = FastVInternVL(model.language_model.model, K=args.fastv_k)

    items = json.load(open(args.items))
    if args.category == "all":
        categories = [t for t in ALL_TYPES if any(it["question_type"] == t for it in items)]
    else:
        categories = [c.strip() for c in args.category.split(",") if c.strip()]

    out_dir = f"logs/{args.model_name}-{tag}-vsibench"
    os.makedirs(out_dir, exist_ok=True)
    print(f">>> VSI-Bench [{tag}] over {len(categories)} categories: {categories}")
    for category in categories:
        run_category(model, tokenizer, items, category, args, compressor, capture, tag, out_dir, llm_cfg, fastv)


if __name__ == "__main__":
    main()
