"""InternVL3 + pluggable visual-token compression on Ego3D-Bench.

Extensible comparison harness (does NOT modify the baseline `internvl3.py`):
  --compress_method none       -> baseline (all tokens per view)
  --compress_method vispruner  -> VisPruner (keep `--keep_ratio` per view)

Both methods run through the SAME instrumented generate path so the efficiency
numbers are comparable. For a fair, LLM-focused measurement the visual features
(full for baseline, pruned for VisPruner) are computed OUTSIDE the timed region;
the timed region is the LLM generate over the (reduced) token sequence. The
vision+prune cost is logged separately as `encode_ms`.

Efficiency metrics (VisPruner / FastV convention, see utils/efficiency.py):
theoretical prefill FLOPs, KV-cache bytes, measured peak GPU memory, CUDA time.
Accuracy is scored by ego3d's eval (ACC for multiple-choice categories).
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import *  # noqa: F401,F403  (eval_logs)
from utils.internvl3_utils import prepare_images_internvl, split_model
from utils import efficiency
from compressors import build_compressor
from compressors.internvl_adapter import AttentionCapture, compute_visual_features

torch.manual_seed(42)

NUMBER_CATEGORIES = ["Ego_Centric_Absolute_Distance", "Object_Centric_Absolute_Distance"]
IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"


def compute_metric(save_path, multi_choice):
    """Recompute the ego3d metric value from a saved jsonl (so it can be
    persisted to json), mirroring utils/eval.py exactly.
    Returns (metric_name, value, n_rows)."""
    import numpy as np
    from sklearn.metrics import mean_squared_error
    rows = [json.loads(l) for l in open(save_path)]
    if not rows:
        return ("ACC" if multi_choice else "RMSE"), None, 0
    if multi_choice:
        correct = sum(
            1 for r in rows
            if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower()
        )
        return "ACC", correct / len(rows), len(rows)
    y_true, y_pred = [], []
    for r in rows:
        pred = extract_number_from_answer_tag_exact_num(r["Processed_Pred"])
        if pred:  # mirrors eval_exact_num (skips None and 0)
            y_pred.append(min(float(pred), 100))
            y_true.append(float(r["GT"]))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if y_true else None
    return "RMSE", rmse, len(rows)

IMAGE_ORDER = {
    "nuscenes": ["Front_Left", "Front", "Front_Right", "Back_Right", "Back", "Back_Left"],
    "waymo": ["Front", "Front_Left", "Side_Left", "Front_Right", "Side_Right"],
    "argoverse": ["Front_Left", "Front", "Front_Right", "Side_Right", "Back_Right", "Back_Left", "Side_Left"],
}


def build_prompt(model, tokenizer, question, num_patches_list, num_image_token):
    """Replicate InternVLChatModel.chat() prompt construction, but with a
    controllable per-tile token count (so we can use a reduced count for the
    pruned path) — returns tokenized inputs + the template for decoding."""
    get_conv_template = sys.modules[type(model).__module__].get_conv_template
    if "<image>" not in question:
        question = "<image>\n" + question
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for num_patches in num_patches_list:
        image_tokens = IMG_START + IMG_CTX * num_image_token * num_patches + IMG_END
        query = query.replace("<image>", image_tokens, 1)
    model_inputs = tokenizer(query, return_tensors="pt")
    return model_inputs, eos_token_id, template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--model_name", type=str, default="InternVL3-8B")
    ap.add_argument("--category", type=str, default="Object_Centric_Absolute_Distance_MultiChoice")
    ap.add_argument("--compress_method", type=str, default="none", help="none | vispruner")
    ap.add_argument("--keep_ratio", type=float, default=0.5, help="fraction of visual tokens to keep per view")
    ap.add_argument("--important_ratio", type=float, default=0.5, help="VisPruner important:diverse split")
    ap.add_argument("--seed", type=int, default=42,
                    help="base seed for stochastic methods (e.g. random); per-(sample,view) "
                         "seeds are derived from it so the same experiment is reproducible")
    ap.add_argument("--image_root", type=str, default="Ego3D-Bench/images")
    ap.add_argument("--limit", type=int, default=None, help="only first N samples in the category (smoke test)")
    args = ap.parse_args()

    device = "cuda"
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, load_in_8bit=False,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True,
        device_map=split_model(args.model_path),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)
    llm_cfg = model.language_model.config

    compressor = build_compressor(
        args.compress_method, important_ratio=args.important_ratio
    ) if args.compress_method != "none" else None
    # only attach the (non-flash) ViT attention capture for methods that use it
    need_capture = compressor is not None and getattr(compressor, "needs_importance", True)
    capture = AttentionCapture(model) if need_capture else None
    # baseline is keep-ratio independent (shared dir); compressed dirs carry the
    # keep-ratio so different rates (e.g. keep50 vs keep25) don't clobber each other.
    tag = "baseline" if compressor is None else f"{args.compress_method}-keep{int(round(args.keep_ratio*100))}"

    dataset = load_dataset("vbdai/Ego3D-Bench")["test"]
    out_dir = f"logs/{args.model_name}-{tag}"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/{args.category}.jsonl"
    processed = sum(1 for _ in open(save_path)) if os.path.exists(save_path) else 0
    if processed:
        print(f"Resuming: {processed} samples already in {save_path}")

    eff_records = []
    seen = 0
    for sample in tqdm(dataset):
        if sample["category"] != args.category:
            continue
        seen += 1
        if seen <= processed:
            continue
        if args.limit is not None and seen > args.limit:
            break

        question = sample["question"]
        if sample["options"]:
            for opt in sample["options"]:
                question += "\n" + opt
        if args.category in NUMBER_CATEGORIES:
            question += "\nOutput the thinking process in <think> </think> and final answer (number only) in <answer> </answer> tags."
        else:
            question += "\nOutput the thinking process in <think> </think> and final answer (only the letter of the choice) in <answer> </answer> tags."

        order = IMAGE_ORDER.get(sample["source"], list(sample["images"].keys()))
        image_paths = [os.path.join(args.image_root, sample["images"][v]) for v in order]
        pixel_values, num_patches_list = prepare_images_internvl(image_paths)
        n_views = sum(num_patches_list)

        # --- visual features (outside the timed region) ---
        with efficiency.GpuProfile() as enc:
            visual_features, tok_per_tile = compute_visual_features(
                model, pixel_values, compressor, args.keep_ratio, capture,
                base_seed=args.seed, sample_id=seen,
            )
        visual_tokens = tok_per_tile * n_views

        model_inputs, eos_token_id, template = build_prompt(
            model, tokenizer, question, num_patches_list, tok_per_tile
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]

        # --- timed LLM generation ---
        with efficiency.GpuProfile() as prof:
            generated = model.generate(
                pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, visual_features=visual_features,
                max_new_tokens=1024, do_sample=False, eos_token_id=eos_token_id,
            )
        response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()

        response_processed = response.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()
        with open(save_path, "a") as f:
            f.write(json.dumps({
                "Question": question,
                "Question_Type": "multi_choice" if sample["options"] else "exact_number",
                "Pred": response,
                "Processed_Pred": response_processed,
                "GT": sample["answer"],
            }) + "\n")

        eff_records.append({
            "visual_tokens": visual_tokens,
            "seq_len": seq_len,
            "flops": efficiency.prefill_flops(seq_len, llm_cfg),
            "kv_bytes": efficiency.kv_cache_bytes(seq_len, llm_cfg),
            "cuda_time_ms": prof.cuda_time_ms,
            "peak_mem_bytes": prof.peak_mem_bytes,
            "encode_ms": enc.cuda_time_ms,
        })

    # --- performance (ego3d eval) ---
    print(f"\n##### {args.category} [{tag}]")
    multi_choice = args.category not in NUMBER_CATEGORIES
    eval_logs(save_path, multi_choice=multi_choice)
    metric_name, metric_value, n_eval = compute_metric(save_path, multi_choice)

    # --- per-category result json: performance + efficiency, one file PER TASK ---
    result = {
        "category": args.category,
        "method": tag,
        "keep_ratio": args.keep_ratio if compressor is not None else 1.0,
        "seed": args.seed,
        "metric": metric_name,
        "value": metric_value,
        "n_eval": n_eval,
    }
    if eff_records:
        summary = efficiency.summarize(eff_records)
        summary["encode_ms"] = sum(r["encode_ms"] for r in eff_records) / len(eff_records)
        result["efficiency"] = summary
    with open(f"{out_dir}/{args.category}.result.json", "w") as f:
        json.dump(result, f, indent=2)

    msg = f"[result:{tag}:{args.category}] {metric_name}={metric_value}"
    if eff_records:
        e = result["efficiency"]
        msg += (f"  visual_tokens={e['visual_tokens']:.0f} TFLOPs={e['tflops']:.2f} "
                f"KV={e['kv_mb']:.1f}MB peakMem={e['peak_mem_mb']:.0f}MB CUDAtime={e['cuda_time_ms']:.1f}ms")
    else:
        msg += "  (no new samples this run; efficiency not re-measured)"
    print(msg)


if __name__ == "__main__":
    main()
