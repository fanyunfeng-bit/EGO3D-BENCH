"""Qwen2.5-VL-7B + visual-token compression on VSI-Bench.

Mirrors models/internvl3_vsibench.py but for Qwen2.5-VL, which needs bespoke
plumbing (see compressors/qwen_adapter.py): no CLS / windowed ViT attention and
M-RoPE positions. We run the pipeline manually so pruning is exact:

  visual -> image_embeds (+ captured importance for vispruner)
  -> pick keep-mask per frame (compressor)
  -> scatter into inputs_embeds, get full M-RoPE position_ids
  -> drop pruned visual positions (kept tokens retain their original 2D position)
  -> manual greedy decode over the reduced prefill (same path for all methods,
     so CUDA-time is comparable).

Frames are forced to 448x448 -> 256 tokens/frame (min_pixels=max_pixels) so the
budget matches the InternVL3 VSI run. Metrics: Accuracy (MC) / MRA (numeric).
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import (extract_number_from_answer_tag_mult_choice,
                        extract_number_from_answer_tag_exact_num)
from utils.common import convert_to_qwen_input
from utils import efficiency
from compressors import build_compressor
from compressors.qwen_adapter import QwenAttentionCapture, merged_importance, qwen_select_per_image

torch.manual_seed(42)

NA_TYPES = {"object_size_estimation", "object_abs_distance", "object_counting", "room_size_estimation"}
ALL_TYPES = sorted(NA_TYPES | {"object_rel_distance", "object_rel_direction_easy",
                               "object_rel_direction_medium", "object_rel_direction_hard",
                               "obj_appearance_order", "route_planning"})
MRA_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]
PIX = 448 * 448  # force ~256 merged tokens/frame


def build_messages(item, frame_paths):
    """Build Qwen chat messages: View N: <image> ... + question + R1 CoT.
    Each frame forced to 448x448 (resized_height/width) -> 32x32 patches ->
    256 merged tokens/frame (square, matching the InternVL3 VSI setup)."""
    n = len(frame_paths)
    content = [{"type": "text",
                "text": f"These are {n} frames sampled in temporal order from a video scanning a scene.\n\n"}]
    for i, fp in enumerate(frame_paths):
        content.append({"type": "text", "text": f"View {i + 1}: "})
        content.append({"type": "image", "image": fp, "resized_height": 448, "resized_width": 448})
        content.append({"type": "text", "text": "\n"})
    q = "\n" + item["question"]
    if item["options"]:
        for opt in item["options"]:
            q += "\n" + opt
    if item["question_type"] in NA_TYPES:
        q += "\nOutput the thinking process in <think> </think> and final answer (number only) in <answer> </answer> tags."
    else:
        q += "\nOutput the thinking process in <think> </think> and final answer (only the letter of the choice) in <answer> </answer> tags."
    content.append({"type": "text", "text": q})
    return [{"role": "user", "content": content}]


def mra(pred, gt):
    if pred is None or gt == 0:
        return 0.0
    rel = abs(float(pred) - gt) / abs(gt)
    return sum(1.0 for th in MRA_THRESHOLDS if rel < 1 - th) / len(MRA_THRESHOLDS)


def compute_metric(save_path, qtype):
    rows = [json.loads(l) for l in open(save_path)]
    if not rows:
        return ("MRA" if qtype in NA_TYPES else "ACC"), None, 0
    if qtype in NA_TYPES:
        vals = []
        for r in rows:
            try:
                vals.append(mra(extract_number_from_answer_tag_exact_num(r["Processed_Pred"]), float(r["GT"])))
            except ValueError:
                pass
        return "MRA", (sum(vals) / len(vals) if vals else None), len(rows)
    correct = sum(1 for r in rows
                  if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower())
    return "ACC", correct / len(rows), len(rows)


@torch.no_grad()
def greedy_decode(model, inputs_embeds, position_ids, eos_ids, max_new_tokens=1024):
    """Manual greedy decode over a (possibly pruned) prefill with M-RoPE positions."""
    L = inputs_embeds.shape[1]
    out = model.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                      use_cache=True, cache_position=torch.arange(L, device=inputs_embeds.device))
    past = out.past_key_values
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    nxt = int(logits.argmax(-1))
    base = int(position_ids.max()) + 1  # next M-RoPE position (text region: all 3 dims equal)
    gen = [nxt]
    for i in range(max_new_tokens - 1):
        if nxt in eos_ids:
            break
        emb = model.model.embed_tokens(torch.tensor([[nxt]], device=inputs_embeds.device))
        pos = torch.full((3, 1, 1), base + i, device=inputs_embeds.device, dtype=torch.long)
        out = model.model(inputs_embeds=emb, position_ids=pos, past_key_values=past, use_cache=True,
                          cache_position=torch.tensor([L + i], device=inputs_embeds.device))
        past = out.past_key_values
        nxt = int(model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(-1))
        gen.append(nxt)
    return gen


@torch.no_grad()
def run_category(model, processor, items, category, args, compressor, capture, tag, out_dir, llm_cfg, eos_ids):
    save_path = f"{out_dir}/{category}.jsonl"
    processed = sum(1 for _ in open(save_path)) if os.path.exists(save_path) else 0
    cat_items = [it for it in items if it["question_type"] == category]
    if args.limit is not None:
        cat_items = cat_items[:args.limit]
    if processed:
        print(f"  resuming {category}: {processed} done")
    image_token_id = model.config.image_token_id
    eff_records = []

    for seen, item in enumerate(tqdm(cat_items, desc=category), start=1):
        if seen <= processed:
            continue
        frames = [os.path.join(args.data_root, f) for f in item["frames"]]
        messages = build_messages(item, frames)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with efficiency.GpuProfile() as enc:
            image_embeds = model.visual(inputs["pixel_values"].type(model.visual.dtype),
                                        grid_thw=inputs["image_grid_thw"])
            if compressor is None:
                keep_mask_vis = torch.ones(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
            else:
                importance = (merged_importance(model, inputs["image_grid_thw"], capture.received)
                              if capture is not None else None)
                keep_mask_vis = qwen_select_per_image(model, image_embeds, importance,
                                                      inputs["image_grid_thw"], compressor,
                                                      args.keep_ratio, args.seed, seen)
            inputs_embeds = model.model.embed_tokens(inputs["input_ids"])
            img_pos = inputs["input_ids"][0] == image_token_id
            inputs_embeds[0, img_pos] = image_embeds.to(inputs_embeds.dtype)
            position_ids, _ = model.get_rope_index(inputs["input_ids"], inputs["image_grid_thw"],
                                                   attention_mask=inputs["attention_mask"])
            keep_full = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
            keep_full[img_pos] = keep_mask_vis
            red_embeds = inputs_embeds[:, keep_full, :]
            red_pos = position_ids[:, :, keep_full]
        kept_visual = int(keep_mask_vis.sum())
        seq_len = int(keep_full.sum())

        with efficiency.GpuProfile() as prof:
            gen_ids = greedy_decode(model, red_embeds, red_pos, eos_ids)
        response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        response_processed = response.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()

        with open(save_path, "a") as f:
            f.write(json.dumps({"Question_Type": item["question_type"], "Pred": response,
                                "Processed_Pred": response_processed, "GT": item["ground_truth"]}) + "\n")
        eff_records.append({"visual_tokens": kept_visual, "seq_len": seq_len,
                            "flops": efficiency.prefill_flops(seq_len, llm_cfg),
                            "kv_bytes": efficiency.kv_cache_bytes(seq_len, llm_cfg),
                            "cuda_time_ms": prof.cuda_time_ms, "peak_mem_bytes": prof.peak_mem_bytes,
                            "encode_ms": enc.cuda_time_ms})

    metric_name, value, n_eval = compute_metric(save_path, category)
    result = {"benchmark": "vsibench", "category": category, "method": tag,
              "keep_ratio": args.keep_ratio if compressor is not None else 1.0,
              "seed": args.seed, "n_frames": args.frames, "metric": metric_name, "value": value, "n_eval": n_eval}
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
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model_name", default="Qwen2.5-VL-7B")
    ap.add_argument("--data_root", default="data/vsibench")
    ap.add_argument("--items", default="data/vsibench/vsibench_items.json")
    ap.add_argument("--category", default="route_planning")
    ap.add_argument("--compress_method", default="none")
    ap.add_argument("--keep_ratio", type=float, default=0.25)
    ap.add_argument("--important_ratio", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=args.attn).eval()

    # last-token lm_head (memory; harmless here) — slices only when >1 position
    class _LastTokenHead(nn.Module):
        def __init__(self, head):
            super().__init__(); self.head = head
        def forward(self, h):
            if h.dim() == 3 and h.size(1) > 1:
                h = h[:, -1:, :]
            return self.head(h)
    model.lm_head = _LastTokenHead(model.lm_head)

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True, min_pixels=PIX, max_pixels=PIX)
    llm_cfg = model.model.config
    eos_ids = {processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}

    compressor = build_compressor(args.compress_method, important_ratio=args.important_ratio) \
        if args.compress_method != "none" else None
    capture = QwenAttentionCapture(model) if (compressor is not None and
                                              getattr(compressor, "needs_importance", True)) else None
    tag = "baseline" if compressor is None else f"{args.compress_method}-keep{int(round(args.keep_ratio*100))}"

    items = json.load(open(args.items))
    categories = ([t for t in ALL_TYPES if any(it["question_type"] == t for it in items)]
                  if args.category == "all" else [c.strip() for c in args.category.split(",") if c.strip()])
    out_dir = f"logs/{args.model_name}-{tag}-vsibench"
    os.makedirs(out_dir, exist_ok=True)
    print(f">>> Qwen2.5-VL VSI-Bench [{tag}] over {categories}")
    for category in categories:
        run_category(model, processor, items, category, args, compressor, capture, tag, out_dir, llm_cfg, eos_ids)


if __name__ == "__main__":
    main()
