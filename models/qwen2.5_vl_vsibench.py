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
import re
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
from compressors import build_compressor, scm
from compressors.qwen_adapter import QwenAttentionCapture, merged_importance, qwen_select_per_image

# methods not in the per-image registry: cross-view SCMPruner (pre-LLM) + in-LLM FastV
SPECIAL_METHODS = {"scmpruner", "fastv", "scmpruner_qa", "mvpruner"}

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
        q += "\nOutput only the final answer (number only) in <answer> </answer> tags. Do not include any reasoning."
    else:
        q += "\nOutput only the final answer (the letter of the choice) in <answer> </answer> tags. Do not include any reasoning."
    content.append({"type": "text", "text": q})
    return [{"role": "user", "content": content}]


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
    correct = sum(1 for r in rows if mc_answer_letter(r["Processed_Pred"]) == r["GT"].lower())
    return "ACC", correct / len(rows), len(rows)


@torch.no_grad()
def greedy_decode(model, inputs_embeds, position_ids, eos_ids, max_new_tokens=16):
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
def run_category(model, processor, items, category, args, compressor, capture, tag, out_dir, llm_cfg, eos_ids,
                 fastv_ctrl=None, qa_ctrl=None, mv_ctrl=None):
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
            method = args.compress_method
            N2 = None
            if method == "scmpruner_qa":                      # stage-1: SCMPruner over-select N1
                importance = merged_importance(model, inputs["image_grid_thw"], capture.received)
                n_views = inputs["image_grid_thw"].shape[0]
                n_tok = image_embeds.shape[0] // n_views
                M = image_embeds.shape[0]
                L = model.model.config.num_hidden_layers
                N1, N2, _ = scm.scmpruner_qa_budgets(args.keep_ratio, M, args.scm_r, args.scm_K, L)
                if args.scm_softweight:
                    q_ids = processor.tokenizer(item["question"], return_tensors="pt").input_ids.to(image_embeds.device)
                    q_emb = model.model.embed_tokens(q_ids)[0]
                    importance = importance * scm.input_cos_relevance(image_embeds, q_emb)
                keep = scm.scmpruner_keep_indices(image_embeds, importance, n_views, n_tok, N1 / M,
                                                  rho_a=args.scm_rho_a, rho_s=args.scm_rho_s,
                                                  anc_tau=args.anc_tau, anc_m=args.anc_m,
                                                  xview=bool(args.scm_xview))
                keep_mask_vis = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
                keep_mask_vis[torch.tensor(keep, device=image_embeds.device)] = True
            elif method == "scmpruner":                        # cross-view pre-LLM selection
                importance = merged_importance(model, inputs["image_grid_thw"], capture.received)
                n_views = inputs["image_grid_thw"].shape[0]
                n_tok = image_embeds.shape[0] // n_views      # 256 (448x448 frames)
                keep = scm.scmpruner_keep_indices(image_embeds, importance, n_views, n_tok, args.keep_ratio,
                                                  rho_a=args.scm_rho_a, rho_s=args.scm_rho_s,
                                                  anc_tau=args.anc_tau, anc_m=args.anc_m,
                                                  xview=bool(args.scm_xview))
                keep_mask_vis = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
                keep_mask_vis[torch.tensor(keep, device=image_embeds.device)] = True
            elif compressor is None:                          # baseline / fastv: keep all pre-LLM
                keep_mask_vis = torch.ones(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
            else:                                             # random / vispruner (per-frame registry)
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

        if fastv_ctrl is not None:      # FastV: prune in-LLM at layer K by last-token attention (per-view)
            n_vis = int((inputs["input_ids"][0] == image_token_id).sum())
            fastv_ctrl.vis_pos = torch.where(keep_full)[0][
                (inputs["input_ids"][0][keep_full] == image_token_id)]   # visual positions in red seq
            n_views = inputs["image_grid_thw"].shape[0]
            fastv_ctrl.n_views = n_views
            fastv_ctrl.keep_pv = max(1, round(args.keep_ratio * (n_vis // n_views)))   # per-view budget
            fastv_ctrl.N2 = fastv_ctrl.keep_pv * n_views
            fastv_ctrl.last_q = red_embeds.shape[1] - 1
            fastv_ctrl.kept = None
            fastv_ctrl.active = True
        if qa_ctrl is not None:          # scmpruner_qa stage-2: in-LLM prune at layer K on the RED seq
            red_ids = inputs["input_ids"][0][keep_full]
            vis_pos_red = torch.where(red_ids == image_token_id)[0]
            last_vis = int(vis_pos_red[-1].item())
            qa_ctrl.vis_pos = vis_pos_red
            qa_ctrl.query_pos = torch.arange(last_vis + 1, red_ids.shape[0], device=red_ids.device)
            qa_ctrl.n_views = inputs["image_grid_thw"].shape[0]
            qa_ctrl.N2 = N2; qa_ctrl.kept = None; qa_ctrl.active = True
        if mv_ctrl is not None:          # MVPruner: two-stage in-LLM prune (keeps all pre-LLM)
            red_ids = inputs["input_ids"][0][keep_full]
            vis_pos_red = torch.where(red_ids == image_token_id)[0]
            last_vis = int(vis_pos_red[-1].item())
            grid = inputs["image_grid_thw"]
            merge = model.config.vision_config.spatial_merge_size
            view_lengths = [int(t * h * w) // (merge * merge) for t, h, w in grid.tolist()]
            assert sum(view_lengths) == vis_pos_red.numel(), (sum(view_lengths), vis_pos_red.numel())
            mv_ctrl.configure(vis_pos_red, view_lengths, last_vis + 1, red_ids.shape[0],
                              args.keep_ratio, model.model.config.num_hidden_layers)
        with efficiency.GpuProfile() as prof:
            gen_ids = greedy_decode(model, red_embeds, red_pos, eos_ids)
        if fastv_ctrl is not None:
            fastv_ctrl.active = False
        if qa_ctrl is not None:
            qa_ctrl.active = False
        if mv_ctrl is not None:
            mv_ctrl.off()
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
    ap.add_argument("--fastv_k", type=int, default=2, help="FastV: prune layer K (default 2)")
    ap.add_argument("--scm_rho_a", type=float, default=0.2, help="SCMPruner anchor budget frac (a20s40=0.2)")
    ap.add_argument("--scm_rho_s", type=float, default=0.4, help="SCMPruner saliency budget frac (a20s40=0.4)")
    ap.add_argument("--anc_m", type=float, default=0.12, help="SCMPruner Lowe-margin/sharpness gate (primary knob)")
    ap.add_argument("--anc_tau", type=float, default=0.6, help="SCMPruner cross-view cosine gate for a 'sharp' match")
    ap.add_argument("--scm_xview", type=int, default=1, help="SCMPruner: 1=xview coverage propagation on, 0=off")
    ap.add_argument("--scm_r", type=float, default=7.0, help="QA over-select ratio N1/N2")
    ap.add_argument("--scm_K", type=int, default=14, help="QA stage-2 prune layer (L/2=14)")
    ap.add_argument("--scm_sig", default="attn", choices=["attn", "cosine"], help="QA stage-2 signal")
    ap.add_argument("--scm_softweight", type=int, default=0, help="QA stage-1 saliency*relu(cos) soft-weight")
    ap.add_argument("--mv_l1", type=int, default=0, help="MVPruner stage-1 prune layer (paper 0)")
    ap.add_argument("--mv_l2", type=int, default=16, help="MVPruner stage-2 prune layer (paper 16)")
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

    method = args.compress_method
    compressor = (build_compressor(method, important_ratio=args.important_ratio)
                  if method not in ({"none"} | SPECIAL_METHODS) else None)
    need_capture = (compressor is not None and getattr(compressor, "needs_importance", True)) \
        or method in ("scmpruner", "scmpruner_qa")             # both need ViT saliency; fastv doesn't
    capture = QwenAttentionCapture(model) if need_capture else None
    tag = "baseline" if method == "none" else f"{method}-keep{int(round(args.keep_ratio*100))}"
    if method == "scmpruner":                                # auto-encode non-default knobs so a
        tag += scm.scmpruner_tag_suffix(args.scm_rho_a, args.scm_rho_s,   # sweep never collides / corrupts resume
                                        args.anc_tau, args.anc_m, bool(args.scm_xview))

    fastv_ctrl = None
    if method == "fastv":
        from compressors.fastv import QwenFastV, make_fastv_forward_qwen
        fastv_ctrl = QwenFastV(K=args.fastv_k)
        make_fastv_forward_qwen(model.model, fastv_ctrl)

    qa_ctrl = None
    if method == "scmpruner_qa":
        from compressors.fastv import QwenFastV, make_fastv_forward_qwen
        tag += scm.scmpruner_qa_tag_suffix(args.scm_r, args.scm_K, args.scm_sig, args.scm_softweight)
        tag += scm.scmpruner_tag_suffix(args.scm_rho_a, args.scm_rho_s,   # stage-1 knobs too, else a
                                        args.anc_tau, args.anc_m, bool(args.scm_xview))  # sweep corrupts resume
        qa_ctrl = QwenFastV(K=args.scm_K)
        qa_ctrl.signal = args.scm_sig; qa_ctrl.query_reduce = "mean"; qa_ctrl.per_view = False
        make_fastv_forward_qwen(model.model, qa_ctrl)

    mv_ctrl = None
    if method == "mvpruner":                                   # two-stage in-LLM (arXiv:2606.27660)
        from compressors.mvpruner import MVPruner, make_mvpruner_forward_qwen
        if (args.mv_l1, args.mv_l2) != (0, 16):
            tag += f"-l{args.mv_l1}_{args.mv_l2}"
        mv_ctrl = MVPruner(stage1_layer=args.mv_l1, stage2_layer=args.mv_l2)
        make_mvpruner_forward_qwen(model.model, mv_ctrl)

    items = json.load(open(args.items))
    categories = ([t for t in ALL_TYPES if any(it["question_type"] == t for it in items)]
                  if args.category == "all" else [c.strip() for c in args.category.split(",") if c.strip()])
    out_dir = f"logs/{args.model_name}-{tag}-vsibench"
    os.makedirs(out_dir, exist_ok=True)
    print(f">>> Qwen2.5-VL VSI-Bench [{tag}] over {categories}")
    for category in categories:
        run_category(model, processor, items, category, args, compressor, capture, tag, out_dir, llm_cfg, eos_ids,
                     fastv_ctrl, qa_ctrl, mv_ctrl)


if __name__ == "__main__":
    main()
