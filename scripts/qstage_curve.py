"""Two-stage Query-Aware Block-CVSP (Notes/CVSP-Method.md §13).

Stage 1 (pre-LLM, block-cvsp): over-select N1 = r/(r+ (L-K)/K) ... tokens (1.75*T at r=7,K=L/2).
Stage 2 (in-LLM @ layer K): prune to N2 by query signal (cosine|attn), cross-view global, PESP.
Reported budget = layer-average (N1*K + N2*(L-K))/L = T. Custom greedy decode, use_cache=False.

Resumable per (ds, task, ratio, signal, r). Logs -> logs/cvsp/<ds>.<task>.keep<pct>.qstage-<sig>-r<r>.jsonl
"""
import argparse, json, math, os, sys

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, SCRIPTS_DIR)

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from utils.eval import extract_number_from_answer_tag_mult_choice, extract_number_from_answer_tag_exact_num
from utils.internvl3_utils import prepare_images_internvl, split_model
import models.internvl3_compress as RUN
from compressors.internvl_adapter import AttentionCapture
import four_way_extreme as FW
import cvsp_curve as CV
from compressors.qstage_llm import QStage, make_qstage_forward

CONFIGS = [("vsi", "object_rel_direction_hard"), ("vsi", "object_rel_direction_medium"),
           ("vsi", "object_rel_distance"), ("ego3d", "Localization"),
           ("ego3d", "Ego_Centric_Absolute_Distance_MultiChoice")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--ratios", default="0.1,0.05")        # reported layer-average T/M
    ap.add_argument("--configs", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    ap.add_argument("--vsi_items", default="data/vsibench/vsibench_items.json")
    ap.add_argument("--rho_a", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--brows", type=int, default=4)
    ap.add_argument("--bcols", type=int, default=4)
    ap.add_argument("--signal", default="cosine", choices=["cosine", "attn", "input_cos"])
    ap.add_argument("--r", type=float, default=7.0)         # N1/N2 over-select ratio
    ap.add_argument("--K", type=int, default=14)            # prune layer (L/2 for Qwen2.5-7B L=28)
    ap.add_argument("--max_new", type=int, default=48)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    configs = [tuple(c.split(":")) for c in args.configs.split(",")] if args.configs else CONFIGS

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(RUN.IMG_CTX)
    capture = AttentionCapture(model)
    ego_ds = load_dataset("vbdai/Ego3D-Bench")["test"]
    os.makedirs("logs/cvsp", exist_ok=True)

    L = model.language_model.config.num_hidden_layers       # 28
    qwen = model.language_model.model
    lm_head = model.language_model.lm_head
    embed = model.language_model.get_input_embeddings()
    qs = QStage(K=args.K, signal=args.signal)
    qwen._qs = qs
    make_qstage_forward(qwen)

    def budgets(ratio, M):
        T = ratio * M
        r, K = args.r, args.K
        N2 = max(1, round(T * L / (r * K + L - K)))
        N1 = min(M, max(N2, round(r * N2)))
        avg = (N1 * K + N2 * (L - K)) / L
        return int(N1), int(N2), avg

    def run(ratio, save, samples):
        done = sum(1 for _ in open(save)) if os.path.exists(save) else 0
        for seen, (q, paths, gt) in enumerate(samples, 1):
            if seen <= done:
                continue
            pv, _ = prepare_images_internvl(paths)
            vit = model.extract_feature(pv)
            n_views, n_tok, C = vit.shape
            M = n_views * n_tok
            N1, N2, avg = budgets(ratio, M)
            imp = CV.compute_imp(model, capture, vit)
            Fn = torch.nn.functional.normalize(vit.reshape(-1, C).float(), dim=-1)
            G = Fn @ Fn.t()
            view_id = torch.arange(M, device=Fn.device) // n_tok
            a = FW.cornerness(G, view_id, n_views) * FW.lowe_max(G, view_id, n_views)
            gi, _ = CV.sel_block_cvsp(G, a, imp, view_id, n_views, n_tok, N1,
                                      args.rho_a, args.tau, args.brows, args.bcols, return_anchor=True)
            per_view = FW.to_perview(gi, view_id, n_views, n_tok)
            feats = torch.cat([vit[t][per_view[t]] for t in range(n_views)], dim=0)
            counts = [int(len(per_view[t])) for t in range(n_views)]
            mi, eos, tmpl = FW.build_prompt_var(model, tokenizer, q, counts)
            ids = mi["input_ids"].cuda()
            assert int((ids == model.img_context_token_id).sum()) == feats.shape[0]

            sel = (ids.reshape(-1) == model.img_context_token_id)
            vis_pos = torch.where(sel)[0]
            last_vis = int(vis_pos[-1].item())
            qpos = torch.arange(last_vis + 1, ids.shape[1], device=ids.device)   # tokens after image
            qs.active = True; qs.N2 = N2; qs.vis_pos = vis_pos
            qs.query_pos = qpos; qs.kept_vis = None
            if args.signal == "input_cos":   # paper-exact: cos(proj(v_i), q_bar) in INPUT-embed space, pre-LLM
                qemb = embed(ids.reshape(-1)[qpos]).float()                      # (Q, C) question token embeds
                qbar = torch.nn.functional.normalize(qemb.mean(0, keepdim=True), dim=-1)
                fv = torch.nn.functional.normalize(feats.float(), dim=-1)        # (N1, C) projected vision
                r = (fv @ qbar.t()).squeeze(1)                                   # (N1,)
                top = torch.topk(r, min(N2, feats.shape[0])).indices
                qs.kept_vis = vis_pos[top].sort().values                         # pre-decided keep set
            gen = model.generate(pixel_values=pv, input_ids=ids,
                                 attention_mask=mi["attention_mask"].cuda(), visual_features=feats,
                                 max_new_tokens=1024, do_sample=False, eos_token_id=eos)
            qs.active = False

            resp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].split(tmpl.sep.strip())[0].strip()
            proc = resp.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()
            with open(save, "a") as f:
                f.write(json.dumps({"Processed_Pred": proc, "GT": gt, "n_kept": round(avg, 1),
                                    "N1": N1, "N2": N2}) + "\n")

    def summarize(dsname, task):
        is_num = task in FW.NUM_CATS
        print(f"\n=== {dsname}/{task} ({'RMSE' if is_num else 'ACC'}, qstage {args.signal} r={args.r}) ===", flush=True)
        for ratio in ratios:
            pct = int(round(ratio * 100))
            p = f"logs/cvsp/{dsname}.{task}.keep{pct}.qstage-{args.signal}-r{int(args.r)}.jsonl"
            if not os.path.exists(p):
                continue
            rs = [json.loads(l) for l in open(p)]
            tok = sum(r["n_kept"] for r in rs) / len(rs)
            if is_num:
                yt, yp = [], []
                for r in rs:
                    pr = extract_number_from_answer_tag_exact_num(r["Processed_Pred"])
                    if pr is not None:
                        yt.append(float(r["GT"])); yp.append(min(float(pr), 100.0))
                rmse = (sum((a - b) ** 2 for a, b in zip(yt, yp)) / len(yt)) ** 0.5 if yt else float("nan")
                print(f"  keep{pct:<3} RMSE={rmse:.3f}  avg_tok={tok:.0f}  n={len(rs)}(valid={len(yt)})", flush=True)
            else:
                acc = sum(extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower()
                          for r in rs) / len(rs)
                print(f"  keep{pct:<3} ACC={acc:.4f}  avg_tok={tok:.0f}  n={len(rs)}", flush=True)

    for dsname, task in configs:
        samples = FW.collect(dsname, task, args.n, args.image_root, args.vsi_items, ego_ds)
        for ratio in ratios:
            pct = int(round(ratio * 100))
            save = f"logs/cvsp/{dsname}.{task}.keep{pct}.qstage-{args.signal}-r{int(args.r)}.jsonl"
            run(ratio, save, samples)
            print(f"[{dsname}/{task} keep{pct} qstage-{args.signal}-r{int(args.r)}] done", flush=True)
        summarize(dsname, task)


if __name__ == "__main__":
    main()
