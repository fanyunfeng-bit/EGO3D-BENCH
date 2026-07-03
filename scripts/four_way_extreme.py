"""Four-way comparison at EXTREME compression (budget < effective rank ~84),
where the leverage diagnostic says selection quality can finally matter.
Tested on BOTH Ego3D and VSI.

Methods (all keep the SAME total budget, fed to the frozen MLLM, scored by ACC):
  strat_random : per-view uniform random over 2x2 spatial blocks (stratified).  <- strong baseline
  anchor       : top tokens by landmark score (cornerness x cross-view Lowe),
                 greedy + cross-view dedup + >=1/view floor.
  leverage     : top tokens by ridge-leverage (representational uniqueness).
  engine       : quality-weighted log-det greedy (q = anchor, sim = cosine).

Configs: Ego3D {Object_MC, Ego_MC} + VSI {rel_direction_easy, rel_distance};
ratios keep3 (~46 tok, decisive) + keep5 (~78); n=200; chance=0.25. Resumable.
"""
import argparse, json, os, sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from utils.eval import extract_number_from_answer_tag_mult_choice
from utils.internvl3_utils import prepare_images_internvl, split_model
import models.internvl3_compress as RUN

METHODS = ["strat_random", "anchor", "leverage", "engine"]
MC_INSTR = "\nOutput the thinking process in <think> </think> and final answer (only the letter of the choice) in <answer> </answer> tags."
NUM_INSTR = "\nOutput the thinking process in <think> </think> and final answer (number only) in <answer> </answer> tags."
NUM_CATS = {"Ego_Centric_Absolute_Distance", "Object_Centric_Absolute_Distance"}  # RMSE-scored
CONFIGS = [
    ("ego3d", "Object_Centric_Absolute_Distance_MultiChoice"),
    ("ego3d", "Ego_Centric_Absolute_Distance_MultiChoice"),
    ("vsi", "object_rel_direction_easy"),
    ("vsi", "object_rel_distance"),
]


# ---------- per-token scores ----------
def cornerness(G, view_id, n_views):
    M = G.shape[0]; corn = torch.zeros(M, device=G.device)
    for v in range(n_views):
        idx = torch.where(view_id == v)[0]
        sub = G[idx][:, idx]
        m = (sub.sum(1) - 1.0) / max(1, idx.numel() - 1)
        corn[idx] = (1 - m).clamp(min=0)
    return corn


def lowe_max(G, view_id, n_views):
    M = G.shape[0]; lowe = torch.zeros(M, device=G.device)
    for v in range(n_views):
        rows = torch.where(view_id == v)[0]
        best = torch.zeros(rows.numel(), device=G.device)
        for u in range(n_views):
            if u == v:
                continue
            cols = torch.where(view_id == u)[0]
            if cols.numel() < 2:
                continue
            top2 = G[rows][:, cols].topk(2, dim=1).values
            best = torch.maximum(best, (top2[:, 0] - top2[:, 1]).clamp(min=0) * top2[:, 0])
        lowe[rows] = best
    return lowe


def ridge_leverage(G, keep_frac):
    M = G.shape[0]
    eig = torch.linalg.eigvalsh(G).clamp(min=0)
    K = max(1, round(keep_frac * M))
    lam = eig.sort(descending=True).values[K - 1].item() + 1e-6
    inv = torch.linalg.inv(G + lam * torch.eye(M, device=G.device, dtype=G.dtype))
    return (1 - lam * torch.diagonal(inv)).clamp(min=0)


def per_view_floor(view_id, n_views, score):
    return [int(torch.where(view_id == v)[0][score[torch.where(view_id == v)[0]].argmax()])
            for v in range(n_views)]


# ---------- selectors (return list of global indices) ----------
def sel_strat_random(view_id, n_views, n_tok, keep_pv, seed):
    hw = int(round(n_tok ** 0.5)); half = hw // 2
    g = torch.Generator().manual_seed(seed)
    out = []
    for v in range(n_views):
        base = v * n_tok
        blocks = {0: [], 1: [], 2: [], 3: []}
        for r in range(hw):
            for c in range(hw):
                blocks[(r >= half) * 2 + (c >= half)].append(base + r * hw + c)
        picked = []
        for b in range(4):
            ids = torch.tensor(blocks[b]); k = min(max(1, keep_pv // 4), len(ids))
            picked += ids[torch.randperm(len(ids), generator=g)[:k]].tolist()
        if len(picked) < keep_pv:
            rest = torch.tensor([i for i in range(base, base + n_tok) if i not in set(picked)])
            picked += rest[torch.randperm(len(rest), generator=g)[:keep_pv - len(picked)]].tolist()
        out += picked[:keep_pv]
    return out


def sel_topk_floor(score, view_id, n_views, K, G=None, dedup_tau=None):
    sel = per_view_floor(view_id, n_views, score); selset = set(sel)
    for i in torch.argsort(score, descending=True).tolist():
        if len(sel) >= K:
            break
        if i in selset:
            continue
        if dedup_tau is not None:
            kept = torch.tensor(sel, device=score.device)
            other = kept[view_id[kept] != view_id[i]]
            if other.numel() and G[i, other].max() > dedup_tau:
                continue
        sel.append(i); selset.add(i)
    return sel[:K]


def sel_engine(G, q, view_id, n_views, K):
    M = G.shape[0]; dev = G.device
    forced = per_view_floor(view_id, n_views, q)
    di2 = (q ** 2).clone(); C = torch.zeros(M, K, device=dev); sel = []
    for step in range(K):
        if step < len(forced):
            j = forced[step]
        else:
            masked = di2.clone()
            if sel:
                masked[torch.tensor(sel, device=dev)] = -1e9
            j = int(masked.argmax())
        sel.append(j)
        dj = torch.sqrt(di2[j].clamp(min=1e-12))
        prod = C[:, :step] @ C[j, :step] if step > 0 else torch.zeros(M, device=dev)
        e = (q[j] * q * G[j] - prod) / dj
        C[:, step] = e; di2 = di2 - e ** 2; di2[j] = -1e9
    return sel[:K]


def to_perview(global_idx, view_id, n_views, n_tok):
    g = torch.tensor(sorted(global_idx), device=view_id.device)
    return [(g[view_id[g] == v] % n_tok).sort().values for v in range(n_views)]


def build_prompt_var(model, tokenizer, question, per_view_counts):
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


# ---------- sample collection (dataset-agnostic output: (question, paths, gt)) ----------
def collect(dsname, task, n, image_root, vsi_items, ego_ds):
    out = []
    if dsname == "ego3d":
        for s in ego_ds:
            if s["category"] != task:
                continue
            q = s["question"]
            for opt in (s["options"] or []):
                q += "\n" + opt
            q += NUM_INSTR if task in NUM_CATS else MC_INSTR     # number tasks -> RMSE prompt
            order = RUN.IMAGE_ORDER.get(s["source"], list(s["images"].keys()))
            paths = [os.path.join(image_root, s["images"][v]) for v in order]
            out.append((q, paths, s["answer"]))
            if len(out) >= n:
                break
    else:  # vsi
        items = json.load(open(vsi_items)); root = os.path.dirname(vsi_items)
        for it in items:
            if it["question_type"] != task:
                continue
            nf = len(it["frames"])
            q = f"These are {nf} frames sampled in temporal order from a video scanning a scene.\n\n"
            for i in range(nf):
                q += f"View {i + 1}: <image>\n"
            q += "\n" + it["question"]
            for opt in (it["options"] or []):
                q += "\n" + opt
            q += MC_INSTR
            paths = [os.path.join(root, f) for f in it["frames"]]
            out.append((q, paths, it["ground_truth"]))
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--ratios", default="0.03,0.05")   # keep3 first (decisive)
    ap.add_argument("--methods", default="")            # csv subset of METHODS; "" = all
    ap.add_argument("--configs", default="")            # csv of ds:task; "" = default CONFIGS
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    ap.add_argument("--vsi_items", default="data/vsibench/vsibench_items.json")
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
    ego_ds = load_dataset("vbdai/Ego3D-Bench")["test"]
    os.makedirs("logs/fourway", exist_ok=True)

    def select(method, vit, G, view_id, n_views, n_tok, ratio, seen):
        keep_pv = max(1, round(n_tok * ratio)); K = keep_pv * n_views
        if method == "strat_random":
            gi = sel_strat_random(view_id, n_views, n_tok, keep_pv, args.seed + seen)
        elif method == "leverage":
            gi = sel_topk_floor(ridge_leverage(G, ratio), view_id, n_views, K)
        else:
            q = cornerness(G, view_id, n_views) * lowe_max(G, view_id, n_views)
            if method == "anchor":
                gi = sel_topk_floor(q, view_id, n_views, K, G=G, dedup_tau=0.85)
            else:
                qn = (q - q.min()) / (q.max() - q.min() + 1e-9) + 1e-3
                gi = sel_engine(G, qn, view_id, n_views, K)
        return to_perview(gi, view_id, n_views, n_tok)

    for dsname, task in configs:
        samples = collect(dsname, task, args.n, args.image_root, args.vsi_items, ego_ds)
        for ratio in ratios:
            pct = int(round(ratio * 100))
            for method in methods:
                save = f"logs/fourway/{dsname}.{task}.keep{pct}.{method}.jsonl"
                done = sum(1 for _ in open(save)) if os.path.exists(save) else 0
                for seen, (q, paths, gt) in enumerate(samples, 1):
                    if seen <= done:
                        continue
                    pv, _ = prepare_images_internvl(paths)
                    vit = model.extract_feature(pv)
                    n_views, n_tok, C = vit.shape
                    Fn = torch.nn.functional.normalize(vit.reshape(-1, C).float(), dim=-1)
                    G = Fn @ Fn.t()
                    view_id = torch.arange(Fn.shape[0], device=Fn.device) // n_tok
                    per_view = select(method, vit, G, view_id, n_views, n_tok, ratio, seen)
                    feats = torch.cat([vit[t][per_view[t]] for t in range(n_views)], dim=0)
                    counts = [int(len(per_view[t])) for t in range(n_views)]
                    mi, eos, tmpl = build_prompt_var(model, tokenizer, q, counts)
                    ids = mi["input_ids"].cuda()
                    assert int((ids == model.img_context_token_id).sum()) == feats.shape[0]
                    gen = model.generate(pixel_values=pv, input_ids=ids,
                                         attention_mask=mi["attention_mask"].cuda(), visual_features=feats,
                                         max_new_tokens=1024, do_sample=False, eos_token_id=eos)
                    resp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].split(tmpl.sep.strip())[0].strip()
                    proc = resp.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()
                    with open(save, "a") as f:
                        f.write(json.dumps({"Processed_Pred": proc, "GT": gt, "n_kept": int(feats.shape[0])}) + "\n")
                print(f"[{dsname}/{task} keep{pct} {method}] done", flush=True)

    print("\n=== FOUR-WAY @ EXTREME (ACC, chance=0.25) ===")
    for dsname, task in configs:
        for ratio in ratios:
            pct = int(round(ratio * 100))
            print(f"\n### {dsname}/{task}  keep{pct}%")
            for method in methods:
                save = f"logs/fourway/{dsname}.{task}.keep{pct}.{method}.jsonl"
                if not os.path.exists(save):
                    continue
                rows = [json.loads(l) for l in open(save)]
                acc = sum(1 for r in rows if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower()) / len(rows)
                tok = sum(r["n_kept"] for r in rows) / len(rows)
                print(f"  {method:<13} ACC={acc:.4f}  tok={tok:.0f}  n={len(rows)}")


if __name__ == "__main__":
    main()
