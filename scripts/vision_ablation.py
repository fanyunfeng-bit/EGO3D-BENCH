"""Vision-ablation control: is the model actually USING the visual tokens, or
answering from language priors / question text?

Runs the InternVL3 Ego3D MC path under 4 conditions on a matched sample subset:
  real_full   : full real visual tokens (256/view)             -> reference
  keep10_visp : VisPruner-pruned to 10% (26/view)              -> the compression
  black_full  : all-black images through the encoder (256/view)-> "no scene content"
  noise_full  : real features replaced by scale-matched Gaussian noise (256/view)

Decisive read (chance = 0.25 on these 4-way MC tasks):
  black/noise -> ~chance  => model uses vision; keep10 staying high = redundancy.
  black/noise -> ~baseline => model ignores vision; "lossless compression" is moot.

Resumable: per-(task,condition) jsonl, skips already-written lines.
"""
import argparse, json, os, sys
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import extract_number_from_answer_tag_mult_choice
from utils.internvl3_utils import prepare_images_internvl, split_model, IMAGENET_MEAN, IMAGENET_STD
from compressors import build_compressor
from compressors.internvl_adapter import AttentionCapture, compute_visual_features
import models.internvl3_compress as R

CONDITIONS = ["real_full", "keep10_visp", "black_full", "noise_full"]
# black RGB(0,0,0) after ImageNet normalisation -> per-channel constant
BLACK = [(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="OpenGVLab/InternVL3-8B")
    ap.add_argument("--tasks", default="Ego_Centric_Absolute_Distance_MultiChoice,"
                                       "Object_Centric_Absolute_Distance_MultiChoice")
    ap.add_argument("--n", type=int, default=250, help="matched subset size per task")
    ap.add_argument("--image_root", default="Ego3D-Bench/images")
    args = ap.parse_args()
    tasks = args.tasks.split(",")

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=split_model(args.model_path),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    IMG_CTX_ID = tokenizer.convert_tokens_to_ids(R.IMG_CTX)
    model.img_context_token_id = IMG_CTX_ID
    full_per_tile = model.num_image_token

    visp = build_compressor("vispruner", important_ratio=0.5)
    capture = AttentionCapture(model)  # for VisPruner importance

    out_dir = "logs/ablation"
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset("vbdai/Ego3D-Bench")["test"]

    @torch.no_grad()
    def feats_for(cond, pixel_values, n_views, seen):
        """Return (visual_features, tok_per_tile) for a condition."""
        if cond == "keep10_visp":
            return compute_visual_features(model, pixel_values, visp, 0.10, capture,
                                           base_seed=42, sample_id=seen)
        if cond == "black_full":
            blk = torch.empty_like(pixel_values)
            for c in range(3):
                blk[:, c, :, :] = BLACK[c]
            vit = model.extract_feature(blk)
            return vit.reshape(-1, vit.shape[-1]), full_per_tile
        # real features (used by real_full and noise_full)
        vit = model.extract_feature(pixel_values)
        C = vit.shape[-1]
        real = vit.reshape(-1, C)
        if cond == "noise_full":
            noise = torch.randn_like(real) * real.std() + real.mean()
            return noise, full_per_tile
        return real, full_per_tile

    @torch.no_grad()
    def run(task, cond):
        save = f"{out_dir}/{task}.{cond}.jsonl"
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
            order = R.IMAGE_ORDER.get(sample["source"], list(sample["images"].keys()))
            paths = [os.path.join(args.image_root, sample["images"][v]) for v in order]
            pv, npl = prepare_images_internvl(paths)
            n_views = sum(npl)
            vf, tpt = feats_for(cond, pv, n_views, seen)
            mi, eos, tmpl = R.build_prompt(model, tokenizer, q, npl, tpt)
            ids = mi["input_ids"].cuda()
            gen = model.generate(pixel_values=pv, input_ids=ids,
                                 attention_mask=mi["attention_mask"].cuda(), visual_features=vf,
                                 max_new_tokens=1024, do_sample=False, eos_token_id=eos)
            resp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            resp = resp.split(tmpl.sep.strip())[0].strip()
            proc = resp.split("<answer>")[-1].split("</answer>")[0].replace("\n", "").strip()
            with open(save, "a") as f:
                f.write(json.dumps({"Processed_Pred": proc, "GT": sample["answer"]}) + "\n")

    def acc(task, cond):
        save = f"{out_dir}/{task}.{cond}.jsonl"
        if not os.path.exists(save):
            return None, 0
        rows = [json.loads(l) for l in open(save)]
        if not rows:
            return None, 0
        c = sum(1 for r in rows
                if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower())
        return c / len(rows), len(rows)

    for task in tasks:
        for cond in CONDITIONS:
            print(f">>> {task} [{cond}]", flush=True)
            run(task, cond)

    print("\n================ VISION ABLATION RESULTS (chance=0.25) ================")
    for task in tasks:
        print(f"\n### {task}")
        print(f"{'condition':<14} {'ACC':>7} {'n':>5}")
        for cond in CONDITIONS:
            a, n = acc(task, cond)
            print(f"{cond:<14} {a:>7.4f} {n:>5}" if a is not None else f"{cond:<14}  (none)")


if __name__ == "__main__":
    main()
