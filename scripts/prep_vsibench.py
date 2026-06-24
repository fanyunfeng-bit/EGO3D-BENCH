"""Prepare VSI-Bench for the compression harness.

Downloads VSI-Bench (QA jsonl + per-source video zips) from HF, samples
`--frames` frames UNIFORMLY per referenced video (treated as multi-view frames),
caches them to data/vsibench/frames/<source>__<scene>/v{0..k-1}.jpg, and writes
data/vsibench/vsibench_items.json over ALL 10 question types (MC + numeric).

Usage (from repo root, ego3d env):
  python scripts/prep_vsibench.py --sources scannetpp                 # quick (cached ~1GB)
  python scripts/prep_vsibench.py --sources scannet,scannetpp,arkitscenes  # full (~5.3GB)
"""

import argparse
import glob
import json
import os
import zipfile

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "nyu-visionx/VSI-Bench"
ROOT = "data/vsibench"
ALL_SOURCES = ["scannet", "scannetpp", "arkitscenes"]


def uniform_frame_indices(n_total, k):
    """k uniformly-spaced frame indices over [0, n_total-1]."""
    if k <= 0 or n_total <= 0:
        return []
    if k >= n_total:
        return list(range(n_total))
    return [int(round(x)) for x in np.linspace(0, n_total - 1, k)]


def ensure_videos(source):
    dst = f"{ROOT}/videos/{source}"
    if glob.glob(f"{dst}/**/*.mp4", recursive=True):
        return dst
    print(f"[{source}] downloading {source}.zip (big step) ...", flush=True)
    zpath = hf_hub_download(REPO, f"{source}.zip", repo_type="dataset")
    os.makedirs(dst, exist_ok=True)
    print(f"[{source}] extracting ...", flush=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dst)
    return dst


def find_video(vid_dir, scene):
    hits = glob.glob(f"{vid_dir}/**/{scene}.mp4", recursive=True)
    return hits[0] if hits else None


def extract_frames(video_path, out_dir, k):
    cached = sorted(glob.glob(f"{out_dir}/v*.jpg"))
    if len(cached) >= k:
        return cached[:k]
    import decord
    os.makedirs(out_dir, exist_ok=True)
    vr = decord.VideoReader(video_path)
    idxs = uniform_frame_indices(len(vr), k)
    frames = vr.get_batch(idxs).asnumpy()  # (k, H, W, 3)
    paths = []
    for j, fr in enumerate(frames):
        p = f"{out_dir}/v{j}.jpg"
        Image.fromarray(fr).save(p, quality=95)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(ALL_SOURCES),
                    help="comma-sep subset of: scannet, scannetpp, arkitscenes")
    ap.add_argument("--frames", type=int, default=6, help="frames sampled per video (uniform)")
    ap.add_argument("--max-scenes", type=int, default=0, help="cap distinct scenes per source (0=all)")
    ap.add_argument("--out", default=f"{ROOT}/vsibench_items.json")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, "test.jsonl", repo_type="dataset"))]
    rows = [r for r in rows if r["dataset"] in sources]
    print(f"{len(rows)} questions in sources={sources}, frames/scene={args.frames}")

    vid_dirs = {s: ensure_videos(s) for s in sources}
    frame_cache, scenes_per_src, items, skipped = {}, {}, [], 0
    for r in rows:
        src, scene = r["dataset"], r["scene_name"]
        tag = f"{src}__{scene}"
        if tag not in frame_cache:
            seen = scenes_per_src.setdefault(src, set())
            if args.max_scenes and scene not in seen and len(seen) >= args.max_scenes:
                continue
            seen.add(scene)
            vp = find_video(vid_dirs[src], scene)
            if vp is None:
                print(f"  WARN: no video for {tag}"); frame_cache[tag] = None; continue
            try:
                paths = extract_frames(vp, f"{ROOT}/frames/{tag}", args.frames)
                frame_cache[tag] = [os.path.relpath(p, ROOT) for p in paths]
            except Exception as e:
                print(f"  WARN: frame extract failed {tag}: {str(e)[:80]}"); frame_cache[tag] = None
        if not frame_cache.get(tag):
            skipped += 1; continue
        items.append({
            "index": r["id"], "dataset": src, "scene": scene,
            "question_type": r["question_type"], "question": r["question"],
            "options": r["options"], "ground_truth": str(r["ground_truth"]),
            "frames": frame_cache[tag],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(items, open(args.out, "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"wrote {len(items)} items ({skipped} skipped) -> {args.out}")
    print("by question_type:", dict(Counter(i["question_type"] for i in items)))


if __name__ == "__main__":
    main()
