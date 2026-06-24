# Ego3D-Bench — Local Run Guide

Complete, reproducible pipeline for **"Spatial Reasoning with Vision-Language Models in
Ego-Centric Multi-View Scenes"** (ICLR 2026, arXiv:2509.06266), set up on this machine
(single **RTX 3090, 24 GB**).

This file documents the environment, the assets that were downloaded, the small code fixes
that were required to run on a 24 GB GPU, and the exact commands for smoke tests and full runs.

---

## 1. Environment

A dedicated conda env was created (the repo pins `transformers==4.51.1`, which is incompatible
with the other envs on this machine that ship transformers 5.x):

```bash
conda create -y -n ego3d python=3.10
conda activate ego3d
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt          # everything except flash-attn
# flash-attn: installed from the official prebuilt wheel (no source compile needed):
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Verified stack: `torch 2.6.0+cu124`, `transformers 4.51.1`, `datasets 4.0.0`,
`flash_attn 2.7.4.post1`, `numpy 1.26.4`. GPU compute capability `8.6` (Ampere).

> The `python` used by the `scripts/*.sh` below must be the `ego3d` env. Either
> `conda activate ego3d` first, or prefix commands with
> `PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH`.

---

## 2. Data & model assets (already downloaded)

| Asset | Location | Notes |
|---|---|---|
| QA data (8,675 samples, 10 categories) | auto via `load_dataset("vbdai/Ego3D-Bench")` | cached under `~/.cache/huggingface` |
| Raw images (1,628 jpgs) | `Ego3D-Bench/images/` | extracted from `raw_images.zip` (483 MB) |
| Qwen2.5-VL-7B-Instruct | HF cache (`Qwen/Qwen2.5-VL-7B-Instruct`) | baseline VLM |
| Grounding-DINO (REC) | HF cache (`IDEA-Research/grounding-dino-base`) | Ego3D-VLM only |
| Depth-Anything-V2 metric | HF cache (`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`) | Ego3D-VLM only |

Sources: `nuscenes` (6 views), `waymo` (5 views), `argoverse` (7 views).
Categories (10): ego/object-centric × {absolute distance (number), absolute-distance MC,
relative distance, motion reasoning} + `Localization` + `Travel_Time`.

---

## 3. Running

All commands are run **from the repo root**. Results are written to `logs/<MODEL_NAME>/<category>.jsonl`
and per-category **Accuracy** (multi-choice) / **RMSE** (exact-number) is printed at the end.
Runs are **resumable** — each category counts already-processed lines and continues.

### 3a. Baseline benchmark (Qwen2.5-VL)

```bash
# Smoke test — 2 samples/category, ~4 min (writes to a separate *-smoke log dir):
bash scripts/qwen_2.5_vl.sh --limit 2 --model_name Qwen2.5-VL-7B-Instruct-smoke

# Full benchmark (all 8,675 samples):
bash scripts/qwen_2.5_vl.sh
```

### 3b. Ego3D-VLM framework (cognitive-map post-training)

Builds a textual cognitive map from Grounding-DINO detections + Depth-Anything metric depth
(unprojected to global 3D coords) and injects it into the VLM prompt.

```bash
# Smoke test — 2 samples/category:
bash scripts/qwen_2.5_vl_ego3dvlm.sh --limit 2 --model_name Qwen2.5-VL-7B-Instruct-smoke

# Full run:
bash scripts/qwen_2.5_vl_ego3dvlm.sh
```

Cognitive-map variants (default is textual): add `--json_cogmap` or `--visual_cogmap`.
Use estimated extrinsics instead of dataset extrinsics: add `--est_rt`.

### Re-evaluating existing logs

The scripts evaluate automatically after generation. To recompute from saved logs only,
see `utils/eval.py` (`eval_logs`).

---

## 4. Useful flags (added for this setup)

| Flag | Scripts | Purpose |
|---|---|---|
| `--limit N` | both | Process only the first N samples per category (smoke testing). |
| `--attn {flash_attention_2,sdpa,eager}` | both | Attention backend. Default `flash_attention_2`. Use `sdpa` if flash-attn is unavailable. |
| `MODEL_PATH=...` (env) | both | Override the VLM checkpoint/path. |
| `ATTN=sdpa` (env) | both | Same as `--attn` via env var. |

InternVL3 scripts (`scripts/internvl3*.sh`) exist too; point `--model_path` at a local
InternVL3 checkpoint. Note InternVL3-8B + Grounding-DINO + Depth-Anything may be tight on 24 GB.

---

## 5. Changes made to the upstream code (and why)

These were necessary/robustness fixes; default behavior is otherwise unchanged.

1. **`os.mkdir` → `os.makedirs(..., exist_ok=True)`** in all four `models/*.py`.
   Upstream crashed with `FileNotFoundError: 'logs/<name>'` because `logs/` did not exist
   (`os.mkdir` creates only one level). Now the log dir tree is created and runs are re-runnable.

2. **Per-view perception in the Ego3D-VLM scripts** (`models/qwen2.5_vl_ego3dvlm.py`,
   `models/internvl3_ego3dvlm.py`). Upstream ran Grounding-DINO + Depth-Anything over a
   *batch* of all 5-7 high-res views at once, which OOMs on 24 GB (Qwen ~16.5 GB resident +
   batched deformable-attention spike > 24 GB). Now each view is processed one at a time and
   the per-view depth maps are stacked back into the original `(V, H, W)` tensor — **identical
   numerical results**, lower peak memory. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   is also exported by the Ego3D-VLM launch script.

3. **`--attn` and `--limit` CLI flags** added to the Qwen scripts (opt-in; defaults preserve
   upstream behavior).

4. **Last-token `lm_head` patch** in both Qwen scripts (`_LastTokenHead`). This transformers
   version's `Qwen2_5_VL.forward` has no `logits_to_keep` argument and runs `lm_head` over the
   **full prompt**. With 5-7 high-res views a prompt is ~20k tokens, so the logits tensor is
   `[1, 20k, 152064]` ≈ 6 GiB and OOMs on a 24 GB GPU (observed at ~59% of the baseline run).
   Greedy decoding only ever uses the last position, so the patch slices `hidden_states[:, -1:, :]`
   before `lm_head` — **numerically identical output, no image downscaling**. Validated on a
   full-res 7-view (20,887-token) sample: peak memory 20.05 GiB vs OOM before. The scripts also
   export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `HF_HUB_OFFLINE=1` (all assets
   are cached). Runs are resumable (`os.makedirs` + per-category line counting; the resume
   skip condition was corrected from `<` to `<=` so the interrupted sample is retried, not
   duplicated).

---

5b. **Baseline prompt bug fixed** (`models/qwen2.5_vl.py`). Upstream told the
`Ego_Centric_Relative_Distance`, `Ego_Centric_Motion_Reasoning`, and `Object_Centric_Motion_Reasoning`
categories to answer **"yes or no"**, but their ground truth is the multiple-choice **letter**
(A/B), so the eval scored a spurious **0.000** for all three. Fixed to ask for "the letter of the
choice" for every multiple-choice category (matching the GT and the Ego3D-VLM script). The first
full run was done before this fix; its three affected categories were corrected post-hoc by a
yes/no→letter remap. See `RESULTS_baseline.md`.

## 6. Expected runtime (single RTX 3090)

Greedy decoding, up to 1024 new tokens/sample over 5-7 images. Rough order-of-magnitude:
~10 s/sample baseline, ~15-20 s/sample for Ego3D-VLM (extra perception passes). The full
8,675-sample benchmark therefore takes **many hours to ~a day** per configuration — run in the
background (`run_in_background`/`nohup`/`tmux`); the built-in resume lets you stop and continue.
