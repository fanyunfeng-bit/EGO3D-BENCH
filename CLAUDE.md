# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **local fork** of Ego3D-Bench (ICLR 2026, arXiv:2509.06266) — a benchmark + post-training
framework for evaluating 3D spatial reasoning of VLMs in ego-centric multi-view driving scenes.
This fork adds the changes needed to run the whole pipeline on a **single RTX 3090 (24 GB)**.
`RUN.md` is the authoritative local run guide; `RESULTS_baseline.md` holds the baseline numbers.
The repo lives under a broader "Multi-View-Compression" research project — this benchmark is used
as a baseline there.

## Environment & commands

All commands run **from the repo root**. The scripts' `python` must be the dedicated `ego3d`
conda env (`transformers==4.51.1`, pinned and incompatible with other envs on this machine):

```bash
conda activate ego3d   # or prefix: PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
```

Baseline (Qwen2.5-VL) and Ego3D-VLM (cognitive-map) runs:

```bash
# Smoke test — 2 samples/category, writes to a *-smoke log dir so it never touches real results:
bash scripts/qwen_2.5_vl.sh         --limit 2 --model_name Qwen2.5-VL-7B-Instruct-smoke
bash scripts/qwen_2.5_vl_ego3dvlm.sh --limit 2 --model_name Qwen2.5-VL-7B-Instruct-smoke

# Full benchmark (all 8,675 samples — many hours to ~a day per config; runs are resumable):
bash scripts/qwen_2.5_vl.sh
bash scripts/qwen_2.5_vl_ego3dvlm.sh
```

There is **no test suite, linter, or build step** — "running a single test" means a `--limit`
smoke run. Scripts evaluate automatically after generation and print per-category Accuracy /
RMSE; to re-score saved logs only, call `eval_logs()` from `utils/eval.py`.

Script knobs (env vars + passthrough flags): `MODEL_PATH`, `MODEL_NAME`, `ATTN`
(`flash_attention_2|sdpa|eager`), `REC_MODEL_PATH`, `DEPTH_MODEL_PATH`, and CLI `--limit`,
`--attn`, `--est_rt`, `--json_cogmap`, `--visual_cogmap`. InternVL3 scripts
(`scripts/internvl3*.sh`) exist but need `--model_path` pointed at a local checkpoint.

## Architecture

**Each `models/*.py` is a self-contained runner** — no shared inference framework. The four
**benchmark** runners span two axes:
- baseline (`qwen2.5_vl.py`, `internvl3.py`) vs **Ego3D-VLM** (`*_ego3dvlm.py`)
- model family: Qwen2.5-VL vs InternVL3

Three more **compression-research** runners (`internvl3_compress.py`, `internvl3_vsibench.py`,
`qwen2.5_vl_vsibench.py`) follow the same self-contained pattern but add the pluggable
token-compressor + efficiency metering — see "Visual-token compression research" below.

Every runner duplicates the **same outer loop**: load `vbdai/Ego3D-Bench['test']` from HF →
order images per source → build prompt → generate (greedy, `max_new_tokens=1024`) → write a
JSONL row → after the loop, evaluate every category. **Changes to that loop (prompt format,
resume logic, output schema, memory patches) must be replicated across all the runner files** —
they do not share the loop, only the helpers in `utils/`.

Shared helpers:
- `utils/common.py` — prompt builders (`convert_to_qwen_input`, `insert_cogmap_to_internvl_input`,
  `strip_question`), `unproject()` (pixel + depth + intrinsics + pose → world coords), image ops.
- `utils/cam_info.py` — **hardcoded** intrinsics/extrinsics per source (nuscenes/waymo/argoverse).
  `est_rt=True` swaps dataset extrinsics for rotations estimated from view angle (`estimate_rt`).
- `utils/eval.py` — scoring (see below).
- `utils/internvl3_utils.py` — InternVL3 image tiling + `split_model` device map.

### Ego3D-VLM perception pipeline (the `*_ego3dvlm.py` runners)

The post-training framework adds, per sample, before the VLM call:
1. Run **Grounding-DINO** (REC) over each view using the question's noun phrases as text labels.
2. Run **Depth-Anything-V2 metric** depth per view; scale by a per-source factor.
3. `unproject()` depth+intrinsics+extrinsics → global 3D coords; take each detection's box center.
4. Build a **cognitive map** of `<view>: detected <object> at 3D location <xyz>` and inject it
   into the prompt. Three formats: textual (default), `--json_cogmap`, `--visual_cogmap`.

### Data conventions (hardcoded, easy to trip on)

- **Two number categories** `['Ego_Centric_Absolute_Distance','Object_Centric_Absolute_Distance']`
  vs **eight multiple-choice** categories. This exact list is duplicated in every runner **twice**
  (prompt selection + eval selection) — keep all occurrences in sync.
- Multiple-choice GT is the option **letter** (A/B). Prompts must ask for "the letter of the
  choice"; an upstream bug asked relative-distance / motion-reasoning for "yes or no", scoring a
  spurious 0.000 (fixed here in `qwen2.5_vl.py` — see `RESULTS_baseline.md`).
- Per-source fixed `image_order` (nuscenes 6 / waymo 5 / argoverse 7 views); sample `images` is
  a dict keyed by view name, resolved against `--image_root` (default `Ego3D-Bench/images`).
- **Resume** = counting existing JSONL lines per category and skipping that many (`<=`, so an
  interrupted sample is retried, not duplicated). Output: `logs/<model_name>/` (baseline) and
  `logs/<model_name>-ego3dvlm/`.
- Eval: multi-choice = exact lowercase match of `<answer>…</answer>` content vs GT; exact-number =
  first regex number, **clamped to 100 m**, RMSE.

## 24 GB GPU memory patches (local fork; numerically identical to upstream)

These are the reason the fork exists — preserve them when editing the runners (details in
`RUN.md` §5):
- **`_LastTokenHead`** wraps `lm_head` to run it only on the last token. Upstream runs `lm_head`
  over the full ~20k-token prompt (5–7 high-res views) → ~6 GiB logits → OOM. Greedy decoding only
  needs the last position, so this is exact. (Qwen runners only.)
- **Per-view perception**: Grounding-DINO + Depth-Anything run one view at a time, then depth maps
  are stacked back to `(V,H,W)`. Upstream batched all views → OOM.
- Scripts export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `HF_HUB_OFFLINE=1`
  (all weights + dataset are cached locally).
- `os.makedirs(..., exist_ok=True)` replaces upstream `os.mkdir` (which crashed when `logs/`
  didn't exist and broke resume).

## Visual-token compression research (CVSP) — the active research layer

This is what the parent "Multi-View-Compression" project actually investigates; the benchmark
above is just the harness. **Question: in multi-view VLM inputs the visual tokens are very
redundant — can an *informed* selection of which tokens to keep beat dropping them at random,
at extreme compression?** The proposed method is **CVSP** (Cross-View Support Pruning): extend
VisPruner from per-view to cross-view, training-free / pose-free / query-agnostic. The `Notes/`
files are the design + empirical archive (read these before proposing method changes — many
approaches have already been tried and falsified, see the verdict at the end of this section).

**Pluggable compressor (`compressors/`)** — a model-agnostic `TokenCompressor.select(importance,
features, keep, seed)` returns the kept-token indices for ONE image, at the granularity that
actually enters the LLM (InternVL3: 256 post-pixel-shuffle tokens/tile). Implementations:
`vispruner` (faithful port: top-k by ViT CLS→patch attention + ToMe diverse tokens) and `random`
(uniform, `needs_importance=False`). Register new methods in `compressors/__init__.py`;
`build_compressor(name)` selects one (`none` = baseline). **Model-specific plumbing lives in the
adapters**, not the compressor: `internvl_adapter.py` (`AttentionCapture` forces one InternViT
layer onto the naive attention path to stash CLS→patch attention; `compute_visual_features`
prunes per tile and returns features for `model.generate(visual_features=...)`) and
`qwen_adapter.py` (Qwen has no CLS / uses windowed attention + a 2×2 merger → patches the
mid full-attention block, uses attention-*received* as the cue, un-permutes via `window_index`).

**Compression runners** — `models/internvl3_compress.py` (Ego3D), `models/internvl3_vsibench.py`
and `models/qwen2.5_vl_vsibench.py` (VSI-Bench). They do NOT modify the baseline runners. The
visual features (full or pruned) are computed *outside* the timed region; the timed region is the
LLM `generate` over the reduced sequence, with `encode_ms` logged separately. Efficiency is
metered by `utils/efficiency.py` (prefill FLOPs / KV-cache bytes / peak GPU mem / CUDA time,
VisPruner/FastV convention).

**Key conventions specific to this layer:**
- `--compress_method {none|vispruner|random}` + `--keep_ratio` (e.g. `0.1` = keep 10% = "keep10",
  the 90%-pruned setting). Log dirs: `logs/<model>-<method>-keep<NN>[-vsibench]/`.
- **Determinism is mandatory**: `random` seeds per view from `SeedSequence([base_seed, sample_id,
  view])`, so a method reproduces identical pruning across reruns and is resume-safe. Same
  resume-by-line-count rule as the benchmark runners.
- **Score Ego3D by ACC, not RMSE.** Pruning *improves* RMSE as an artifact (regression to the
  mean / fewer clamped outliers) while ACC drops — RMSE is misleading on these tasks.
- VSI-Bench must be prepped first: `python scripts/prep_vsibench.py --sources …` samples N uniform
  frames/video into `data/vsibench/`. Treat the frames as the multi-view input.

**Diagnostic / experiment scripts (`scripts/*.py`, not part of the benchmark)** — each is a
standalone study, resumable per-(task,method) JSONL, run under the `ego3d` env. They set
`torch.set_grad_enabled(False)` after model load (`.eval()` alone OOMs on ViT activations):
- `leverage_diagnostic.py` — the go/no-go "体检": effective rank, ridge-leverage Gini, coherence
  of the token Gram vs a random-Gaussian null. (Finding: effective rank ≈84 ≪ keep10 budget,
  so random over-covers the subspace — selection can only matter at keep ≤5%.)
- `four_way_extreme.py` — stratified-random vs anchor (cornerness × cross-view Lowe) vs
  ridge-leverage vs quality-weighted log-det "engine", at keep5/keep3 on Ego3D + VSI.
- `vision_ablation.py` — real / keep10 / black-image / noise-feature controls (does the model use
  vision at all). `redundancy_analysis.py` + `p4_fps_causal.py` — cross-view redundancy/coverage
  metrics and the FPS causal test.

**Current empirical verdict (2026-06-18; authoritative source = `Notes/Visual-Compression.md`
§实测发现 D–J and `Notes/CVSP-Story.md` 实验现状):** dropping 90% of visual tokens barely hurts
Ego3D ACC; **no query-agnostic informed selector (saliency / diversity / leverage / anchor /
log-det engine) reliably beats stratified random** across budgets and both datasets. The vision
ablation shows the visual signal is small and redundant and the model leans heavily on language
priors. So CVSP currently has **no evidence as an accuracy method**; the defensible paper is the
efficiency + counter-intuitive-findings story. The one live lead is `anchor` on high-overlap VSI
spatial tasks (modest, inconsistent). The `Notes/` design docs carry ⚠️ time-banners marking the
older "compression also *improves* accuracy" framing as superseded — **don't resurrect a
falsified approach; check the verdict first.**
