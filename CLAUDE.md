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
- Multiple-choice GT is the option **letter**, but the option count differs by category, so the
  **chance baseline differs** — this matters when comparing to the paper:
  - **4-option (A/B/C/D, chance 25%):** `*_Absolute_Distance_MultiChoice`, `Localization`,
    `Travel_Time`.
  - **binary (A/B, chance 50%):** `*_Relative_Distance`, `*_Motion_Reasoning`.
  Prompts must ask for "the letter of the choice"; an upstream bug asked relative-distance /
  motion-reasoning for "yes or no", scoring a spurious 0.000 (fixed here in `qwen2.5_vl.py` — see
  `RESULTS_baseline.md`). NB: our `*_Absolute_Distance_MultiChoice` ACC (~0.50 for InternVL3-8B) is
  ~2× the paper's ~0.26–0.29; that is **not** a scoring bug (scoring verified: 4-way, GT letter
  ~uniform, clean single-letter parse, exact match) and **not** the CoT prompt (a no-`<think>`
  single-letter run scores the same ~0.40–0.54 — tested). Root cause = the released HF dataset's
  **MC distractors are uncalibrated**: the GT distance clusters near ~14–16 m and is almost never
  the largest option (Ego 3% / Object 8% of GTs are the max value), so a **value-only prior that
  never looks at the image** already scores ~0.40–0.55 ("closest-to-global-mean" = 0.55 on Ego,
  "2nd-smallest" = 0.48). The paper's ≈chance numbers imply **balanced distractors** that neutralize
  this prior. Consistent with the vision ablation (black-image still 0.31–0.39 ≫ 0.25; real images
  add only ~9–12 pp) → these MC ACCs are largely a distractor artifact + language prior, not metric
  spatial ability. See `logs/ablation/` and the verdict below.
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
at extreme compression?** The original method is **CVSP** (Cross-View Support Pruning): extend
VisPruner from per-view to cross-view, training-free / pose-free / query-agnostic. After CVSP's
query-agnostic line was found to not beat random (verdict below), the active direction pivoted to
**query-aware staged compression** (working name **GeoScaffold** / "two-stage"): a cacheable
query-agnostic cross-view 3D scaffold pre-LLM, then a query-conditional coarse-to-fine refine
*inside* the LLM. The `Notes/` files are the design + empirical archive — **read these before
proposing method changes** (`CVSP-Method.md` = current spec §12 single-stage / §13 two-stage;
`SCMPruner-Method.md` = the pure-feature multi-view pruner spec derived from the anchor finding
(anchor `support×sharpness` no-dedup + margin-aware saliency dedup + facility-location coverage);
`GeoScaffold-Story.md` = the pivot; `Visual-Compression.md` + `CVSP-Story.md` = empirical log;
`Anchor-Validation.md` = the open track to validate whether cross-view "load-bearing" tokens
exist / where the model binds across views, via clean probes + a geometry oracle instead of the
contaminated MC ACC — incl. a probing→LoRA "activate unused spatial info" agenda);
many approaches have already been tried and falsified, see the verdict at the end of this section.

**`docs/算法运行指南.md` is the authoritative run guide for this whole compression layer** (every
method's exact command, params, output path, and scoring) — keep it and this section in sync.

**Two harnesses — and the headline methods live in only one of them.** This is the most common
trip-up:
- **Harness A** — `models/internvl3_compress.py` (Ego3D), `models/internvl3_vsibench.py` and
  `models/qwen2.5_vl_vsibench.py` (VSI). Selected with `--compress_method`. `internvl3_compress.py`
  is **registry-only** (`none|vispruner|random`); the two **VSI runners additionally accept
  `scmpruner` and `fastv`** (a `SPECIAL_METHODS` set handled *outside* the registry — `scmpruner`
  calls `compressors/scm.py::scmpruner_keep_indices` pre-LLM, `fastv` installs an in-LLM layer-K
  pruner from `compressors/fastv.py`). This is the **efficiency + simple-baseline** harness:
  it writes `*.result.json` with FLOPs/KV/mem/time. Visual features are computed *outside* the
  timed region; the timed region is the LLM `generate` over the reduced sequence (`encode_ms`
  logged separately). Metered by `utils/efficiency.py` (VisPruner/FastV convention). NB: the VSI
  runners use a **no-`<think>` prompt + `max_new_tokens=16` + robust `\b[a-d]\b` MC scoring**
  (commit `c62da1b`), diverging from the benchmark loop's `max_new_tokens=1024`.
- **Harness B** — `scripts/cvsp_curve.py` (hosts `cvsp`=a20s40, `block_cvsp`, `scmpruner`, plus
  baselines) and
  `scripts/qstage_curve.py` (the **two-stage query-aware** method). Selected with `--methods` /
  `--signal`, **not** `--compress_method`. Writes **accuracy-only** JSONL to `logs/cvsp/`. The
  headline methods (a20s40, block-cvsp, two-stage) exist **only here** — they are not in the
  registry and cannot be run via `--compress_method`. `scripts/four_way_extreme.py` is the shared
  backbone (sample `collect`, prompt build, `cornerness`/`lowe_max`, selectors) imported by both.

**Pluggable compressor (`compressors/`, used by Harness A only)** — a model-agnostic
`TokenCompressor.select(importance, features, keep, seed)` returns the kept-token indices for ONE
image, at the granularity that actually enters the LLM (InternVL3: 256 post-pixel-shuffle
tokens/tile). Registry implementations: `vispruner` (faithful port: top-k by ViT CLS→patch
attention + ToMe diverse tokens) and `random` (uniform, `needs_importance=False`). Register new
per-image selectors in `compressors/__init__.py`; `build_compressor(name)` selects one (`none` =
baseline). **Three modules in `compressors/` are deliberately *not* in the registry** (they are
cross-view or in-LLM, not per-image `TokenCompressor`s): `scm.py` = the **SCMPruner core**
(`scmpruner_keep_indices` — the single source of truth for the `support×sharpness` anchor +
three-bucket selector, imported by BOTH `cvsp_curve.py` and the two VSI runners so InternVL3 and
Qwen prune byte-identically); `fastv.py` = **FastV** (per-view in-LLM prune at layer K by
last-token attention, one class per model family); and `qstage_llm.py` = the in-LLM two-stage
controller (`QStage` + `make_qstage_forward` patch InternVL3's `Qwen2Model.forward` to prune
vision tokens at decoder layer K with PESP position handling) used by `qstage_curve.py`.
**Model-specific plumbing lives in the adapters**, not the compressor: `internvl_adapter.py`
(`AttentionCapture` forces one InternViT layer onto the naive attention path to stash CLS→patch
attention; `compute_visual_features` prunes per tile and returns features for
`model.generate(visual_features=...)`) and `qwen_adapter.py` (Qwen has no CLS / uses windowed
attention + a 2×2 merger → patches the mid full-attention block, uses attention-*received* as the
cue, un-permutes via `window_index`).

**Key conventions specific to this layer:**
- Harness A: `--compress_method {none|vispruner|random}` (VSI runners add `scmpruner|fastv`) +
  `--keep_ratio` (e.g. `0.1` = keep 10% = "keep10", the 90%-pruned setting). Log dirs:
  `logs/<model>-<method>-keep<NN>[-vsibench]/`.
- Harness B: `--ratios` (comma-sep keep fractions) + `--methods`/`--signal`; for the **`TAGGED`
  methods (`cvsp`, `block_cvsp`, `scmpruner`) always pass the same `--tag`** when re-running or you
  won't find/score the file. Output: `logs/cvsp/<ds>.<task>.keep<pct>.<method><tag>.jsonl`.
  `--methods cvsp` = a20s40 (ρ_a=0.2/ρ_s=0.4, 3-bucket anchor/saliency/coverage);
  `--methods scmpruner` = **SCMPruner** (pure-feature `support×sharpness` anchor, no cross-view
  dedup; margin-aware saliency dedup; facility-location coverage). Its **primary knob is `--anc_m`**
  (Lowe-margin / sharpness gate, default 0.12 — the project's most sensitive knob); secondary
  `--anc_tau` (0.6, "clear match" cosine gate), `--scm_rho_a/--scm_rho_s` (0.2/0.4 = the canonical
  **20/40/40** split), and `--scm_xview` (cross-view coverage propagation, default on — **ablated
  to a no-op**). **All five knobs are exposed identically (same names + defaults) in all three
  harnesses** — `cvsp_curve.py` and both VSI runners — routing into the one shared selector in
  `compressors/scm.py` (`--rho_a/--rho_s` in `cvsp_curve.py` stay reserved for `cvsp`, not
  `scmpruner`). The **VSI runners auto-encode any *non-default* knob into the log-dir name** via
  `scm.scmpruner_tag_suffix` (e.g. `…-scmpruner-keep10-m08-vsibench`, `…-noxv`) so a sweep never
  collides / corrupts resume, and the default config keeps the bare `…-scmpruner-keep<NN>-vsibench`
  dir; `cvsp_curve.py` keeps its manual `--tag`. Spec: `Notes/SCMPruner-Method.md` (§6's "ρ=1/3
  fixed" is superseded by §12.3's move to 20/40/40). `qstage_curve.py --signal input_cos --r 7` =
  the headline two-stage. `utils/aggregate_compress.py` collates Harness-A results. A
  query-aware two-stage variant `scmpruner_qa` (SCMPruner stage-1 → in-LLM layer-K query prune,
  no anchor protection; `--scm_r/--scm_K/--scm_sig/--scm_softweight`) runs on both VSI runners —
  spec `docs/superpowers/specs/2026-07-06-query-aware-scmpruner-design.md`.
- **Determinism is mandatory**: `random` seeds per view from `SeedSequence([base_seed, sample_id,
  view])`, so a method reproduces identical pruning across reruns and is resume-safe. Resume is by
  JSONL line count, and sample order is deterministic, so an `--n 200` file is a true prefix of the
  `--n 99999` (full) file and is auto-extended, not re-run.
- **Score Ego3D by ACC, not RMSE.** Pruning *improves* RMSE as an artifact (regression to the
  mean / fewer clamped outliers) while ACC drops — RMSE is misleading on these tasks.
- VSI-Bench must be prepped first: `python scripts/prep_vsibench.py --sources …` samples N uniform
  frames/video into `data/vsibench/`. Treat the frames as the multi-view input. Keep `--frames`
  consistent between prep and the VSI runners.

**Diagnostic / experiment scripts (`scripts/*.py`, not part of the benchmark)** — each is a
standalone study, resumable per-(task,method) JSONL, run under the `ego3d` env. They set
`torch.set_grad_enabled(False)` after model load (`.eval()` alone OOMs on ViT activations):
- `leverage_diagnostic.py` — the go/no-go "体检": effective rank, ridge-leverage Gini, coherence
  of the token Gram vs a random-Gaussian null. (Finding: effective rank ≈84 ≪ keep10 budget,
  so random over-covers the subspace — selection can only matter at keep ≤5%.)
- `four_way_extreme.py` — stratified-random vs anchor (cornerness × cross-view Lowe) vs
  ridge-leverage vs quality-weighted log-det "engine", at keep5/keep3 on Ego3D + VSI. **Also the
  shared backbone imported by the Harness-B curve runners** (`cvsp_curve.py` / `qstage_curve.py`).
- `anchor_spread.py` — how concentrated block-cvsp Layer-1 anchors are across views (finding:
  anchors already spread, so explicit cross-view pairing isn't worth it).
- `vision_ablation.py` — real / keep10 / black-image / noise-feature controls (does the model use
  vision at all). `redundancy_analysis.py` + `p4_fps_causal.py` — cross-view redundancy/coverage
  metrics and the FPS causal test.
- `eval_cvsp.py` / `eval_sweep1.py` / `eval_stage2.py` — Harness-B **viewers + go/no-go gates**:
  `eval_cvsp.py` prints the per-config ACC table over `logs/cvsp/` (safe to run mid-curve);
  `eval_sweep1.py` / `eval_stage2.py` encode the sign-test decisions (does a cvsp setting beat
  VisPruner **and** random on ≥N/M task×budget cells) and write `logs/cvsp/*.decision.json`.
  The many `scripts/run_*.sh` are thin wrappers that chain these curve/sweep/decision steps —
  `docs/算法运行指南.md` is authoritative for which to run.

**Current empirical verdict (2026-06-24, anchor line 2026-07-01, SCMPruner line 2026-07-05;
authoritative source = `Notes/Visual-Compression.md` §实测发现 D–J + §O full-data table,
`Notes/CVSP-Story.md` 实验现状, `Notes/Anchor-Validation.md` §9–11, and
`Notes/SCMPruner-Method.md` §12):** dropping 90% of visual
tokens barely hurts Ego3D ACC, and the vision ablation shows the visual signal is small and
redundant while the model leans heavily on language priors. Across budgets and both datasets, **no
informed selector reliably beats stratified random — and this now extends to the query-aware
methods, not just the query-agnostic ones:**
- Query-agnostic line (saliency / diversity / leverage / anchor / log-det engine, and the tuned
  single-stage **a20s40**): overall ≈ or < random.
- **SCMPruner (`Notes/SCMPruner-Method.md` §12, 2026-07-05, Qwen2.5-VL 16-frame no-think VSI, 5
  cross-view relational tasks):** the pure-feature `support×sharpness` pruner lands **SCM ≈ random
  ≈ VisPruner** (per-method mean ACC within <0.3pp; VisPruner marginally best) and only clearly
  **beats FastV (+0.011)**. The 20/40/40 ρ split fixed a20s40's earlier keep10 regression (which
  lost to random by −0.021) → now a statistical tie. Two counter-intuitive notes: **compression
  itself hurts** (full-token baseline is best overall, ~0.384 vs 0.35–0.37), and SCM's edge over
  random is *largest at keep25 (+0.013) and reverses at keep5 (−0.012)* — opposite to the
  "selection only matters at extreme compression" thesis. The **`--scm_xview` cross-view coverage
  propagation ablated to a no-op** (mean Δ 0.09pp) — removed from the "has potential" list.
- **Anchor existence — refined 2026-07-01 (`Notes/Anchor-Validation.md` §9–11).** The flat "anchor
  doesn't help" is too strong: a *no-dedup* geometric oracle (VGGT cross-view co-visibility × surface
  variation, injected as `ρ_a` anchors + rest per-view-random) gives a **dose-dependent +~0.02 ACC on
  cross-view *relational* tasks** (`object_rel_direction/distance`), washing out on
  coverage/counting. So load-bearing cross-view tokens **do exist** — earlier NO-GO was a
  cross-view-dedup artifact (dedup destroys anchor multiplicity; **never dedup anchors across views**).
  A **pure-feature** anchor `support × sharpness` (support = #views with cross-view cosine `s1>τ` &
  Lowe `margin>m`; best **τ0.6/m0.12**) reaches ~2/3 of the oracle (+0.016–0.022) and **beats
  a20s40's `cornerness × lowe_max`** (which ≈ random) head-to-head — but is still **within ~1 SE,
  τ/m-sensitive, and does not beat random overall**. Mechanism: margin median ≈0.019, so only ~3% of
  cross-view matches are sharp/unique (real landmarks) → gains are inherently small. Defensible claim:
  *"anchors exist and `support×sharpness` > `cornerness×lowe_max`,"* **not** *"beats random."*
- Query-aware **two-stage** (`input_cos r7`, the GeoScaffold direction) on full Ego3D (§O): beats
  **VisPruner 8/14** and **greatly improves absolute-distance** (Object_AbsD RMSE 13.9 vs 17.5),
  but **does not beat `plain_random` overall** (5/14); it wins the absolute-distance family and
  loses the relative/localization family (query pruning sacrifices coverage). Single-stage a20s40
  ≥ two-stage (5/7 at keep10) — the extra in-LLM query prune gave no net gain.

So the defensible empirical claim is **"beats VisPruner + improves absolute-distance,"** NOT "beats
random." The **"efficiency / counter-intuitive-findings" paper was explicitly withdrawn as a safety
net (2026-06-18 decision)** — the project's accuracy story now rests on actually beating random in
the extreme-compression regime (keep ≤5%, where the budget < effective rank ~84). The `Notes/`
design docs carry ⚠️ time-banners marking the older "compression also *improves* accuracy" framing
as superseded — **don't resurrect a falsified approach; check the verdict (and §O) first.**
