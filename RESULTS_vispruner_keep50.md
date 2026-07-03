# VisPruner vs baseline — InternVL3-8B on Ego3D-Bench

Token-compression comparison on the **`Object_Centric_Absolute_Distance_MultiChoice`**
category (937 multiple-choice samples), single RTX 3090. Baseline = full visual tokens;
VisPruner = keep **50%** of the visual tokens **per view** (256→128 per 448px view,
important:diverse = 50:50, importance = InternViT CLS→patch attention).

Run with `models/internvl3_compress.py` (`--compress_method none|vispruner`); both paths go
through the same instrumented generate so the numbers are comparable.

## Accuracy (ACC ↑, ego3d eval)

| Method | Visual tokens / view | ACC |
|---|---|---|
| Baseline (InternVL3-8B) | 256 (100%) | 0.4952 |
| **VisPruner** | **128 (↓50%)** | **0.5037** |

VisPruner retains **101.7%** of baseline accuracy (+0.85 pp) while discarding half the
visual tokens — consistent with the paper's finding that pruning redundant tokens preserves
(or slightly improves) performance.

## Computational efficiency

| Metric | Baseline | VisPruner | Δ |
|---|---|---|---|
| Visual tokens (mean) | 1548 | 773 | **−50.1%** |
| Input seq len (mean) | 1733 | 958 | −44.7% |
| Prefill FLOPs | 9.70 TFLOPs | 5.21 TFLOPs | **−46.3%** |
| KV-cache | 94.8 MB | 52.4 MB | **−44.7%** |
| Peak GPU memory | 15509 MB | 15356 MB | −1.0% |
| CUDA time / sample | 3869 ms | 3755 ms | −2.9% |
| Encode + prune | 82 ms | 91 ms | +9 ms |

### How the metrics are computed (VisPruner / FastV convention, `utils/efficiency.py`)

- **FLOPs** — theoretical LLM *prefill* FLOPs as a closed form of the token count
  `Σ_layers (4·n·d² + 2·n²·d + 2·n·d·m)` + lm_head. Same formula for both, so the relative
  comparison is exact. This is the term the visual-token reduction directly changes.
- **Storage** — KV-cache bytes for the prefill sequence (`2·L·n·n_kv_heads·head_dim·2B`, bf16).
- **Memory** — measured `torch.cuda.max_memory_allocated` during generation.
- **Time** — measured `cuda.Event` wall time around generation. Visual features (full for
  baseline, pruned for VisPruner) are computed **outside** the timed region so the timer
  isolates the LLM forward (the part token pruning changes); the vision+prune cost is logged
  separately as *encode+prune*.

## Reading the result

The visual-token budget halves, and the **prefill FLOPs (−46%)** and **KV-cache (−45%)**
drop accordingly — this is VisPruner's intended win, with **no accuracy cost** (slightly
better here). **Peak memory and wall-clock barely move** because (a) the 8B weights (~15 GB)
dominate resident memory, so the KV savings (~40 MB) are negligible against it, and (b) greedy
decoding of the answer dominates wall-time, while pruning only shrinks the prefill. The benefit
would grow with longer visual contexts (more/higher-res views), larger batch sizes, or
KV-cache-bound regimes.

## Notes / caveats

- Baseline efficiency means are averaged over the 756 samples generated in the final
  (resumed) session; ACC is over all 937. VisPruner efficiency + ACC are over all 937. The
  per-sample token relationship is exact (VisPruner keeps exactly ⌊256·0.5⌋=128 per view); the
  small deviation from a perfect 2× in the *means* is just a slightly different source mix
  (nuscenes 6 / waymo 5 / argoverse 7 views) across the two averaged sets.
- Importance cue is the InternViT **CLS→patch attention** from the feature layer
  (`select_layer=-1`), aggregated from 1024 patches to the 256 post-pixel-shuffle tokens via
  the model's own `pixel_shuffle` (index-consistent with the embeddings). The ViT feature
  layer is run on the naive attention path to expose the weights (≈ free: +9 ms/sample).
- Logs: `logs/InternVL3-8B-{baseline,vispruner}/` (`*.jsonl` predictions + `efficiency.json`).
- Extensible: add a `TokenCompressor` in `compressors/`, register it, and rerun with
  `--compress_method <name>` — baseline code and this harness are untouched.
