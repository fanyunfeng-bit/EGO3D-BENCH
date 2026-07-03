# VSI-Bench — baseline vs VisPruner@25% vs random@25% (InternVL3-8B)

Token-compression comparison on **VSI-Bench**, InternVL3-8B, **6 uniformly-sampled
frames/video** presented as `View 1..6` with our R1-style CoT prompt. Each frame =
256 visual tokens → 1536 total; keep 25% → 384. Methods run through the same
harness as Ego3D (`models/internvl3_vsibench.py`, `compressors/`). Metrics per
VSI-Bench: **Accuracy** for multiple-choice, **MRA** (mean relative accuracy over
thresholds {0.5..0.95}) for numeric. Recorded per task in
`logs/InternVL3-8B-<method>-vsibench/<task>.{jsonl,result.json}`.

## Full test — two smallest MC tasks

### route_planning (n=194)
| method | ACC ↑ | tokens | TFLOPs | KV MB | peak MB | CUDA ms |
|---|---|---|---|---|---|---|
| baseline | 0.2938 | 1536 | 10.09 | 98.5 | 15522 | 2546 |
| **VisPruner@25%** | **0.3144** | 384 | 3.49 | 35.5 | 15294 | 2810 |
| random@25% | 0.2990 | 384 | 3.49 | 35.5 | 15294 | 3283 |

### object_rel_direction_easy (n=217)
| method | ACC ↑ | tokens | TFLOPs | KV MB | peak MB | CUDA ms |
|---|---|---|---|---|---|---|
| baseline | 0.4931 | 1536 | 9.55 | 93.5 | 15505 | 1709 |
| VisPruner@25% | 0.5069 | 384 | 2.98 | 30.5 | 15277 | 1598 |
| **random@25%** | **0.5161** | 384 | 2.98 | 30.5 | 15277 | 1665 |

## Takeaways
- At keep 25% (75% pruned), both methods **match or exceed baseline** on both tasks,
  with **−65%/−69% prefill FLOPs** and **−64%/−67% KV-cache**. VSI frames are highly
  redundant, so aggressive pruning is essentially free for these tasks.
- **VisPruner vs random is mixed/small** (VisPruner +1.5pp on route_planning, random
  +0.9pp on rel_direction_easy) — within noise at n≈200, unlike Ego3D where VisPruner
  clearly beat random.
- Peak memory ~flat (8B weights dominate), CUDA time noisy/decode-dominated — the
  compression win is in tokens/FLOPs/KV, as on Ego3D.

## Validation
Small-sample sweep (all 10 VSI types × 3 methods, 5/type) completed for every type
(`logs/InternVL3-8B-vsismoke-*-vsibench/`), exercising both ACC and MRA paths.

## Setup notes
- Data: full VSI-Bench (5,130 Q, 3 sources), `scripts/prep_vsibench.py` downloaded
  the 3 video zips (~5.3 GB) and sampled 6 uniform frames/video → 0 videos skipped.
- Same caveat as Ego3D: absolute numbers use our CoT prompt (not VSI paper's exact
  prompt), so they aren't directly paper-comparable; the baseline-vs-compression
  comparison is valid (identical setup all sides).
