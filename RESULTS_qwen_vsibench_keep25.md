# Cross-architecture: token pruning on VSI-Bench (keep 25%)

Does "pruning 75% of visual tokens barely hurts (even helps)" generalize beyond
InternVL3? Same setup on **Qwen2.5-VL-7B** (6 uniform 448px frames = 256 tok/frame
= 1536; keep 25% → 384; R1-style CoT + `View N:`; baseline / VisPruner / random).

Qwen needs a bespoke adapter (`compressors/qwen_adapter.py`,
`models/qwen2.5_vl_vsibench.py`): no-CLS windowed ViT (patch one full-attn block
to expose attention), patch→merged importance aggregation via `get_window_index`,
and M-RoPE-aware pruning with a manual greedy decode (no `visual_features=` hook).
Same Qwen2.5-7B LLM backbone as InternVL3-8B, so efficiency is directly comparable.

## route_planning (n=194), keep 25%

### Qwen2.5-VL-7B
| method | ACC ↑ | tokens | TFLOPs | KV MB | peak MB | CUDA ms |
|---|---|---|---|---|---|---|
| baseline | 0.2680 | 1536 | 9.94 | 97.1 | 16215 | 4650 |
| **VisPruner@25%** | **0.2938** | 384 | 3.35 | 34.1 | 15994 | 5180 |
| random@25% | 0.2835 | 384 | 3.35 | 34.1 | 15994 | 4930 |

### InternVL3-8B (for reference)
| method | ACC ↑ | tokens | TFLOPs | KV MB | peak MB | CUDA ms |
|---|---|---|---|---|---|---|
| baseline | 0.2938 | 1536 | 10.09 | 98.5 | 15522 | 2546 |
| **VisPruner@25%** | **0.3144** | 384 | 3.49 | 35.5 | 15294 | 2810 |
| random@25% | 0.2990 | 384 | 3.49 | 35.5 | 15294 | 3283 |

## Findings
- **The phenomenon holds across architectures.** On both Qwen2.5-VL-7B and
  InternVL3-8B, pruning to 25% **matches or beats** the full-token baseline while
  cutting FLOPs ~66% and KV ~65%: Qwen +2.6pp (VisPruner) / +1.5pp (random) over
  baseline; InternVL3 +2.1pp / +0.5pp.
- **VisPruner ≥ random on both** (Qwen +1.0pp, InternVL3 +1.5pp) — informed
  selection is consistently a touch better than random here.
- Efficiency is identical across models (shared Qwen2.5-7B LLM). Qwen's wall-clock
  is higher (manual decode + longer generations), but CUDA-time is decode-dominated
  and noisy; the compression win is in tokens/FLOPs/KV.
- Interpretation: heavy visual-token redundancy in multi-frame VSI inputs is a
  property of the **input/task**, not a quirk of one model architecture.

## Caveat
Single task (route_planning, the smallest & ~chance-level). The trend is consistent
across two architectures but should be confirmed on more VSI tasks for a firm claim.
