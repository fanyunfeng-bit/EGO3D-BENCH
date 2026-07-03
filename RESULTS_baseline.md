# Baseline results — Qwen2.5-VL-7B-Instruct on Ego3D-Bench (full, 8675/8675)

Run completed on a single RTX 3090. Greedy decoding, full image resolution, flash-attention-2,
last-token `lm_head` patch. All 8,675 samples processed; per-category log row counts match the
dataset exactly (no duplicates/gaps from the mid-run resume).

## Multiple-choice (Accuracy ↑)

| Category | Accuracy | Note |
|---|---|---|
| Ego_Centric_Absolute_Distance_MultiChoice | 0.354 | |
| Object_Centric_Absolute_Distance_MultiChoice | 0.336 | |
| Object_Centric_Relative_Distance | 0.590 | |
| Ego_Centric_Relative_Distance | **0.589** | remapped (see below); raw eval reported 0.000 |
| Ego_Centric_Motion_Reasoning | **0.654** | remapped; raw eval reported 0.000 |
| Object_Centric_Motion_Reasoning | **0.583** | remapped; raw eval reported 0.000 |
| Localization | 0.306 | |
| Travel_Time | 0.382 | |

## Exact-number (RMSE ↓, meters)

| Category | RMSE |
|---|---|
| Ego_Centric_Absolute_Distance | 28.39 |
| Object_Centric_Absolute_Distance | 36.99 |

## The "remapped" categories — upstream prompt bug

For `Ego_Centric_Relative_Distance`, `Ego_Centric_Motion_Reasoning`, and
`Object_Centric_Motion_Reasoning`, the upstream baseline script (`models/qwen2.5_vl.py`)
prompted the model to answer **"yes or no"**, but these are multiple-choice questions whose
ground truth is the option **letter** (A/B). So the eval compared e.g. `"yes" == "a"` and scored
**0.000** for all three — an artifact, not real performance. (The Ego3D-VLM script
`models/qwen2.5_vl_ego3dvlm.py` does not have this bug; it asks for the letter.)

The numbers above for those three were recovered post-hoc by mapping each yes/no answer to the
option letter whose text matches it, then comparing to GT — no re-run needed. The baseline
prompt has since been fixed to ask for the letter for every multiple-choice category, so a fresh
run is natively correct.

Reproduce the remap: see the alignment script in the chat history (loads the dataset, zips
per-category in order with the log rows, maps yes/no→letter).
