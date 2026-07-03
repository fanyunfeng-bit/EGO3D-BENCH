# vispruner (keep 25%) vs baseline — InternVL3-8B on Ego3D-Bench

Visual tokens reduced to 25% per view. Each task recorded separately. ACC↑ for multiple-choice, RMSE↓ (meters) for numeric. Efficiency: theoretical prefill FLOPs + KV-cache bytes + measured peak GPU mem + cuda.Event time (see `utils/efficiency.py`).

## Performance

| Category | Metric | Baseline | vispruner@25% | Δ |
|---|---|---|---|---|
| Object_Centric_Absolute_Distance_MultiChoice | ACC | 0.495 | 0.475 | -0.020 |
| Ego_Centric_Absolute_Distance_MultiChoice | ACC | 0.531 | 0.515 | -0.016 |
| Localization | ACC | 0.351 | 0.358 | +0.008 |
| Travel_Time | ACC | 0.448 | 0.439 | -0.009 |
| Ego_Centric_Absolute_Distance | RMSE | 12.784 | 12.848 | +0.064 |
| Object_Centric_Absolute_Distance | RMSE | 28.140 | 23.816 | -4.325 |

## Efficiency (mean per sample)

| Category | tokens B→C | TFLOPs B→C | KV MB B→C | peakMem MB B→C | CUDA ms B→C |
|---|---|---|---|---|---|
| Object_Centric_Absolute_Distance_MultiChoice | 1546→386 | 9.69→3.06 | 94.7→31.3 | 15509→15280 | 4058.9→3626.5 |
| Ego_Centric_Absolute_Distance_MultiChoice | 1578→395 | 9.84→3.08 | 96.1→31.4 | 15514→15280 | 3012.4→2476.6 |
| Localization | 1564→391 | 9.86→3.15 | 96.3→32.1 | 15515→15283 | 3817.9→3634.6 |
| Travel_Time | 1578→395 | 10.04→3.26 | 97.9→33.2 | 15521→15286 | 5535.0→5044.2 |
| Ego_Centric_Absolute_Distance | 1578→395 | 9.68→2.93 | 94.6→29.9 | 15509→15275 | 3011.0→3057.5 |
| Object_Centric_Absolute_Distance | 1546→386 | 9.53→2.92 | 93.3→29.9 | 15504→15275 | 3882.0→4048.5 |
