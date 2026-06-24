"""Aggregate per-category result jsons into a baseline-vs-compressed comparison.

Reads logs/<model>-baseline/<cat>.result.json and
logs/<model>-<method>-keep<pct>/<cat>.result.json for each category, and writes
a markdown table + a combined json. Results are kept SEPARATE per task.

Usage:
  python -m utils.aggregate_compress --model_name InternVL3-8B --method vispruner --keep_ratio 0.25
"""

import argparse
import json
import os

DEFAULT_CATEGORIES = [
    "Object_Centric_Absolute_Distance_MultiChoice",
    "Ego_Centric_Absolute_Distance_MultiChoice",
    "Localization",
    "Travel_Time",
    "Ego_Centric_Absolute_Distance",
    "Object_Centric_Absolute_Distance",
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="InternVL3-8B")
    ap.add_argument("--method", default="vispruner")
    ap.add_argument("--keep_ratio", type=float, default=0.25)
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    args = ap.parse_args()

    pct = int(round(args.keep_ratio * 100))
    base_dir = f"logs/{args.model_name}-baseline"
    comp_dir = f"logs/{args.model_name}-{args.method}-keep{pct}"

    rows = []
    for cat in args.categories:
        b = load(f"{base_dir}/{cat}.result.json")
        c = load(f"{comp_dir}/{cat}.result.json")
        rows.append({"category": cat, "baseline": b, "compressed": c})

    out_json = f"logs/compress_summary_{args.method}_keep{pct}.json"
    with open(out_json, "w") as f:
        json.dump({"model": args.model_name, "method": args.method,
                   "keep_ratio": args.keep_ratio, "results": rows}, f, indent=2)

    def fmt(v, nd=3):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    def eff(r, k):
        return r["efficiency"][k] if r and r.get("efficiency") else None

    lines = []
    lines.append(f"# {args.method} (keep {pct}%) vs baseline — {args.model_name} on Ego3D-Bench\n")
    lines.append(f"Visual tokens reduced to {pct}% per view. Each task recorded separately. "
                 "ACC↑ for multiple-choice, RMSE↓ (meters) for numeric. Efficiency: theoretical "
                 "prefill FLOPs + KV-cache bytes + measured peak GPU mem + cuda.Event time "
                 "(see `utils/efficiency.py`).\n")
    lines.append("## Performance\n")
    lines.append("| Category | Metric | Baseline | " + f"{args.method}@{pct}% | Δ |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        b, c = r["baseline"], r["compressed"]
        m = (b or c or {}).get("metric", "?")
        bv = b["value"] if b else None
        cv = c["value"] if c else None
        d = "—"
        if bv is not None and cv is not None:
            d = f"{cv-bv:+.3f}"
        lines.append(f"| {r['category']} | {m} | {fmt(bv)} | {fmt(cv)} | {d} |")

    lines.append("\n## Efficiency (mean per sample)\n")
    lines.append("| Category | tokens B→C | TFLOPs B→C | KV MB B→C | peakMem MB B→C | CUDA ms B→C |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        b, c = r["baseline"], r["compressed"]
        def pair(k, nd=1):
            return f"{fmt(eff(b,k),nd)}→{fmt(eff(c,k),nd)}"
        lines.append(f"| {r['category']} | {pair('visual_tokens',0)} | {pair('tflops',2)} | "
                     f"{pair('kv_mb')} | {pair('peak_mem_mb',0)} | {pair('cuda_time_ms')} |")

    md = f"RESULTS_{args.method}_keep{pct}.md"
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {md} and {out_json}")


if __name__ == "__main__":
    main()
