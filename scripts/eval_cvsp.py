"""Viewer for the CVSP main curve: scan logs/cvsp/*.jsonl and print per-config
ACC tables (baseline + every keep ratio x method). Safe to run anytime, even
while the curve is still being generated. Usage (from repo root):

  PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH python scripts/eval_cvsp.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import extract_number_from_answer_tag_mult_choice

ORDER = ["baseline", "plain_random", "strat_random", "vispruner", "cvsp"]


def acc_of(path):
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return None
    a = sum(1 for r in rows if extract_number_from_answer_tag_mult_choice(r["Processed_Pred"]) == r["GT"].lower()) / len(rows)
    return a, sum(r["n_kept"] for r in rows) / len(rows), len(rows)


def main():
    files = glob.glob("logs/cvsp/*.jsonl")
    configs = {}
    for f in files:
        base = os.path.basename(f)[:-6]                       # strip .jsonl
        m = re.match(r"(ego3d|vsi)\.(.+?)\.(baseline|keep(\d+)\.(\w+))$", base)
        if not m:
            continue
        ds, task, tag = m.group(1), m.group(2), m.group(3)
        configs.setdefault((ds, task), []).append((tag, f))

    for (ds, task) in sorted(configs):
        print(f"\n=== {ds}/{task} (ACC, MC chance=0.25) ===")
        entries = dict(configs[(ds, task)])
        if "baseline" in entries:
            r = acc_of(entries["baseline"])
            if r:
                print(f"  baseline                 ACC={r[0]:.4f}  tok={r[1]:.0f}  n={r[2]}")
        pcts = sorted({int(re.match(r"keep(\d+)\.", t).group(1)) for t in entries if t.startswith("keep")},
                      reverse=True)
        for pct in pcts:
            print(f"  -- keep{pct}% --")
            for method in ORDER:
                tag = f"keep{pct}.{method}"
                if tag in entries:
                    r = acc_of(entries[tag])
                    if r:
                        print(f"  {method:<15} keep{pct:<3} ACC={r[0]:.4f}  tok={r[1]:.0f}  n={r[2]}")


if __name__ == "__main__":
    main()
