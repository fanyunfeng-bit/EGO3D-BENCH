"""Score the QA-SCMPruner ablation (Qwen2.5-VL VSI 16f) + baselines.

Re-scores from JSONL with the runner's robust MC letter extraction (\\b[a-d]\\b), so it
is consistent with the live scorer and works even if result.json is stale/absent.
Doubles as a progress monitor: shows per-task row counts as the run fills in.

Usage:  python scripts/score_qa_ablation.py          # all keeps
        python scripts/score_qa_ablation.py 10        # just keep10
"""
import json, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import extract_number_from_answer_tag_mult_choice as _tag

MODEL = "Qwen2.5-VL-7B"
TASKS = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
         "route_planning", "object_rel_distance"]
SHORT = {"object_rel_direction_easy": "dir_e", "object_rel_direction_medium": "dir_m",
         "object_rel_direction_hard": "dir_h", "route_planning": "route", "object_rel_distance": "dist"}
FULL = {"object_rel_direction_easy": 217, "object_rel_direction_medium": 378,
        "object_rel_direction_hard": 373, "route_planning": 194, "object_rel_distance": 710}


def letter(pred):
    s = _tag(pred)
    m = re.search(r"\b([a-d])\b", s)
    return m.group(1) if m else s


def acc_of(path):
    if not os.path.exists(path):
        return None, 0
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return None, 0
    c = sum(1 for r in rows if letter(r["Processed_Pred"]) == r["GT"].lower())
    return c / len(rows), len(rows)


def row(label, d):
    accs, ns = {}, {}
    for t in TASKS:
        accs[t], ns[t] = acc_of(f"{d}/{t}.jsonl")
    complete = all(ns[t] >= FULL[t] for t in TASKS)
    valid = [accs[t] for t in TASKS if accs[t] is not None]
    mean = sum(valid) / len(valid) if valid else None
    cells = "  ".join(f"{SHORT[t]}={('%.3f' % accs[t]) if accs[t] is not None else '  -  '}"
                      f"({ns[t]}/{FULL[t]})" for t in TASKS)
    flag = "" if complete else "  <partial>"
    mstr = f"MEAN={'%.3f' % mean}" if mean is not None else "MEAN=  -  "
    print(f"  {label:22} {cells}  | {mstr}{flag}")


def main():
    keeps = [int(sys.argv[1])] if len(sys.argv) > 1 else [25, 10, 5]
    print(f"full-token baseline (keep-independent):")
    row("baseline", f"logs/{MODEL}-baseline-vsibench")
    for keep in keeps:
        print(f"\n===== keep{keep} =====")
        for m in ["random", "vispruner", "scmpruner", "fastv"]:
            row(m, f"logs/{MODEL}-{m}-keep{keep}-vsibench")
        print("  " + "-" * 100)
        for sw in [0, 1]:
            for sig in ["attn", "cosine"]:
                suf = ("-sigcos" if sig == "cosine" else "") + ("-sw1" if sw else "")
                row(f"qa[{sig},sw{sw}]", f"logs/{MODEL}-scmpruner_qa-keep{keep}{suf}-vsibench")


if __name__ == "__main__":
    main()
