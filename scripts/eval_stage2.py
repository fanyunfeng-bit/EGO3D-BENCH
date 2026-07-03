"""Stage-2 eval: does a cvsp setting (default tag -a20s40-k2) beat plain_random AND
vispruner on MOST of the 6 spatial tasks across keep10/5/3? Writes GOOD/BAD to
logs/cvsp/stage2.decision.json. GOOD = mean(cvsp-visp) > +0.5pp AND cvsp>=visp on
>=11/18 cells AND cvsp>=plain on >=12/18 (judged on the aggregate, per the n=200
plan: per-task +1~2% is below noise, the sign-test over cells is the real evidence)."""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import extract_number_from_answer_tag_mult_choice as M

TAG = sys.argv[1] if len(sys.argv) > 1 else "-a20s40-k2"
D = "logs/cvsp"
TASKS = ["vsi.object_rel_direction_easy", "vsi.object_rel_direction_medium",
         "vsi.object_rel_direction_hard", "vsi.object_rel_distance",
         "ego3d.Object_Centric_Absolute_Distance_MultiChoice",
         "ego3d.Ego_Centric_Absolute_Distance_MultiChoice"]
RATIOS = ["keep10", "keep5", "keep3"]


def acc(p):
    if not os.path.exists(p):
        return None
    r = [json.loads(l) for l in open(p)]
    return sum(1 for x in r if M(x["Processed_Pred"]) == x["GT"].lower()) / len(r) if len(r) >= 200 else None


cells = []
incomplete = 0
print(f"=== Stage-2 eval (tag {TAG}) : cvsp vs vispruner / plain, 6 spatial x keep10/5/3 ===")
print(f"{'task':<44}{'ratio':>7}{'cvsp':>8}{'visp':>8}{'plain':>8}{'Dvp':>8}")
for t in TASKS:
    for r in RATIOS:
        c = acc(f"{D}/{t}.{r}.cvsp{TAG}.jsonl")
        v = acc(f"{D}/{t}.{r}.vispruner.jsonl")
        p = acc(f"{D}/{t}.{r}.plain_random.jsonl")
        if c is None or v is None or p is None:
            incomplete += 1
            continue
        cells.append((t, r, c, v, p))
        print(f"{t:<44}{r:>7}{c:>8.3f}{v:>8.3f}{p:>8.3f}{c - v:>+8.3f}")

if incomplete or not cells:
    dec = {"decision": "INCOMPLETE", "incomplete": incomplete, "tag": TAG}
else:
    dvp = [c - v for _, _, c, v, p in cells]
    wvp = sum(1 for _, _, c, v, p in cells if c >= v)
    wpr = sum(1 for _, _, c, v, p in cells if c >= p)
    mean = sum(dvp) / len(dvp)
    GOOD = mean > 0.005 and wvp >= 11 and wpr >= 12
    dec = {"decision": "GOOD" if GOOD else "BAD", "tag": TAG, "n_cells": len(cells),
           "meanDvp": round(mean, 4), "wins_vp": f"{wvp}/{len(cells)}", "wins_pr": f"{wpr}/{len(cells)}"}
json.dump(dec, open(f"{D}/stage2.decision.json", "w"), indent=2)
print("DECISION:", json.dumps(dec))
