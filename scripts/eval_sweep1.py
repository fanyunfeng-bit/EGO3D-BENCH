"""Stage-1 gate: on keep5, pick the cvsp budget variant that best beats BOTH
vispruner and plain_random over the 3 discrimination tasks. Emits the winning
(rho_a, rho_s, tag) to logs/cvsp/sweep1.decision.json so run_cvsp_auto.sh can
decide whether to launch the full keep10/5/3 confirmation.

GO gate: best variant has mean(cvsp-vispruner) >= +0.005 over the 3 tasks AND
beats vispruner on >=2/3 AND beats plain_random on >=2/3."""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.eval import extract_number_from_answer_tag_mult_choice as M

D = "logs/cvsp"
RATIO = "keep5"
TASKS = ["vsi.object_rel_direction_easy", "vsi.object_rel_direction_medium",
         "ego3d.Object_Centric_Absolute_Distance_MultiChoice"]
VARIANTS = [("-a40s30-k2", 0.4, 0.3), ("-a30s20-k2", 0.3, 0.2),
            ("-a30s30-k2", 0.3, 0.3), ("-a20s50-k2", 0.2, 0.5)]


def acc(p):
    if not os.path.exists(p):
        return None
    r = [json.loads(l) for l in open(p)]
    return sum(1 for x in r if M(x["Processed_Pred"]) == x["GT"].lower()) / len(r) if r else None


vp = {t: acc(f"{D}/{t}.{RATIO}.vispruner.jsonl") for t in TASKS}
pr = {t: acc(f"{D}/{t}.{RATIO}.plain_random.jsonl") for t in TASKS}
best = None
print(f"=== Stage-1 gate (tuned on {RATIO}) ===")
print(f"  vispruner ACC : {[round(vp[t], 3) for t in TASKS]}")
print(f"  plain_random  : {[round(pr[t], 3) for t in TASKS]}")
print(f"{'variant':<14} {'cvsp ACC':<22} meanD_vp  winVP winPR")
for tag, ra, rs in VARIANTS:
    ac = {t: acc(f"{D}/{t}.{RATIO}.cvsp{tag}.jsonl") for t in TASKS}
    if any(v is None for v in ac.values()):
        print(f"{tag:<14} incomplete")
        continue
    dvp = [ac[t] - vp[t] for t in TASKS]
    mean = sum(dvp) / 3
    wvp = sum(ac[t] >= vp[t] for t in TASKS)
    wpr = sum(ac[t] >= pr[t] for t in TASKS)
    print(f"{tag:<14} {[round(ac[t], 3) for t in TASKS]} {mean:+.4f}  {wvp}/3  {wpr}/3")
    if best is None or mean > best[0]:
        best = (mean, wvp, wpr, tag, ra, rs)

GO = best is not None and best[0] >= 0.005 and best[1] >= 2 and best[2] >= 2
dec = {"decision": "GO" if GO else "STOP", "ratio_tuned": RATIO}
if best:
    dec.update(meanD=round(best[0], 4), wins_vp=best[1], wins_pr=best[2],
               tag=best[3], rho_a=best[4], rho_s=best[5], kappa=2)
json.dump(dec, open(f"{D}/sweep1.decision.json", "w"), indent=2)
print("DECISION:", json.dumps(dec))
