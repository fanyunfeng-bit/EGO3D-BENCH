"""Standalone unit tests for the QA-SCMPruner pure helpers (no model load).
Run: python scripts/test_scmpruner_qa_units.py  -> prints 'ALL OK' on success."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from compressors.scm import (scmpruner_qa_budgets, input_cos_relevance,
                             cosine_relevance, scmpruner_qa_tag_suffix)


def test_budgets():
    # r=7,K=14,L=28,M=4096: N2=0.25*T*M, N1=1.75*T*M
    N1, N2, avg = scmpruner_qa_budgets(0.25, 4096, 7, 14, 28)
    assert (N1, N2) == (1792, 256), (N1, N2)
    assert abs(avg - 0.25 * 4096) < 1.0, avg
    N1, N2, _ = scmpruner_qa_budgets(0.10, 4096, 7, 14, 28)
    assert (N1, N2) == (714, 102), (N1, N2)           # integer path: round(409.6*0.25)=102, round(7*102)=714
    N1, N2, _ = scmpruner_qa_budgets(0.10, 4096, 3, 14, 28)
    assert (N1, N2) == (615, 205), (N1, N2)           # gentler r=3
    N1, N2, _ = scmpruner_qa_budgets(0.9, 100, 7, 14, 28)
    assert N1 <= 100, N1                                # N1 capped at M


def test_input_cos_relevance_nonneg():
    v = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0]])
    r = input_cos_relevance(v, q)
    assert r.shape == (3,)
    assert torch.all(r >= 0), r                        # relu -> no negatives
    assert r[0] > 0 and r[1].item() == 0.0             # aligned>0, anti-aligned clamped to 0


def test_cosine_relevance():
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # tok0 vis, tok1 vis, tok2 query
    score = cosine_relevance(h, torch.tensor([0, 1]), torch.tensor([2]))
    assert score.shape == (2,)
    assert score[0] > score[1]                         # vis0 aligns with query, vis1 does not


def test_tag_suffix():
    assert scmpruner_qa_tag_suffix() == ""             # canonical default
    assert scmpruner_qa_tag_suffix(r=3) == "-r3"
    assert scmpruner_qa_tag_suffix(K=7) == "-k7"
    assert scmpruner_qa_tag_suffix(sig="cosine") == "-sigcos"
    assert scmpruner_qa_tag_suffix(softweight=1) == "-sw1"
    assert scmpruner_qa_tag_suffix(r=3, sig="cosine", softweight=1) == "-r3-sigcos-sw1"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("ALL OK")
