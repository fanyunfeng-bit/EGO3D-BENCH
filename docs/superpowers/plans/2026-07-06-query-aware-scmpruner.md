# Query-Aware SCMPruner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a query-aware two-stage token pruner `scmpruner_qa` to both VSI runners: SCMPruner over-selects N1 cross-view tokens pre-LLM, then an in-LLM prune at layer K=L/2 cuts to N2 by query relevance (no anchor protection), so it runs on Qwen2.5-VL-7B **and** InternVL3-8B and is directly comparable to the existing `scmpruner`.

**Architecture:** Reuse the existing dual-model in-LLM layer-K prune machinery (`compressors/qstage_llm.py` for InternVL, `compressors/fastv.py::make_fastv_forward_qwen` for Qwen). Stage-1 = existing `scmpruner_keep_indices` at `keep_ratio=N1/M`. All additions are config-gated so `fastv`/`scmpruner`/qstage behavior is unchanged. Score by ACC.

**Tech Stack:** Python 3, PyTorch, HuggingFace transformers 4.51.1 (pinned `ego3d` conda env), InternVL3-8B / Qwen2.5-VL-7B on a single RTX 3090.

## Global Constraints

- Run everything under the `ego3d` env: `PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH` (or `conda activate ego3d`). All commands from repo root.
- **16-frame data (all smokes + experiments):** pass `--items data/vsibench/vsibench_items_16f.json --frames 16`. The 16-frame frames are already prepped under `data/vsibench/frames16/`; the default `--items data/vsibench/vsibench_items.json` is 6-frame — do NOT use it (6-frame is out of scope per the user).
- No pytest / no build step in this repo. "Tests" = (a) standalone assert scripts run with the env python for pure functions, (b) `python -m py_compile` for syntax, (c) `--limit 2` smoke runs for model-dependent integration.
- **Determinism is mandatory** and resume is by JSONL line count — the method must produce identical output across reruns (SCMPruner is RNG-free; the query signals are deterministic).
- Both LLMs have **L=28 layers** → default prune layer **K=14 (=L/2)**; both frames tile to **n_tok=256** per view.
- Budget = layer-average `T`: `N2=round(T·L/(r·K+L−K))`, `N1=min(M,round(r·N2))`, canonical `r=7`, `K=14`.
- Config-gated additions only: do NOT change the defaults of `QStage` (`query_reduce` default `"last"`), `QwenFastV` (`signal` default `"attn"`, `query_reduce` default `"last"`), or FastV wrappers.
- Commit after each task. Repo is on `main`; if the executor is told to branch, branch first — otherwise commit to the current branch.
- Method name string: `scmpruner_qa`. New CLI flags (both runners): `--scm_r` (float, 7), `--scm_K` (int, 14), `--scm_sig` (str, `attn`, choices `attn|cosine`), `--scm_softweight` (int, 0). Reuse existing `--scm_rho_a/--scm_rho_s/--anc_m/--anc_tau/--scm_xview`.

---

### Task 1: Pure helpers in `compressors/scm.py`

**Files:**
- Modify: `compressors/scm.py` (append 4 functions after `scmpruner_tag_suffix`)
- Test: `scripts/test_scmpruner_qa_units.py` (create)

**Interfaces:**
- Produces:
  - `scmpruner_qa_budgets(keep_ratio: float, M: int, r: float, K: int, L: int) -> (N1:int, N2:int, layer_avg:float)`
  - `input_cos_relevance(vis_feats: Tensor(M,C), query_embeds: Tensor(Q,C)) -> Tensor(M,)` (= `relu(cos)`, non-negative)
  - `cosine_relevance(hidden: Tensor(S,D), vis_idx: LongTensor, query_idx: LongTensor) -> Tensor(len(vis_idx),)` (cos of vision hidden vs mean query hidden; may be negative — used only for top-N2 ranking)
  - `scmpruner_qa_tag_suffix(r=7, K=14, sig="attn", softweight=0) -> str`

- [ ] **Step 1: Write the failing test**

Create `scripts/test_scmpruner_qa_units.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/test_scmpruner_qa_units.py`
Expected: FAIL — `ImportError: cannot import name 'scmpruner_qa_budgets' from 'compressors.scm'`

- [ ] **Step 3: Implement the helpers**

Append to `compressors/scm.py` (after `scmpruner_tag_suffix`, before `scmpruner_keep_indices`):

```python
def scmpruner_qa_budgets(keep_ratio, M, r, K, L):
    """Two-stage layer-average budget (Notes/CVSP-Method.md §13). Given target
    layer-average token count T = keep_ratio*M, over-select ratio r, prune layer K,
    total layers L: return (N1 stage-1 over-select, N2 stage-2 final, layer_avg)."""
    T = keep_ratio * M
    N2 = max(1, round(T * L / (r * K + L - K)))
    N1 = min(M, max(N2, round(r * N2)))
    layer_avg = (N1 * K + N2 * (L - K)) / L
    return int(N1), int(N2), layer_avg


def input_cos_relevance(vis_feats, query_embeds):
    """Pre-LLM query relevance for the stage-1 soft-weight: relu(cos(v_i, q_bar)) with
    q_bar = mean query token embedding. relu clamps anti-relevant (cos<0) tokens to 0
    so they sort to the bottom of the saliency bucket. Returns (M,) >= 0."""
    v = F.normalize(vis_feats.float(), dim=-1)
    q = F.normalize(query_embeds.float().mean(0, keepdim=True), dim=-1)
    return torch.relu((v @ q.t()).squeeze(1))


def cosine_relevance(hidden, vis_idx, query_idx):
    """In-LLM stage-2 'cosine' signal: cos(hidden[vis], mean hidden[query]) at layer K.
    Not clamped (only used for top-N2 ranking, where the ordering is what matters)."""
    q = F.normalize(hidden[query_idx].float().mean(0, keepdim=True), dim=-1)
    v = F.normalize(hidden[vis_idx].float(), dim=-1)
    return (v @ q.t()).squeeze(1)


def scmpruner_qa_tag_suffix(r=7, K=14, sig="attn", softweight=0):
    """Log-dir suffix encoding only non-default QA knobs (canonical r7/K14/attn/sw0 -> '')
    so a sweep never collides with resume. e.g. '-r3', '-k7', '-sigcos', '-sw1'."""
    s = ""
    if round(r) != 7:
        s += f"-r{int(round(r))}"
    if int(K) != 14:
        s += f"-k{int(K)}"
    if sig != "attn":
        s += f"-sig{sig[:3]}"
    if int(softweight):
        s += "-sw1"
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/test_scmpruner_qa_units.py`
Expected: `ALL OK` (5 `ok` lines)

- [ ] **Step 5: Commit**

```bash
git add compressors/scm.py scripts/test_scmpruner_qa_units.py
git commit -m "feat(scm): QA-SCMPruner pure helpers (budgets, relevance, tag suffix) + unit tests"
```

---

### Task 2: InternVL stage-2 — query-mean `attn` + shared cosine in `qstage_llm.py`

**Files:**
- Modify: `compressors/qstage_llm.py` (`QStage.__init__` add field; `_select_keep` attn/cosine branches)
- Test: `scripts/test_scmpruner_qa_units.py` (add `test_select_keep_reduce`)

**Interfaces:**
- Consumes: `compressors.scm.cosine_relevance` (Task 1)
- Produces: `QStage.query_reduce` field (default `"last"`); `_select_keep` honoring `query_reduce ∈ {last, mean}` for `attn`, and using `cosine_relevance` for `cosine`.

- [ ] **Step 1: Write the failing test**

Add to `scripts/test_scmpruner_qa_units.py` (import at top: `from compressors.qstage_llm import QStage, _select_keep`):

```python
def test_select_keep_reduce():
    # 4 tokens: idx0,1 vision; idx2,3 query. attn signal from a fake (S,S) weight matrix.
    qs = QStage(K=1, signal="attn")
    qs.vis_pos = torch.tensor([0, 1]); qs.query_pos = torch.tensor([2, 3])
    qs.N2 = 1; qs.per_view = False
    # attn_K_minus_1 shape (1, heads=1, S, S); rows=query positions attend to cols=keys
    aw = torch.zeros(1, 1, 4, 4)
    aw[0, 0, 2] = torch.tensor([0.9, 0.1, 0.0, 0.0])   # query tok2 -> vis0 strong
    aw[0, 0, 3] = torch.tensor([0.3, 0.7, 0.0, 0.0])   # query tok3 (LAST) -> vis1
    hidden = torch.zeros(1, 4, 2)
    qs.query_reduce = "last"
    keep_last = _select_keep(qs, hidden, aw)
    assert keep_last.tolist() == [1], keep_last.tolist()   # last query row picks vis1
    qs.query_reduce = "mean"; qs.kept_vis = None
    keep_mean = _select_keep(qs, hidden, aw)
    assert keep_mean.tolist() == [0], keep_mean.tolist()   # mean [0.6,0.4] picks vis0 -> differs from last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/test_scmpruner_qa_units.py`
Expected: FAIL on `test_select_keep_reduce` — `AttributeError`/wrong pick, because `query_reduce` is not honored yet (current attn always uses the last row).

- [ ] **Step 3: Implement**

In `compressors/qstage_llm.py`, add to `QStage.__init__` (after `self.keep_pv = None`):

```python
        self.query_reduce = "last"        # 'last' (FastV) | 'mean' (ITS-style over query tokens)
```

Replace the `_select_keep` signal block (the `if qs.signal == "attn": ... else: # cosine ...` up to the `score = ...`) with:

```python
    from compressors.scm import cosine_relevance
    vis = qs.vis_pos.to(hidden_states.device)
    if qs.signal == "attn":
        a = attn_K_minus_1.mean(dim=1).squeeze(0)            # (S,S) avg over heads, batch=1
        if getattr(qs, "query_reduce", "last") == "mean" and qs.query_pos is not None:
            qp = qs.query_pos.to(a.device)
            score = a[qp][:, vis].mean(0)                    # mean over query tokens (ITS)
        else:
            last_q = qs.query_pos[-1].item() if qs.query_pos is not None else hidden_states.shape[1] - 1
            score = a[last_q, vis]                           # last query token (FastV default)
    else:  # cosine
        score = cosine_relevance(hidden_states[0], vis, qs.query_pos.to(hidden_states.device))
```

(Delete the old inline `vis = ...`, `q_bar`, and `v @ q_bar.t()` lines that this replaces; keep the rest of `_select_keep` — the `per_view` block and the final `topk` — unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/test_scmpruner_qa_units.py`
Expected: `ALL OK`

- [ ] **Step 5: py_compile + commit**

```bash
python -m py_compile compressors/qstage_llm.py
git add compressors/qstage_llm.py scripts/test_scmpruner_qa_units.py
git commit -m "feat(qstage): query_reduce=mean for attn + shared cosine_relevance (gated, FastV default unchanged)"
```

---

### Task 3: InternVL runner — `scmpruner_qa` method

**Files:**
- Modify: `models/internvl3_vsibench.py` (imports, `SPECIAL_METHODS`, new stage helper, `run_category` branch, `main` setup, CLI, tag)

**Interfaces:**
- Consumes: `scm.scmpruner_qa_budgets`, `scm.input_cos_relevance`, `scm.scmpruner_qa_tag_suffix`, `scm.scmpruner_keep_indices` (Task 1); `QStage`, `make_qstage_forward` (Task 2)
- Produces: `--compress_method scmpruner_qa` on InternVL, writing `logs/<model>-scmpruner_qa-keep<NN>[suffix]-vsibench/<cat>.jsonl`

- [ ] **Step 1: Add imports, SPECIAL_METHODS, CLI, setup**

In `models/internvl3_vsibench.py`:

Change line 34:
```python
SPECIAL_METHODS = {"scmpruner", "fastv", "scmpruner_qa"}
```

Add import near the other compressor imports (after line 31):
```python
from compressors.qstage_llm import QStage, make_qstage_forward
```

Add CLI flags in `main()` next to the other `--scm_*` args:
```python
    ap.add_argument("--scm_r", type=float, default=7.0, help="QA over-select ratio N1/N2")
    ap.add_argument("--scm_K", type=int, default=14, help="QA stage-2 prune layer (L/2=14)")
    ap.add_argument("--scm_sig", default="attn", choices=["attn", "cosine"], help="QA stage-2 signal")
    ap.add_argument("--scm_softweight", type=int, default=0, help="QA stage-1 saliency*relu(cos) soft-weight")
```

In `main()`, update `need_capture` and `tag`, and install the qstage patch for `scmpruner_qa`:
```python
    need_capture = (compressor is not None and getattr(compressor, "needs_importance", True)) \
        or method in ("scmpruner", "scmpruner_qa")           # both need ViT saliency
    capture = AttentionCapture(model) if need_capture else None
    tag = "baseline" if method == "none" else f"{method}-keep{int(round(args.keep_ratio*100))}"
    if method == "scmpruner":
        tag += scm.scmpruner_tag_suffix(args.scm_rho_a, args.scm_rho_s,
                                        args.anc_tau, args.anc_m, bool(args.scm_xview))
    qs = None
    if method == "scmpruner_qa":
        tag += scm.scmpruner_qa_tag_suffix(args.scm_r, args.scm_K, args.scm_sig, args.scm_softweight)
        qs = QStage(K=args.scm_K, signal=args.scm_sig)
        qs.query_reduce = "mean"; qs.per_view = False
        model.language_model.model._qs = qs
        make_qstage_forward(model.language_model.model)
```

Pass `qs` into `run_category` (add a keyword arg). Change the call at the bottom of `main()`:
```python
        run_category(model, tokenizer, items, category, args, compressor, capture, tag, out_dir, llm_cfg, fastv, qs)
```
and the signature:
```python
def run_category(model, tokenizer, items, category, args, compressor, capture, tag, out_dir, llm_cfg, fastv=None, qs=None):
```

- [ ] **Step 2: Add the stage-1+stage-2 helper**

Add this function above `run_category` in `models/internvl3_vsibench.py`:

```python
@torch.no_grad()
def scmpruner_qa_stage1(model, tokenizer, pixel_values, capture, question, args):
    """Stage-1: SCMPruner over-select N1 (optional soft-weight), return kept features +
    per-view counts + (N1, N2). N1/N2 come from the layer-average budget."""
    vit = model.extract_feature(pixel_values)                 # (n_views, n_tok, C)
    n_views, n_tok, C = vit.shape
    M = n_views * n_tok
    L = model.language_model.config.num_hidden_layers
    N1, N2, _ = scm.scmpruner_qa_budgets(args.keep_ratio, M, args.scm_r, args.scm_K, L)
    cls_attn = capture.cls_attn
    hw = int(cls_attn.shape[1] ** 0.5)
    imp = model.pixel_shuffle(cls_attn.reshape(n_views, hw, hw, 1).to(vit.dtype),
                              scale_factor=model.downsample_ratio)
    imp = imp.mean(dim=-1).reshape(n_views, -1).reshape(-1).float()   # (M,)
    if args.scm_softweight:
        q_ids = tokenizer(question, return_tensors="pt").input_ids.to(vit.device)
        q_emb = model.language_model.get_input_embeddings()(q_ids)[0]
        imp = imp * scm.input_cos_relevance(vit.reshape(-1, C), q_emb)
    keep = scm.scmpruner_keep_indices(vit.reshape(-1, C), imp, n_views, n_tok, N1 / M,
                                      rho_a=args.scm_rho_a, rho_s=args.scm_rho_s,
                                      anc_tau=args.anc_tau, anc_m=args.anc_m, xview=bool(args.scm_xview))
    keep_t = torch.tensor(sorted(keep), device=vit.device)
    counts, feats = [], []
    for v in range(n_views):
        local = (keep_t[keep_t // n_tok == v] % n_tok).sort().values
        counts.append(int(local.numel())); feats.append(vit[v][local])
    return torch.cat(feats, dim=0), counts, N1, N2
```

- [ ] **Step 3: Wire the branch in `run_category`**

In `run_category`, inside the `with efficiency.GpuProfile() as enc:` block, add a `scmpruner_qa` branch (before the existing `if args.compress_method == "scmpruner":`):

```python
            if args.compress_method == "scmpruner_qa":
                visual_features, img_tokens, N1, N2 = scmpruner_qa_stage1(
                    model, tokenizer, pixel_values, capture, question, args)
            elif args.compress_method == "scmpruner":
                ...  # unchanged
```

After `model_inputs`/`input_ids` are built (after the `build_prompt(...)` call), configure stage-2 for `scmpruner_qa` right before the generate profile block:

```python
        if args.compress_method == "scmpruner_qa":
            ids = input_ids.reshape(-1)
            vis_pos = torch.where(ids == model.img_context_token_id)[0]
            last_vis = int(vis_pos[-1].item())
            qs.vis_pos = vis_pos
            qs.query_pos = torch.arange(last_vis + 1, ids.shape[0], device=ids.device)
            qs.N2 = N2; qs.kept_vis = None; qs.active = True
```

and after the generate block:

```python
        if args.compress_method == "scmpruner_qa":
            qs.active = False
```

- [ ] **Step 4: Verify — syntax + smoke (needs GPU)**

Run: `python -m py_compile models/internvl3_vsibench.py`
Expected: no output (OK).

Run smoke (loads InternVL3-8B; ~1-2 min):
```bash
python models/internvl3_vsibench.py --compress_method scmpruner_qa \
  --items data/vsibench/vsibench_items_16f.json --frames 16 \
  --keep_ratio 0.10 --category object_rel_distance --limit 2
```
Expected: prints `[result:scmpruner_qa-keep10:vsibench:object_rel_distance] ACC=... n=2`; writes `logs/InternVL3-8B-scmpruner_qa-keep10-vsibench/object_rel_distance.jsonl` with 2 rows. No assertion error (the `assert ... == feats.shape[0]` in prompt build must hold).

Sanity smoke on a non-default knob (must write a DIFFERENT dir):
```bash
python models/internvl3_vsibench.py --compress_method scmpruner_qa \
  --items data/vsibench/vsibench_items_16f.json --frames 16 \
  --keep_ratio 0.10 --scm_sig cosine --category object_rel_distance --limit 2
```
Expected: dir `logs/InternVL3-8B-scmpruner_qa-keep10-sigcos-vsibench/`.

- [ ] **Step 5: Commit**

```bash
git add models/internvl3_vsibench.py
git commit -m "feat(internvl-vsi): scmpruner_qa two-stage (SCMPruner stage-1 + in-LLM query prune)"
```

---

### Task 4: Qwen stage-2 — `signal`/`query_reduce`/cosine in `fastv.py`

**Files:**
- Modify: `compressors/fastv.py` (`QwenFastV` fields; `QwenLLMAttentionCapture` mean reduce; `make_fastv_forward_qwen` cosine branch)

**Interfaces:**
- Consumes: `compressors.scm.cosine_relevance` (Task 1)
- Produces: `QwenFastV.signal` (default `"attn"`), `QwenFastV.query_reduce` (default `"last"`), `QwenFastV.query_pos`; `make_fastv_forward_qwen` honoring `signal ∈ {attn, cosine}` and `query_reduce ∈ {last, mean}` (defaults preserve FastV).

- [ ] **Step 1: Extend `QwenFastV` and the capture**

In `compressors/fastv.py`, add to `QwenFastV.__init__` (after `self.keep_pv = None`):
```python
        self.signal = "attn"          # 'attn' | 'cosine'
        self.query_reduce = "last"    # 'last' (FastV) | 'mean' (ITS over query tokens)
        self.query_pos = None         # LongTensor of query token indices (for 'mean' / 'cosine')
```

In `QwenLLMAttentionCapture.__init__`, add (after `self.last_row = None`):
```python
        self.query_pos = None         # set by the controller; enables mean-over-query reduce
        self.query_reduce = "last"
```

In `QwenLLMAttentionCapture._forward`, replace the line `self.last_row = aw[0, :, -1, :].mean(0)` with:
```python
        if self.query_reduce == "mean" and self.query_pos is not None:
            qp = self.query_pos.to(aw.device)
            self.last_row = aw[0][:, qp, :].mean(dim=(0, 1))   # mean over heads and query rows
        else:
            self.last_row = aw[0, :, -1, :].mean(0)            # last query row (FastV)
```

- [ ] **Step 2: Add the cosine branch + wire capture reduce in `make_fastv_forward_qwen`**

In `make_fastv_forward_qwen`, right after `ctrl.capture = QwenLLMAttentionCapture(...)`:
```python
    ctrl.capture.query_reduce = ctrl.query_reduce
    ctrl.capture.query_pos = ctrl.query_pos
```
(These are set once here; the runner sets `ctrl.query_reduce`/`ctrl.query_pos` before calling, and re-syncs per sample — see Task 5.)

Inside `forward`, at the prune point, compute `ctrl.kept` for the `cosine` signal from `hidden_states` **before** pruning. Change the `if do_prune and idx == ctrl.K:` block to first fill `ctrl.kept` when cosine:
```python
            if do_prune and idx == ctrl.K:
                if ctrl.kept is None and ctrl.signal == "cosine":
                    from compressors.scm import cosine_relevance
                    vis = ctrl.vis_pos.to(hidden_states.device)
                    score = cosine_relevance(hidden_states[0], vis, ctrl.query_pos.to(hidden_states.device))
                    if ctrl.per_view and ctrl.n_views:
                        nv = int(ctrl.n_views); ntok = vis.numel() // nv; kp = min(int(ctrl.keep_pv), ntok)
                        loc = score.view(nv, ntok).topk(kp, dim=1).indices
                        off = (torch.arange(nv, device=vis.device) * ntok).unsqueeze(1)
                        ctrl.kept = vis[(loc + off).reshape(-1)].sort().values
                    else:
                        n2 = min(int(ctrl.N2), vis.numel())
                        ctrl.kept = vis[torch.topk(score, n2).indices].sort().values
                vis_set = set(ctrl.vis_pos.tolist()); kept = set(ctrl.kept.tolist())
                ...  # rest of the existing prune block unchanged
```
Also sync the capture reduce/pos each prefill (the `cap` capture block only runs for `attn`; keep it, it already sets `ctrl.kept` from `last_row`). At the top of `forward`, after computing `do_prune`, add:
```python
        if do_prune and ctrl.signal == "attn":
            ctrl.capture.query_reduce = ctrl.query_reduce
            ctrl.capture.query_pos = ctrl.query_pos
```

(The existing `cap` block that computes `ctrl.kept` from `ctrl.capture.last_row` stays as-is; with `per_view=False` it already does the global `topk(score, n2)`.)

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile compressors/fastv.py`
Expected: no output (OK).

- [ ] **Step 4: Non-regression check (FastV unchanged, needs GPU — run in Task 5's smoke)**

No standalone unit test (the patch requires a live Qwen forward). Correctness is verified by the Task 5 smoke; FastV non-regression is guaranteed by defaults (`signal="attn"`, `query_reduce="last"`, `per_view=True` set by `FastVInternVL`/`QwenFastV` callers unchanged).

- [ ] **Step 5: Commit**

```bash
git add compressors/fastv.py
git commit -m "feat(fastv-qwen): add cosine signal + query_reduce=mean for stage-2 (gated, FastV default unchanged)"
```

---

### Task 5: Qwen runner — `scmpruner_qa` method

**Files:**
- Modify: `models/qwen2.5_vl_vsibench.py` (imports, `SPECIAL_METHODS`, CLI, setup, `run_category` branch, tag)

**Interfaces:**
- Consumes: `scm.scmpruner_qa_*` (Task 1); `QwenFastV`, `make_fastv_forward_qwen` (Task 4)
- Produces: `--compress_method scmpruner_qa` on Qwen, writing `logs/<model>-scmpruner_qa-keep<NN>[suffix]-vsibench/<cat>.jsonl`

- [ ] **Step 1: SPECIAL_METHODS, CLI, setup**

In `models/qwen2.5_vl_vsibench.py`:

Line 39:
```python
SPECIAL_METHODS = {"scmpruner", "fastv", "scmpruner_qa"}
```

Add CLI flags next to the other `--scm_*` args:
```python
    ap.add_argument("--scm_r", type=float, default=7.0)
    ap.add_argument("--scm_K", type=int, default=14)
    ap.add_argument("--scm_sig", default="attn", choices=["attn", "cosine"])
    ap.add_argument("--scm_softweight", type=int, default=0)
```

In `main()`, extend `need_capture`/`tag` and install the Qwen stage-2 patch for `scmpruner_qa`:
```python
    need_capture = (compressor is not None and getattr(compressor, "needs_importance", True)) \
        or method in ("scmpruner", "scmpruner_qa")
    capture = QwenAttentionCapture(model) if need_capture else None
    tag = "baseline" if method == "none" else f"{method}-keep{int(round(args.keep_ratio*100))}"
    if method == "scmpruner":
        tag += scm.scmpruner_tag_suffix(args.scm_rho_a, args.scm_rho_s,
                                        args.anc_tau, args.anc_m, bool(args.scm_xview))
    qa_ctrl = None
    if method == "scmpruner_qa":
        from compressors.fastv import QwenFastV, make_fastv_forward_qwen
        tag += scm.scmpruner_qa_tag_suffix(args.scm_r, args.scm_K, args.scm_sig, args.scm_softweight)
        qa_ctrl = QwenFastV(K=args.scm_K)
        qa_ctrl.signal = args.scm_sig; qa_ctrl.query_reduce = "mean"; qa_ctrl.per_view = False
        make_fastv_forward_qwen(model.model, qa_ctrl)
```

Pass `qa_ctrl` to `run_category` (reuse the `fastv_ctrl` slot is unsafe since fastv installs its own; add a new kw). Update the call:
```python
        run_category(model, processor, items, category, args, compressor, capture, tag, out_dir,
                     llm_cfg, eos_ids, fastv_ctrl, qa_ctrl)
```
and the signature:
```python
def run_category(model, processor, items, category, args, compressor, capture, tag, out_dir, llm_cfg, eos_ids,
                 fastv_ctrl=None, qa_ctrl=None):
```

- [ ] **Step 2: Wire stage-1 + stage-2 in `run_category`**

In the `with efficiency.GpuProfile() as enc:` block, add a `scmpruner_qa` branch. It over-selects N1 with SCMPruner (optional soft-weight), builds `red_embeds` with the N1 visual tokens, and sets `qa_ctrl` with **red-sequence** vis/query positions:

```python
            method = args.compress_method
            if method == "scmpruner_qa":
                importance = merged_importance(model, inputs["image_grid_thw"], capture.received)
                n_views = inputs["image_grid_thw"].shape[0]
                n_tok = image_embeds.shape[0] // n_views
                M = image_embeds.shape[0]
                L = model.model.config.num_hidden_layers
                N1, N2, _ = scm.scmpruner_qa_budgets(args.keep_ratio, M, args.scm_r, args.scm_K, L)
                if args.scm_softweight:
                    q_ids = processor.tokenizer(item["question"], return_tensors="pt").input_ids.to(image_embeds.device)
                    q_emb = model.model.embed_tokens(q_ids)[0]
                    importance = importance * scm.input_cos_relevance(image_embeds, q_emb)
                keep = scm.scmpruner_keep_indices(image_embeds, importance, n_views, n_tok, N1 / M,
                                                  rho_a=args.scm_rho_a, rho_s=args.scm_rho_s,
                                                  anc_tau=args.anc_tau, anc_m=args.anc_m,
                                                  xview=bool(args.scm_xview))
                keep_mask_vis = torch.zeros(M, dtype=torch.bool, device=image_embeds.device)
                keep_mask_vis[torch.tensor(keep, device=image_embeds.device)] = True
            elif method == "scmpruner":
                ...  # unchanged
```

The existing code already builds `keep_full`, `red_embeds`, `red_pos` from `keep_mask_vis`. **After** those are built, configure the stage-2 controller with red-sequence positions (add before the `greedy_decode` profile block):

```python
        if qa_ctrl is not None:
            red_ids = inputs["input_ids"][0][keep_full]
            vis_pos_red = torch.where(red_ids == image_token_id)[0]
            last_vis = int(vis_pos_red[-1].item())
            qa_ctrl.vis_pos = vis_pos_red
            qa_ctrl.query_pos = torch.arange(last_vis + 1, red_ids.shape[0], device=red_ids.device)
            qa_ctrl.n_views = inputs["image_grid_thw"].shape[0]
            qa_ctrl.N2 = N2; qa_ctrl.kept = None; qa_ctrl.active = True
```

and after `gen_ids = greedy_decode(...)`:
```python
        if qa_ctrl is not None:
            qa_ctrl.active = False
```

(`image_token_id = model.config.image_token_id` is already defined at the top of `run_category`.)

- [ ] **Step 3: Verify — syntax + smoke (needs GPU)**

Run: `python -m py_compile models/qwen2.5_vl_vsibench.py`
Expected: OK.

Smoke (loads Qwen2.5-VL-7B):
```bash
python models/qwen2.5_vl_vsibench.py --compress_method scmpruner_qa \
  --items data/vsibench/vsibench_items_16f.json --frames 16 \
  --keep_ratio 0.10 --category object_rel_distance --limit 2
```
Expected: `[result:scmpruner_qa-keep10:vsibench:object_rel_distance] ACC=... n=2`; dir `logs/Qwen2.5-VL-7B-scmpruner_qa-keep10-vsibench/`.

Cosine + soft-weight smoke (different dir):
```bash
python models/qwen2.5_vl_vsibench.py --compress_method scmpruner_qa \
  --items data/vsibench/vsibench_items_16f.json --frames 16 \
  --keep_ratio 0.10 --scm_sig cosine --scm_softweight 1 \
  --category object_rel_distance --limit 2
```
Expected: dir `logs/Qwen2.5-VL-7B-scmpruner_qa-keep10-sigcos-sw1-vsibench/`.

Determinism: run the first smoke twice into a fresh dir; the 2 JSONL rows must be byte-identical across runs (delete the dir between runs, or diff after re-running with a 3rd sample count).

- [ ] **Step 4: Non-regression smoke (FastV unchanged)**

```bash
python models/qwen2.5_vl_vsibench.py --compress_method fastv \
  --items data/vsibench/vsibench_items_16f.json --frames 16 \
  --keep_ratio 0.10 --category object_rel_distance --limit 2
```
Expected: runs as before (FastV per-view, attn) — confirms Task 4 defaults didn't regress FastV.

- [ ] **Step 5: Commit**

```bash
git add models/qwen2.5_vl_vsibench.py
git commit -m "feat(qwen-vsi): scmpruner_qa two-stage (SCMPruner stage-1 + in-LLM query prune)"
```

---

### Task 6: Run driver + docs

**Files:**
- Create: `scripts/run_scmpruner_qa.sh`
- Modify: `docs/算法运行指南.md` (add a QA-SCMPruner section), `CLAUDE.md` (one-line pointer in the compression section)

**Interfaces:**
- Consumes: the `scmpruner_qa` method on both runners (Tasks 3, 5)

- [ ] **Step 1: Write the driver**

Create `scripts/run_scmpruner_qa.sh`:

```bash
#!/usr/bin/env bash
# QA-SCMPruner vs baselines on the 5 cross-view relational VSI tasks (both models).
# Usage: bash scripts/run_scmpruner_qa.sh [MODEL] [KEEP]   MODEL=qwen|internvl  KEEP=0.10
set -euo pipefail
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${1:-qwen}"; KEEP="${2:-0.10}"
ITEMS=data/vsibench/vsibench_items_16f.json          # 16-frame manifest (frames16/ already prepped)
TASKS="object_rel_direction_easy,object_rel_direction_medium,object_rel_direction_hard,route_planning,object_rel_distance"
if [ "$MODEL" = "qwen" ]; then RUNNER=models/qwen2.5_vl_vsibench.py; else RUNNER=models/internvl3_vsibench.py; fi

# QA-SCMPruner ablation grid: signal x softweight x over-select r (K=14 fixed first)
for SIG in attn cosine; do for SW in 0 1; do for R in 7 3; do
  python "$RUNNER" --compress_method scmpruner_qa --items "$ITEMS" --frames 16 --keep_ratio "$KEEP" \
    --scm_sig "$SIG" --scm_softweight "$SW" --scm_r "$R" --category "$TASKS"
done; done; done

# baselines for the same budget (idempotent / resumable)
for M in none plain_random vispruner scmpruner fastv; do
  python "$RUNNER" --compress_method "$M" --items "$ITEMS" --frames 16 --keep_ratio "$KEEP" --category "$TASKS" || true
done
echo "done: logs/*-scmpruner_qa-keep* + baselines"
```

- [ ] **Step 2: Verify the script parses**

Run: `bash -n scripts/run_scmpruner_qa.sh`
Expected: no output (valid bash).

- [ ] **Step 3: Doc updates**

In `docs/算法运行指南.md`, add a `## QA-SCMPruner (query-aware two-stage)` section documenting: the method (`--compress_method scmpruner_qa`), the knobs (`--scm_r/--scm_K/--scm_sig/--scm_softweight` + reused SCMPruner knobs), the budget split table (from the spec §3.1), output dirs (`logs/<model>-scmpruner_qa-keep<NN>[suffix]-vsibench/`), and the `scripts/run_scmpruner_qa.sh` driver.

In `CLAUDE.md`, in the compression section's SCMPruner bullet, append one sentence: "A query-aware two-stage variant `scmpruner_qa` (SCMPruner stage-1 → in-LLM layer-K query prune, no anchor protection; `--scm_r/--scm_K/--scm_sig/--scm_softweight`) runs on both VSI runners — spec `docs/superpowers/specs/2026-07-06-query-aware-scmpruner-design.md`."

- [ ] **Step 4: Commit**

```bash
git add scripts/run_scmpruner_qa.sh docs/算法运行指南.md CLAUDE.md
git commit -m "docs+driver: QA-SCMPruner run script + run-guide/CLAUDE.md pointers"
```

---

## Experiment plan (after implementation)

Run `bash scripts/run_scmpruner_qa.sh qwen 0.10` (and `0.25`, `0.05`; and `internvl`), then score with the existing viewers. **Success = QA-SCMPruner beats `plain_random` on ≥ majority of (relational task × budget) cells AND ≥ `scmpruner`.** Report which signal wins, whether soft-weight adds, and the r/K trend. Score Ego3D by ACC if extended later (out of scope here).

## Notes / honest caveats (from spec §6)

- Headline (sw0, no protection) ≈ existing qstage with SCMPruner stage-1; §O prior = two-stage did not beat random overall. This is an experimental increment.
- The budget is equal *layer-average*; the two-stage front-loads N1 tokens early — read results as "equal layer-average," not "fewer tokens."
- `attn` signal has known position bias; `query_reduce=mean` over all post-image tokens includes the format instruction. Restricting to the question span is a v2 refinement.
