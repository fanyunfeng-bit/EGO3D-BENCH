"""Computational-efficiency meters for the token-compression comparison.

Four metrics, matching how the VisPruner / FastV line of work reports them:

  * FLOPs   - theoretical LLM prefill FLOPs as a closed-form function of the
              input token count (the term that the visual-token reduction
              actually changes). Same formula for baseline and compressed, so
              the relative comparison is exact regardless of MAC convention.
  * Storage - KV-cache size in bytes for the prefill sequence.
  * Memory  - measured peak GPU memory during generation.
  * Time    - measured CUDA wall time during generation (cuda.Event).

The LLM dimensions are read generically from a HF config so this works for any
decoder backbone (InternVL3-8B uses a Qwen2.5-7B LLM).
"""

import torch


def _llm_dims(cfg):
    return dict(
        L=cfg.num_hidden_layers,
        d=cfg.hidden_size,
        m=cfg.intermediate_size,
        h=cfg.num_attention_heads,
        kv=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        vocab=cfg.vocab_size,
    )


def prefill_flops(seq_len, cfg, lm_head_tokens=1):
    """Theoretical prefill FLOPs of the LLM over `seq_len` input tokens.

    Per-layer term (FastV / VisPruner convention):
        4*n*d^2   (Q,K,V,O projections)
      + 2*n^2*d   (attention: QK^T and AV)
      + 2*n*d*m   (FFN)
    summed over layers, plus the lm_head over `lm_head_tokens` positions
    (greedy decoding only needs the last position during prefill).
    """
    p = _llm_dims(cfg)
    n, d, m, L = seq_len, p["d"], p["m"], p["L"]
    per_layer = 4 * n * d * d + 2 * n * n * d + 2 * n * d * m
    return float(L * per_layer + lm_head_tokens * d * p["vocab"])


def kv_cache_bytes(seq_len, cfg, dtype_bytes=2):
    """KV-cache size in bytes for a prefill of `seq_len` tokens (bf16 -> 2)."""
    p = _llm_dims(cfg)
    head_dim = p["d"] // p["h"]
    return 2 * p["L"] * seq_len * p["kv"] * head_dim * dtype_bytes  # 2 = K and V


class GpuProfile:
    """Context manager measuring CUDA time (ms) and peak memory (bytes).

        with GpuProfile() as prof:
            model.generate(...)
        prof.cuda_time_ms, prof.peak_mem_bytes
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.cuda_time_ms = None
        self.peak_mem_bytes = None

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *exc):
        self._end.record()
        torch.cuda.synchronize(self.device)
        self.cuda_time_ms = self._start.elapsed_time(self._end)
        self.peak_mem_bytes = torch.cuda.max_memory_allocated(self.device)
        return False


def summarize(records):
    """Mean of per-sample efficiency dicts -> summary dict (with MB/TFLOPs)."""
    if not records:
        return {}
    keys = ["visual_tokens", "seq_len", "flops", "kv_bytes", "cuda_time_ms", "peak_mem_bytes"]
    mean = {k: sum(r[k] for r in records) / len(records) for k in keys if k in records[0]}
    out = {"n_samples": len(records), **mean}
    if "flops" in mean:
        out["tflops"] = mean["flops"] / 1e12
    if "kv_bytes" in mean:
        out["kv_mb"] = mean["kv_bytes"] / 1024**2
    if "peak_mem_bytes" in mean:
        out["peak_mem_mb"] = mean["peak_mem_bytes"] / 1024**2
    return out
