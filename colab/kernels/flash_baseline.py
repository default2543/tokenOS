"""FlashAttention-2 decode: the production baseline on Ampere and newer.

On an A100, vLLM 0.6.6 decodes with its FlashAttention backend, not the
PagedAttention kernel it uses on Turing. `flash_attn_with_kvcache` is that
decode path (split-KV, "flash-decoding"). Here it reads a contiguous cache,
like the Triton kernels, so the comparison has no layout caveat. The cache is
transposed into FlashAttention's [batch, context, kv_heads, head_dim] layout
once, outside timing.

Uses the FlashAttention build that ships with vLLM; needs compute capability 8.0+.
"""

from __future__ import annotations

import math

import torch


def available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


def _kvcache_fn():
    try:
        from vllm.vllm_flash_attn import flash_attn_with_kvcache
    except ImportError:
        from vllm_flash_attn import flash_attn_with_kvcache
    return flash_attn_with_kvcache


def flash_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Return a zero-argument callable running FlashAttention decode."""
    flash_attn_with_kvcache = _kvcache_fn()
    batch, kv_heads, context, head_dim = k.shape
    k_cache = k.transpose(1, 2).contiguous()
    v_cache = v.transpose(1, 2).contiguous()
    seqlens = torch.full((batch,), context, dtype=torch.int32, device=k.device)
    query = q.unsqueeze(1)  # one new token: [batch, 1, q_heads, head_dim]
    scale = 1.0 / math.sqrt(head_dim)

    def run():
        out = flash_attn_with_kvcache(query, k_cache, v_cache, cache_seqlens=seqlens,
                                      softmax_scale=scale, causal=True)
        return out.squeeze(1)

    return run
