"""vLLM's decode attention: the production baseline on this GPU.

vLLM 0.6.6's V0 engine decodes on Turing with its PagedAttention CUDA kernels
(v1, or v2 with a split-K reduction for long contexts), chosen by
`PagedAttention.forward_decode` exactly as in serving. The KV cache is copied
into vLLM's paged layout once, outside timing; the bytes each step reads are
the same as for the Triton kernels.

Needs vLLM installed, so it runs in the serving notebook's runtime.
"""

from __future__ import annotations

import math

import torch

BLOCK_SIZE = 16  # vLLM's default KV block size


def to_paged(k: torch.Tensor, v: torch.Tensor):
    """[batch, kv_heads, context, head_dim] caches -> vLLM's paged layout.

    key_cache:   [blocks, kv_heads, head_dim // x, BLOCK_SIZE, x], x = 16 bytes
    value_cache: [blocks, kv_heads, head_dim, BLOCK_SIZE]
    Each sequence owns a contiguous run of blocks in `block_tables`.
    """
    batch, kv_heads, context, head_dim = k.shape
    if context % BLOCK_SIZE:
        raise ValueError(f"context must be a multiple of {BLOCK_SIZE}")
    per_seq = context // BLOCK_SIZE
    x = 16 // k.element_size()
    key_cache = (k.reshape(batch, kv_heads, per_seq, BLOCK_SIZE, head_dim // x, x)
                 .permute(0, 2, 1, 4, 3, 5)
                 .reshape(batch * per_seq, kv_heads, head_dim // x, BLOCK_SIZE, x)
                 .contiguous())
    value_cache = (v.reshape(batch, kv_heads, per_seq, BLOCK_SIZE, head_dim)
                   .permute(0, 2, 1, 4, 3)
                   .reshape(batch * per_seq, kv_heads, head_dim, BLOCK_SIZE)
                   .contiguous())
    block_tables = torch.arange(batch * per_seq, dtype=torch.int32,
                                device=k.device).reshape(batch, per_seq)
    return key_cache, value_cache, block_tables


def paged_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Return a zero-argument callable running vLLM's decode on these tensors."""
    from vllm.attention.ops.paged_attn import PagedAttention

    key_cache, value_cache, block_tables = to_paged(k, v)
    batch, kv_heads, context, head_dim = k.shape
    seq_lens = torch.full((batch,), context, dtype=torch.int32, device=k.device)
    scale = 1.0 / math.sqrt(head_dim)

    def run(kv_scale):
        return PagedAttention.forward_decode(
            q, key_cache, value_cache, block_tables, seq_lens, context, "auto",
            kv_heads, scale, None, kv_scale, kv_scale,
        )

    # The fp8 KV scales were floats in vLLM 0.6.x and tensors later.
    try:
        run(1.0)
        kv_scale = 1.0
    except (TypeError, RuntimeError):
        kv_scale = torch.tensor(1.0, dtype=torch.float32, device=k.device)
    return lambda: run(kv_scale)
