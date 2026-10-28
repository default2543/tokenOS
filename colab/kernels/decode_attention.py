"""Single-query (decode) attention over a KV cache, in Triton.

Generating one token attends one query vector per head to every cached key
and value. Each cached byte is read once and used for two multiply-adds, so
the step is memory-bound: its speed is set by how fast the KV cache streams
from GPU memory, and its cost grows linearly with context length. That is the
link to TokenOS: a shorter prompt is a shorter KV cache, which makes every
*output* token cheaper, not only the prompt.

Shapes follow Qwen2.5-7B (grouped-query attention):
    q:   [batch, q_heads, head_dim]
    k,v: [batch, kv_heads, context, head_dim], q_heads % kv_heads == 0

Three kernels:
  * `decode_attention`         one program per (batch, query head) walks the
                               whole context with an online softmax.
  * `decode_attention_split`   flash-decoding: the context is split across
                               programs for parallelism, then partial results
                               are merged. Matters when batch * heads is small
                               compared to the GPU's SM count (T4 has 40).
  * `decode_attention_grouped` one program per (batch, KV head) serves all of
                               that head's query heads. The per-query-head
                               kernels above read each KV block once per query
                               head, 7x for Qwen2.5-7B; this one reads it once
                               and scores the whole group with `tl.dot`.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, m_ptr, l_ptr,
    context, chunk, scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_os,
    q_heads, group,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    batch = pid // q_heads
    head = pid % q_heads
    kv_head = head // group
    dims = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + batch * stride_qb + head * stride_qh + dims).to(tl.float32)
    k_base = k_ptr + batch * stride_kb + kv_head * stride_kh
    v_base = v_ptr + batch * stride_vb + kv_head * stride_vh

    start = split * chunk
    end = tl.minimum(start + chunk, context)
    m_i = tl.full([1], float("-inf"), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc = tl.zeros([HEAD_DIM], tl.float32)
    for block in range(start, end, BLOCK_N):
        rows = block + tl.arange(0, BLOCK_N)
        valid = rows < end
        k = tl.load(k_base + rows[:, None] * stride_kn + dims[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(v_base + rows[:, None] * stride_vn + dims[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # Unnormalized partial result; the caller normalizes (one split) or merges.
    out = out_ptr + batch * stride_ob + head * stride_oh + split * stride_os
    tl.store(out + dims, acc)
    stat = (batch * q_heads + head) * tl.num_programs(1) + split
    tl.store(m_ptr + stat + tl.arange(0, 1), m_i)
    tl.store(l_ptr + stat + tl.arange(0, 1), l_i)


@triton.jit
def _grouped_decode_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, m_ptr, l_ptr,
    context, chunk, scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_os,
    q_heads, kv_heads, group,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    batch = pid // kv_heads
    kv_head = pid % kv_heads
    dims = tl.arange(0, HEAD_DIM)
    # The group is padded to BLOCK_G (tl.dot needs at least 16 rows); padded
    # query rows are zero and never stored.
    members = tl.arange(0, BLOCK_G)
    in_group = members < group
    heads = kv_head * group + members

    q = tl.load(q_ptr + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :],
                mask=in_group[:, None], other=0.0)
    k_base = k_ptr + batch * stride_kb + kv_head * stride_kh
    v_base = v_ptr + batch * stride_vb + kv_head * stride_vh

    start = split * chunk
    end = tl.minimum(start + chunk, context)
    m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_G], tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], tl.float32)
    for block in range(start, end, BLOCK_N):
        rows = block + tl.arange(0, BLOCK_N)
        valid = rows < end
        k = tl.load(k_base + rows[:, None] * stride_kn + dims[None, :],
                    mask=valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_G, BLOCK_N], fp32
        scores = tl.where(valid[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(v_base + rows[:, None] * stride_vn + dims[None, :],
                    mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = out_ptr + batch * stride_ob + split * stride_os
    tl.store(out + heads[:, None] * stride_oh + dims[None, :], acc,
             mask=in_group[:, None])
    stat = (batch * q_heads + heads) * tl.num_programs(1) + split
    tl.store(m_ptr + stat, m_i, mask=in_group)
    tl.store(l_ptr + stat, l_i, mask=in_group)


def _on_device(*tensors) -> bool:
    # TRITON_INTERPRET=1 runs kernels on CPU tensors, for testing without a GPU.
    return all(t.is_cuda for t in tensors) or os.environ.get("TRITON_INTERPRET") == "1"


def _launch(q, k, v, splits: int, block_n: int, grouped: bool = False):
    if not _on_device(q, k, v):
        raise ValueError("expected CUDA tensors")
    batch, q_heads, head_dim = q.shape
    _, kv_heads, context, _ = k.shape
    if q_heads % kv_heads or head_dim & (head_dim - 1):
        raise ValueError("q_heads must be a multiple of kv_heads; head_dim a power of 2")
    chunk = triton.cdiv(context, splits)
    partial = torch.empty(batch, q_heads, splits, head_dim, device=q.device,
                          dtype=torch.float32)
    m = torch.empty(batch, q_heads, splits, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    args = (
        q, k, v, partial, m, l,
        context, chunk, 1.0 / math.sqrt(head_dim),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        partial.stride(0), partial.stride(1), partial.stride(2),
    )
    group = q_heads // kv_heads
    if grouped:
        _grouped_decode_kernel[(batch * kv_heads, splits)](
            *args, q_heads, kv_heads, group,
            HEAD_DIM=head_dim, BLOCK_N=block_n,
            BLOCK_G=max(16, triton.next_power_of_2(group)), num_warps=4,
        )
    else:
        _decode_kernel[(batch * q_heads, splits)](
            *args, q_heads, group,
            HEAD_DIM=head_dim, BLOCK_N=block_n, num_warps=4,
        )
    return partial, m, l


def decode_attention(q, k, v, *, block_n: int = 64) -> torch.Tensor:
    partial, _, l = _launch(q, k, v, splits=1, block_n=block_n)
    return (partial[:, :, 0] / l[:, :, :1]).to(q.dtype)


def decode_attention_split(q, k, v, *, splits: int = 8, block_n: int = 64) -> torch.Tensor:
    return _merge(*_launch(q, k, v, splits=splits, block_n=block_n), q.dtype)


def decode_attention_grouped(q, k, v, *, splits: int = 8, block_n: int = 64) -> torch.Tensor:
    return _merge(*_launch(q, k, v, splits=splits, block_n=block_n, grouped=True), q.dtype)


def _merge(partial, m, l, dtype) -> torch.Tensor:
    """Merge partial softmaxes: rescale each split to the global max."""
    m_max = m.amax(dim=2, keepdim=True)
    weight = torch.exp(m - m_max)  # [batch, heads, splits]
    numerator = (partial * weight.unsqueeze(-1)).sum(dim=2)
    denominator = (l * weight).sum(dim=2, keepdim=True)
    return (numerator / denominator).to(dtype)


def reference(q, k, v) -> torch.Tensor:
    """Plain PyTorch in float32, with KV heads repeated for each query group."""
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = (q.float().unsqueeze(2) @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return (scores.softmax(dim=-1) @ v).squeeze(2).to(q.dtype)


def torch_grouped(q, k, v) -> torch.Tensor:
    """Eager PyTorch that reads each KV head once: the fair baseline.

    Query heads are reshaped into [kv_heads, group] so one batched matmul
    scores a whole group against its shared keys, with no KV copy.
    """
    batch, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    qg = q.view(batch, kv_heads, q_heads // kv_heads, head_dim)
    scores = (qg @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    out = scores.float().softmax(dim=-1).to(v.dtype) @ v
    return out.reshape(batch, q_heads, head_dim)


def torch_sdpa(q, k, v) -> torch.Tensor:
    """PyTorch's fused attention with native GQA.

    On Turing this can fall back to the unfused math backend, so it is
    reported, not used as the headline baseline.
    """
    f = torch.nn.functional.scaled_dot_product_attention
    try:
        out = f(q.unsqueeze(2), k, v, enable_gqa=True)  # torch >= 2.5
    except TypeError:
        group = q.shape[1] // k.shape[1]
        out = f(q.unsqueeze(2), k.repeat_interleave(group, 1),
                v.repeat_interleave(group, 1))
    return out.squeeze(2)


def kv_bytes(k: torch.Tensor, v: torch.Tensor) -> int:
    """Bytes the step must read at minimum: the KV cache, once."""
    return k.numel() * k.element_size() + v.numel() * v.element_size()
