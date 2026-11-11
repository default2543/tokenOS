"""Row-wise fused softmax in Triton: the warm-up kernel.

One program per row reads the row once, computes max, exponentials, sum and
the normalized result in registers, and writes once. PyTorch's eager softmax
is a single kernel too, so the point is learning the model, not beating it.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)  # subtract the max for numerical stability
    numerator = tl.exp(x)
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_stride + cols, y.to(out_ptr.dtype.element_ty),
             mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    interpret = os.environ.get("TRITON_INTERPRET") == "1"  # CPU testing
    if x.ndim != 2 or not (x.is_cuda or interpret):
        raise ValueError("expected a 2-D CUDA tensor")
    rows, cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    num_warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
    _softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols,
                             BLOCK=block, num_warps=num_warps)
    return out
