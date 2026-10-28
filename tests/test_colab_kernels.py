"""Kernel correctness. Runs on a CUDA GPU, or on CPU with TRITON_INTERPRET=1."""

import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if not (torch.cuda.is_available() or INTERPRET):
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab" / "kernels"))
import decode_attention as da  # noqa: E402
from softmax import softmax  # noqa: E402


@pytest.mark.parametrize("cols", [1, 7, 128, 1000, 4096])
def test_softmax_matches_torch(cols: int) -> None:
    x = torch.randn(33, cols, device=DEVICE, dtype=torch.float16)
    torch.testing.assert_close(softmax(x), torch.softmax(x.float(), -1).half(),
                               atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("context", [1, 63, 64, 65, 1000, 4096])
@pytest.mark.parametrize("splits", [1, 3, 8])
def test_decode_attention_matches_reference(context: int, splits: int) -> None:
    q = torch.randn(2, 28, 128, device=DEVICE, dtype=torch.float16)
    k = torch.randn(2, 4, context, 128, device=DEVICE, dtype=torch.float16)
    v = torch.randn_like(k)
    expected = da.reference(q, k, v)
    torch.testing.assert_close(da.decode_attention(q, k, v), expected,
                               atol=2e-3, rtol=2e-2)
    # More splits than blocks leaves some splits empty; the merge must cope.
    torch.testing.assert_close(da.decode_attention_split(q, k, v, splits=splits),
                               expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(da.decode_attention_grouped(q, k, v, splits=splits),
                               expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(da.torch_grouped(q, k, v), expected,
                               atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("context", [1, 255, 300, 4096])
def test_split_defaults_match_reference(context: int) -> None:
    q = torch.randn(1, 28, 128, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 4, context, 128, device=DEVICE, dtype=torch.float16)
    v = torch.randn_like(k)
    torch.testing.assert_close(da.decode_attention_split(q, k, v),
                               da.reference(q, k, v), atol=2e-3, rtol=2e-2)


def test_default_splits() -> None:
    assert da.default_splits(1, 28, 100) == 1
    assert da.default_splits(1, 28, 8192) == 32
    assert da.default_splits(8, 28, 8192) == 18


@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (16, 2), (32, 1)])
def test_grouped_decode_handles_group_sizes(q_heads: int, kv_heads: int) -> None:
    """Group sizes 1, 8 and 32: padded below 16, exact at 16, larger than 16."""
    q = torch.randn(1, q_heads, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, kv_heads, 300, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn_like(k)
    torch.testing.assert_close(da.decode_attention_grouped(q, k, v, splits=3),
                               da.reference(q, k, v), atol=2e-3, rtol=2e-2)
