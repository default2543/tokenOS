"""Check the Triton kernels against PyTorch, then time them on the GPU.

Correctness first: no timing is reported for a kernel that disagrees with the
reference. Timings are medians from `triton.testing.do_bench`, which warms up
and synchronizes with CUDA events. Bandwidth is the minimum bytes the
operation must move divided by its median time, compared to the GPU's peak.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import triton
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_attention as da  # noqa: E402
from softmax import softmax  # noqa: E402

# Qwen2.5-7B attention, the model served in serving_bench.py.
Q_HEADS, KV_HEADS, HEAD_DIM, LAYERS = 28, 4, 128, 28
# Context lengths around TokenOS's measured prompts: ~1.4K tokens for adaptive
# at every attempt, ~8K for full history by attempt 10.
CONTEXTS = (1024, 1536, 4096, 8192, 16384)
PEAK_GBPS = {"Tesla T4": 320.0, "NVIDIA L4": 300.0,
             "NVIDIA A100-SXM4-40GB": 1555.0, "NVIDIA A100-SXM4-80GB": 2039.0}


def _ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median")


def softmax_rows(device: str) -> list[dict]:
    rows = []
    for cols in (256, 1024, 4096, 16384):
        x = torch.randn(4096, cols, device=device, dtype=torch.float16)
        torch.testing.assert_close(softmax(x), torch.softmax(x.float(), -1).half(),
                                   atol=2e-3, rtol=2e-2)
        moved = 2 * x.numel() * x.element_size()  # read once, write once
        for name, fn in (("triton", lambda: softmax(x)),
                         ("torch", lambda: torch.softmax(x, -1))):
            ms = _ms(fn)
            rows.append({"cols": cols, "impl": name, "ms": round(ms, 4),
                         "gbps": round(moved / ms / 1e6, 1)})
    return rows


def decode_rows(device: str, batch: int) -> list[dict]:
    rows = []
    for context in CONTEXTS:
        q = torch.randn(batch, Q_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
        k = torch.randn(batch, KV_HEADS, context, HEAD_DIM, device=device,
                        dtype=torch.float16)
        v = torch.randn_like(k)
        expected = da.reference(q, k, v)
        impls = {
            "triton": lambda: da.decode_attention(q, k, v),
            "triton-split": lambda: da.decode_attention_split(q, k, v),
            "triton-grouped": lambda: da.decode_attention_grouped(q, k, v, splits=8),
            "torch-grouped": lambda: da.torch_grouped(q, k, v),
            "torch-sdpa": lambda: da.torch_sdpa(q, k, v),
        }
        for name, fn in impls.items():
            torch.testing.assert_close(fn(), expected, atol=2e-3, rtol=2e-2)
        moved = da.kv_bytes(k, v)
        for name, fn in impls.items():
            ms = _ms(fn)
            rows.append({
                "batch": batch, "context": context, "impl": name,
                "us": round(ms * 1000, 2),
                "gbps": round(moved / ms / 1e6, 1),
                # One output token runs this once per layer.
                "attention_ms_per_token_all_layers": round(ms * LAYERS, 3),
            })
    return rows


# The third run's sweep (splits 4-64, blocks 32-128, warps 2-8) found split-K
# fastest with 128-token blocks and 2 warps, the edge of that grid, and found
# tl.dot does not lower to mma.sync on Turing. This sweep extends past the edge.
SWEEP_SPLITS = (4, 8, 16, 32)
SWEEP_BLOCK_N = (64, 128, 256)
SWEEP_WARPS = (1, 2, 4)
SWEEP_SHAPES = ((1, 1536), (1, 8192), (8, 1536), (8, 8192))


def sweep_rows(device: str) -> dict:
    kernels = {"triton-split": da.decode_attention_split}
    rows = []
    for batch, context in SWEEP_SHAPES:
        q = torch.randn(batch, Q_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
        k = torch.randn(batch, KV_HEADS, context, HEAD_DIM, device=device,
                        dtype=torch.float16)
        v = torch.randn_like(k)
        expected = da.reference(q, k, v)
        moved = da.kv_bytes(k, v)
        for name, fn in kernels.items():
            for splits in SWEEP_SPLITS:
                for block_n in SWEEP_BLOCK_N:
                    for warps in SWEEP_WARPS:
                        run = lambda: fn(q, k, v, splits=splits, block_n=block_n,
                                         num_warps=warps)
                        try:
                            torch.testing.assert_close(run(), expected,
                                                       atol=2e-3, rtol=2e-2)
                            ms = triton.testing.do_bench(run, warmup=10, rep=50,
                                                         return_mode="median")
                        except Exception as error:  # e.g. out of shared memory
                            rows.append({"batch": batch, "context": context,
                                         "impl": name, "splits": splits,
                                         "block_n": block_n, "warps": warps,
                                         "error": type(error).__name__})
                            continue
                        rows.append({"batch": batch, "context": context, "impl": name,
                                     "splits": splits, "block_n": block_n,
                                     "warps": warps, "us": round(ms * 1000, 2),
                                     "gbps": round(moved / ms / 1e6, 1)})
    q = torch.randn(1, Q_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    k = torch.randn(1, KV_HEADS, 256, HEAD_DIM, device=device, dtype=torch.float16)
    compiled = da._launch(q, k, k, splits=1, block_n=64, grouped=True)[3]
    ptx = compiled.asm.get("ptx", "") if compiled is not None else ""
    return {"configs": rows, "grouped_uses_mma": "mma.sync" in ptx}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/kernels.json"))
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (in Colab: Runtime > Change runtime type > T4)")
    gpu = torch.cuda.get_device_name()
    report = {
        "gpu": gpu, "peak_gbps": PEAK_GBPS.get(gpu),
        "torch": torch.__version__, "triton": triton.__version__,
        "softmax": softmax_rows("cuda"),
        "decode_attention": [row for b in args.batches.split(",")
                             for row in decode_rows("cuda", int(b))],
    }
    if not args.no_sweep:
        report["sweep"] = sweep_rows("cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{gpu}: peak {report['peak_gbps']} GB/s")
    for row in report["decode_attention"]:
        share = (f"  {100 * row['gbps'] / report['peak_gbps']:5.1f}% of peak"
                 if report["peak_gbps"] else "")
        print(f"  b={row['batch']:<2} ctx={row['context']:<6} {row['impl']:<15} "
              f"{row['us']:>9} us  {row['gbps']:>6} GB/s{share}")
    if "sweep" in report:
        print(f"\nSweep (best of {len(SWEEP_SPLITS) * len(SWEEP_BLOCK_N) * len(SWEEP_WARPS)}"
              f" configs each); grouped kernel uses mma.sync: "
              f"{report['sweep']['grouped_uses_mma']}")
        best = {}
        for row in report["sweep"]["configs"]:
            key = (row["batch"], row["context"], row["impl"])
            if "us" in row and (key not in best or row["us"] < best[key]["us"]):
                best[key] = row
        for (batch, context, impl), row in sorted(best.items()):
            print(f"  b={batch:<2} ctx={context:<6} {impl:<15} {row['us']:>9} us  "
                  f"{row['gbps']:>6} GB/s  splits={row['splits']} "
                  f"block_n={row['block_n']} warps={row['warps']}")
        failed = sum("error" in row for row in report["sweep"]["configs"])
        if failed:
            print(f"  {failed} configs failed to compile or run; see the JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
