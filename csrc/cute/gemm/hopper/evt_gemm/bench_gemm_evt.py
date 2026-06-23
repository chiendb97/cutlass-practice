"""Standalone benchmark for the EVT GEMM: D = alpha*(A@B^T) + beta*C (float32 C/D).

Self-contained: loads the compiled extension itself, and carries its own CUDA-event timing, fp32
reference check, and CLI (no shared bench helpers). ``--modal`` runs on a Modal H100 via the
co-located modal_app.py.

    python csrc/cute/gemm/hopper/evt_gemm/bench_gemm_evt.py --M 4096 --N 4096 --K 4096 --dtype fp16
"""

from __future__ import annotations

import argparse

_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def _load_ops(torch):
    # TORCH_LIBRARY extension (no PyInit): locate the .so and dlopen it.
    import importlib.util

    spec = importlib.util.find_spec("gemm_evt_C")
    if spec is None or spec.origin is None:
        raise ImportError("gemm_evt_C not built; run: pip install -e csrc/cute/gemm/hopper/evt_gemm")
    torch.ops.load_library(spec.origin)


def benchmark(M, N, K, dtype="fp16", iters=50):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    _load_ops(torch)
    td = getattr(torch, _DTYPES[dtype])

    s = K ** -0.25
    a = torch.randn(M, K, device="cuda", dtype=td) * s
    b = torch.randn(N, K, device="cuda", dtype=td) * s
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref = (a.float() @ b.float().t()) + c  # alpha=beta=1
    flops = 2.0 * M * N * K

    def time_ms(fn, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def row(name, fn, check=False):
        try:
            out = fn()
            err = (out.float() - ref).abs().max().item() if check else None
            ms = time_ms(fn)
            line = f"  {name:<18} {ms:8.3f} ms   {flops / (ms * 1e-3) / 1e12:8.1f} TFLOP/s"
            if err is not None:
                line += f"   max_abs_err={err:.3e}"
            print(line)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"  {name:<18} FAILED: {type(e).__name__}: {e}")

    print(f"\ngemm_evt  M={M} N={N} K={K}  dtype={dtype}  (D = A@B^T + C)")
    row("torch baseline", lambda: torch.matmul(a, b.t()).float() + c)
    row("gemm_evt", lambda: torch.ops.cute_kernels.gemm_evt(a, b, c, 1.0, 1.0), check=True)


def main():
    p = argparse.ArgumentParser(description="Benchmark torch.ops.cute_kernels.gemm_evt")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--modal", action="store_true", help="run on a Modal H100 instead of locally")
    args = p.parse_args()

    if args.modal:
        from modal_app import run_remote

        run_remote(__file__, M=args.M, N=args.N, K=args.K, dtype=args.dtype, iters=args.iters)
    else:
        benchmark(args.M, args.N, args.K, args.dtype, args.iters)


if __name__ == "__main__":
    main()
