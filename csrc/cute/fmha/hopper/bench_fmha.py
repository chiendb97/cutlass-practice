"""Standalone benchmark for the CuTe Hopper FMHA (torch.ops.cute_kernels.fmha_forward) vs. SDPA.

Non-causal forward attention, q=(B,H,S,D), k/v=(B,H,S,D). Self-contained: loads the compiled
extension itself, and carries its own CUDA-event timing, fp32 reference check, and CLI (no shared
bench helpers). ``--modal`` runs on a Modal H100 via the co-located modal_app.py.

    python csrc/cute/fmha/hopper/bench_fmha.py --B 4 --H 8 --S 512 --D 128 --dtype fp16
    python csrc/cute/fmha/hopper/bench_fmha.py --B 4 --H 8 --S 512 --D 128 --modal
"""

from __future__ import annotations

import argparse

_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def _load_ops(torch):
    # TORCH_LIBRARY extension (no PyInit): locate the .so and dlopen it.
    import importlib.util

    spec = importlib.util.find_spec("fmha_C")
    if spec is None or spec.origin is None:
        raise ImportError("fmha_C not built; run: pip install -e csrc/cute/fmha/hopper")
    torch.ops.load_library(spec.origin)


def benchmark(B, H, S, D, dtype="fp16", iters=50):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    _load_ops(torch)
    td = getattr(torch, _DTYPES[dtype])

    q = torch.randn(B, H, S, D, device="cuda", dtype=td)
    k = torch.randn(B, H, S, D, device="cuda", dtype=td)
    v = torch.randn(B, H, S, D, device="cuda", dtype=td)
    scale = 1.0 / (D ** 0.5)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    # two GEMMs of (S,S,D) and (S,D,S) per (B,H) -> 4*B*H*S*S*D
    flops = 4.0 * B * H * S * S * D

    # The kernel runs natively in a BMHK (B, S, H, D) contiguous layout. Build the inputs in that
    # layout ONCE here (outside the timed loop) and call the fmha_forward op, which does NO
    # transpose -- so the timing reflects the kernel itself, not the BHMK<->BMHK layout copies.
    # SDPA keeps the standard BHMK tensors (q/k/v). The BMHK output is transposed back to BHMK only
    # for the correctness check (untimed), via row()'s `post` hook.
    qb = q.transpose(1, 2).contiguous()  # (B, S, H, D) BMHK
    kb = k.transpose(1, 2).contiguous()  # (B, S, H, D) BMHK
    vb = v.transpose(1, 2).contiguous()  # (B, S, H, D) BMHK

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

    def row(name, fn, check=False, post=None):
        # `post` adjusts the output for the correctness check ONLY (e.g. BMHK->BHMK so it lines up
        # with `ref`); it is applied outside time_ms so layout fixes never count toward the timing.
        try:
            out = fn()
            cmp = post(out) if post is not None else out
            err = (cmp.float() - ref).abs().max().item() if check else None
            ms = time_ms(fn)
            line = f"  {name:<18} {ms:8.3f} ms   {flops / (ms * 1e-3) / 1e12:8.1f} TFLOP/s"
            if err is not None:
                line += f"   max_abs_err={err:.3e}"
            print(line)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"  {name:<18} FAILED: {type(e).__name__}: {e}")

    print(f"\nfmha  B={B} H={H} S={S} D={D}  dtype={dtype}")
    row("torch sdpa", lambda: F.scaled_dot_product_attention(q, k, v, scale=scale))
    # BMHK op: no transpose inside the call, so timing is the kernel only. Its (B,S,H,D) output
    # is transposed back to (B,H,S,D) just for the ref comparison (untimed, via `post`).
    row("cute fmha", lambda: torch.ops.cute_kernels.fmha_forward(qb, kb, vb, scale), check=True,
        post=lambda o: o.transpose(1, 2))


def main():
    p = argparse.ArgumentParser(description="Benchmark torch.ops.cute_kernels.fmha_forward")
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--S", type=int, default=512)
    p.add_argument("--D", type=int, choices=[64, 128, 256], default=128)
    p.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--modal", action="store_true", help="run on a Modal H100 instead of locally")
    args = p.parse_args()

    if args.modal:
        from modal_app import run_remote

        run_remote(__file__, B=args.B, H=args.H, S=args.S, D=args.D, dtype=args.dtype, iters=args.iters)
    else:
        benchmark(args.B, args.H, args.S, args.D, args.dtype, args.iters)


if __name__ == "__main__":
    main()
