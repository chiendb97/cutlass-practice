"""Elementwise fused multiply-add in the CuTe DSL: ``c = alpha * a + b``.

A minimal, self-contained example for learning the CuTe DSL Python frontend. Unlike the
hand-written C++ kernels under ``csrc/``, here the *kernel itself* is written in Python and
JIT-compiled by the DSL for the current GPU. Because it only uses universal (arch-agnostic)
copies and register arithmetic, it runs on any CUDA GPU -- including non-Hopper cards.

Key DSL concepts demonstrated:
  * ``@cute.kernel`` / ``@cute.jit``  -- device kernel vs. host launch function.
  * Thread-Value (TV) layouts + ``cute.make_tiled_copy_tv`` -- canonical CuTe partitioning.
  * ``cute.zipped_divide`` -- tile a global tensor into (tile, rest) for per-CTA slicing.
  * Coordinate tensors + ``cute.elem_less`` -- predication to guard out-of-bounds elements
    when the problem shape is not a multiple of the tile.
  * ``from_dlpack`` + ``cute.compile`` -- wrap torch tensors and compile the kernel.

Run:
    python -m cute_dsl.elementwise --M 1024 --N 1024
"""

import argparse
from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def axpy_kernel(
    gA: cute.Tensor,        # ((TileM,TileN),(RestM,RestN)) tiled view of A
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,        # identity/coordinate tensor, same tiling as gC
    shape: cute.Shape,      # original (M, N), for predication
    alpha: cutlass.Float32,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Slice out this CTA's tile: index the "rest" mode with the block id.
    blk = ((None, None), bidx)
    blkA, blkB, blkC, blkCrd = gA[blk], gB[blk], gC[blk], cC[blk]

    # Universal copy atoms (work on any architecture).
    ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)
    tiled_ld = cute.make_tiled_copy_tv(ld, thr_layout, val_layout)
    tiled_st = cute.make_tiled_copy_tv(st, thr_layout, val_layout)

    thr_ld = tiled_ld.get_slice(tidx)
    thr_st = tiled_st.get_slice(tidx)

    thrA = thr_ld.partition_S(blkA)
    thrB = thr_ld.partition_S(blkB)
    thrC = thr_st.partition_S(blkC)
    thrCrd = thr_st.partition_S(blkCrd)

    # Per-thread register fragments.
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # Predicate: which of this thread's elements are in-bounds?
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(ld, thrA, frgA, pred=frgPred)
    cute.copy(ld, thrB, frgB, pred=frgPred)

    # The actual math, in registers: c = alpha * a + b.
    frgC.store(alpha * frgA.load() + frgB.load())

    cute.copy(st, frgC, thrC, pred=frgPred)


@cute.jit
def axpy(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, alpha: cutlass.Float32):
    # 128 threads per CTA arranged 4x32; each thread owns a 4x4 value tile (vectorizable).
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    axpy_kernel(gA, gB, gC, cC, mC.shape, alpha, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run(M: int, N: int, alpha: float = 2.0, dtype: Type[cutlass.Numeric] = cutlass.Float32):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to run this example.")

    torch_dtype = cutlass_torch.dtype(dtype)
    a = torch.randn(M, N, device="cuda", dtype=torch_dtype)
    b = torch.randn(M, N, device="cuda", dtype=torch_dtype)
    c = torch.zeros_like(a)

    ta = from_dlpack(a).mark_layout_dynamic()
    tb = from_dlpack(b).mark_layout_dynamic()
    tc = from_dlpack(c).mark_layout_dynamic()

    compiled = cute.compile(axpy, ta, tb, tc, cutlass.Float32(alpha))
    compiled(ta, tb, tc, cutlass.Float32(alpha))
    torch.cuda.synchronize()

    ref = alpha * a + b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print(f"elementwise axpy ({M}x{N}, alpha={alpha}, {dtype}): PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CuTe DSL elementwise c = alpha*a + b")
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=2.0)
    args = p.parse_args()
    run(args.M, args.N, args.alpha)
