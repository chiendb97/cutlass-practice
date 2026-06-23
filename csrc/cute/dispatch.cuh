#pragma once

// Shared helpers for the PyTorch bindings of the CuTe Hopper kernels.
//
// The kernels are written for cutlass::half_t / cutlass::bfloat16_t inputs. These
// types are bit-compatible with at::Half / at::BFloat16, so tensor data pointers are
// reinterpret_cast to the cutlass element type at the call boundary.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

// Dispatch a statement block over the two supported floating dtypes, binding the
// cutlass element type to `ElemT` inside each branch. Usage:
//
//   torch::Tensor out;
//   CUTE_DISPATCH_HALF_BF16(a.scalar_type(), Elem, { out = run<Elem>(a, b); });
//
#define CUTE_DISPATCH_HALF_BF16(SCALAR_TYPE, ElemT, ...)                                                                                    \
    do                                                                                                                                      \
    {                                                                                                                                       \
        if ((SCALAR_TYPE) == at::kHalf)                                                                                                     \
        {                                                                                                                                   \
            using ElemT = cutlass::half_t;                                                                                                  \
            __VA_ARGS__                                                                                                                      \
        }                                                                                                                                   \
        else if ((SCALAR_TYPE) == at::kBFloat16)                                                                                            \
        {                                                                                                                                   \
            using ElemT = cutlass::bfloat16_t;                                                                                              \
            __VA_ARGS__                                                                                                                      \
        }                                                                                                                                   \
        else                                                                                                                                \
        {                                                                                                                                   \
            TORCH_CHECK(false, "cute_kernels: unsupported dtype ", (SCALAR_TYPE), " (expected float16 or bfloat16)");                        \
        }                                                                                                                                   \
    } while (0)

namespace cute_bindings
{

inline void check_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cute_kernels: inputs must be CUDA tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "cute_kernels: expected 2D tensors a=(M,K), b=(N,K)");
    TORCH_CHECK(a.size(1) == b.size(1), "cute_kernels: inner dim mismatch, a=(M,K), b=(N,K) need equal K");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "cute_kernels: a and b must have the same dtype");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16, "cute_kernels: dtype must be float16 or bfloat16");
}

inline cudaStream_t current_stream()
{
    return at::cuda::getCurrentCUDAStream().stream();
}

template <class Elem>
Elem* as(const torch::Tensor& t)
{
    return reinterpret_cast<Elem*>(t.data_ptr());
}

template <class Elem>
const Elem* as_const(const torch::Tensor& t)
{
    return reinterpret_cast<const Elem*>(t.data_ptr());
}

} // namespace cute_bindings
