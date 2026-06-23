// PyTorch binding for the warp-specialized (producer/consumer) Hopper GEMM
// (csrc/cute/gemm/hopper/pipeline_gemm/gemm_warp_specialization/hopper_gemm_kernel_launch.h).
//
// Op:  gemm_ws(a, b) -> a @ b^T
//   a : (M, K) row-major, float16/bfloat16
//   b : (N, K) row-major, float16/bfloat16
//   out: (M, N), same dtype as a
//
// gemm_tn() is TN-only and writes C row-major (stride (N, 1)). The default (non-validation)
// kernel traits use 256x256x96 tiles with a 1x2 cluster, so M/N/K work best as multiples of
// the tile (256/256/96); odd sizes are handled by the tile scheduler but are untested here.

#include "dispatch.cuh"
#include "hopper_gemm_kernel_launch.h"

namespace
{
using namespace cute_bindings;

template <class Elem>
torch::Tensor run(const torch::Tensor& a_in, const torch::Tensor& b_in)
{
    auto a = a_in.contiguous();
    auto b = b_in.contiguous();
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);

    auto out = torch::zeros({M, N}, a.options());

    Elem alpha = Elem(1.0f);
    Elem beta = Elem(0.0f);
    // ::gemm_tn lives at global scope (its signature differs from the namespaced GEMMs).
    gemm_tn(M, N, K, alpha, as_const<Elem>(a), /*ldA=*/K, as_const<Elem>(b), /*ldB=*/K, beta, as<Elem>(out), /*ldC=*/N, current_stream());

    return out; // (M, N) = a @ b^T
}

torch::Tensor gemm_ws(const torch::Tensor& a, const torch::Tensor& b)
{
    check_gemm_inputs(a, b);
    torch::Tensor out;
    CUTE_DISPATCH_HALF_BF16(a.scalar_type(), Elem, { out = run<Elem>(a, b); });
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(cute_kernels, m)
{
    m.def("gemm_ws(Tensor a, Tensor b) -> Tensor", &gemm_ws);
}
