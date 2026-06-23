// PyTorch binding for the multistage Hopper GEMM with TMA store
// (csrc/cute/gemm/hopper/pipeline_gemm/gemm_multistage/gemm_multistage_sm90.cuh).
//
// Op:  gemm_multistage(a, b) -> a @ b^T
//   a : (M, K) row-major, float16/bfloat16
//   b : (N, K) row-major, float16/bfloat16
//   out: (M, N), same dtype as a
//
// The TN path writes C row-major (stride (N, 1)), so we launch directly into a row-major
// (M, N) output tensor.

#include "dispatch.cuh"
#include "gemm_multistage_sm90.cuh"

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
    cute_gemm_multistage::gemm('T', 'N', M, N, K, alpha, as_const<Elem>(a), /*ldA=*/K, as_const<Elem>(b), /*ldB=*/K, beta, as<Elem>(out),
                               /*ldC=*/N, current_stream());

    return out; // (M, N) = a @ b^T
}

torch::Tensor gemm_multistage(const torch::Tensor& a, const torch::Tensor& b)
{
    check_gemm_inputs(a, b);
    torch::Tensor out;
    CUTE_DISPATCH_HALF_BF16(a.scalar_type(), Elem, { out = run<Elem>(a, b); });
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(cute_kernels, m)
{
    m.def("gemm_multistage(Tensor a, Tensor b) -> Tensor", &gemm_multistage);
}
