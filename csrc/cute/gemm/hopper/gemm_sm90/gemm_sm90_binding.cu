// PyTorch binding for the basic Hopper TMA + WGMMA GEMM (csrc/cute/gemm/hopper/gemm_sm90/gemm_sm90.cuh).
//
// Op:  gemm_sm90(a, b) -> a @ b^T
//   a : (M, K) row-major, float16/bfloat16
//   b : (N, K) row-major, float16/bfloat16
//   out: (M, N), same dtype as a
//
// The kernel's TN path expects row-major A=(M,K) and B=(N,K) and writes C column-major
// (stride (1, M)). We therefore launch into an (N, M) row-major buffer (whose memory is
// exactly column-major (M, N)) and return its transpose as the (M, N) result.

#include "dispatch.cuh"
#include "gemm_sm90.cuh"

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

    auto cbuf = torch::zeros({N, M}, a.options()); // column-major (M, N) view of the output

    Elem alpha = Elem(1.0f);
    Elem beta = Elem(0.0f);
    cute_gemm_sm90::gemm('T', 'N', M, N, K, alpha, as_const<Elem>(a), /*ldA=*/K, as_const<Elem>(b), /*ldB=*/K, beta, as<Elem>(cbuf),
                         /*ldC=*/M, current_stream());

    return cbuf.transpose(0, 1); // (M, N) = a @ b^T
}

torch::Tensor gemm_sm90(const torch::Tensor& a, const torch::Tensor& b)
{
    check_gemm_inputs(a, b);
    torch::Tensor out;
    CUTE_DISPATCH_HALF_BF16(a.scalar_type(), Elem, { out = run<Elem>(a, b); });
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(cute_kernels, m)
{
    m.def("gemm_sm90(Tensor a, Tensor b) -> Tensor", &gemm_sm90);
}
