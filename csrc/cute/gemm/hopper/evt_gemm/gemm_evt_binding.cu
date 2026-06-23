// PyTorch binding for the EVT (Epilogue Visitor Tree) Hopper GEMM
// (csrc/cute/gemm/hopper/evt_gemm/gemm_evt_sm90.cuh).
//
// Op:  gemm_evt(a, b, c, alpha, beta) -> alpha * (a @ b^T) + beta * c
//   a : (M, K) row-major, float16/bfloat16
//   b : (N, K) row-major, float16/bfloat16
//   c : (M, N) float32 (the epilogue source / bias)
//   out: (M, N) float32
//
// The EVT kernel accumulates in float and reads C / writes D column-major (stride (1, M)).
// We feed a column-major view of C and launch into a column-major output buffer, returning
// the transpose as the (M, N) result.

#include "dispatch.cuh"
#include "gemm_evt_sm90.cuh"

namespace
{
using namespace cute_bindings;

template <class Elem>
torch::Tensor run(const torch::Tensor& a_in, const torch::Tensor& b_in, const torch::Tensor& c_in, double alpha, double beta)
{
    auto a = a_in.contiguous();
    auto b = b_in.contiguous();
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);
    TORCH_CHECK(c_in.dim() == 2 && c_in.size(0) == M && c_in.size(1) == N, "cute_kernels: gemm_evt expects c of shape (M, N)");

    // Column-major (M, N) views are stored as contiguous (N, M) row-major buffers.
    auto cbuf = c_in.to(torch::kFloat32).t().contiguous(); // (N, M) == column-major (M, N) source
    auto dbuf = torch::empty({N, M}, c_in.options().dtype(torch::kFloat32));

    cute_gemm_evt::evt_gemm_run<Elem, Elem>('T', 'N', M, N, K, static_cast<float>(alpha), as_const<Elem>(a), /*ldA=*/K, as_const<Elem>(b),
                                            /*ldB=*/K, static_cast<float>(beta), as<float>(cbuf), /*ldC=*/M, as<float>(dbuf), /*ldD=*/M,
                                            current_stream());

    return dbuf.transpose(0, 1); // (M, N) float32
}

torch::Tensor gemm_evt(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& c, double alpha, double beta)
{
    check_gemm_inputs(a, b);
    TORCH_CHECK(c.is_cuda(), "cute_kernels: c must be a CUDA tensor");
    torch::Tensor out;
    CUTE_DISPATCH_HALF_BF16(a.scalar_type(), Elem, { out = run<Elem>(a, b, c, alpha, beta); });
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(cute_kernels, m)
{
    m.def("gemm_evt(Tensor a, Tensor b, Tensor c, float alpha, float beta) -> Tensor", &gemm_evt);
}
