// PyTorch binding for the FlashAttention-style fused multi-head attention forward
// (csrc/cute/fmha/hopper/fmha_forward.cuh).
//
// The kernel natively addresses Q/K/V/O in a BMHK memory layout -- (B, S, H, D) contiguous, heads
// interleaved within each sequence position (see gmemLayout* in fmha_forward.cuh: logical dims
// (M, K, H, B) with strides (K*H, 1, K, H*M*K)). One op is exposed:
//
//   fmha_forward(q, k, v, scale?) -> o            raw kernel layout BMHK (B, S, H, D) in/out,
//                                                 contiguous, NO transpose (zero-copy).
//
// q/k/v are float16/bfloat16; o has the same dtype. Head dim D must be 64, 128, or 256 (compile-time
// template). The device routine consumes scale already multiplied by log2(e), since it uses exp2
// internally for the online softmax.

#include <cmath>

#include "dispatch.cuh"
#include "fmha_forward.cuh"

namespace
{
using namespace cute_bindings;

constexpr double kLog2e = 1.4426950408889634;

// Core launcher: q=(B,Sq,H,D), k/v=(B,Sk,H,D) contiguous in the kernel's native BMHK layout;
// returns o=(B,Sq,H,D) BMHK. No transposes -- callers must hand it contiguous BMHK tensors.
template <class Elem, int HEADDIM>
torch::Tensor run(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double softmax_scale)
{
    const int B = q.size(0);
    const int Sq = q.size(1);
    const int H = q.size(2);
    const int Sk = k.size(1);
    const int L = B * H;

    auto o = torch::empty({B, Sq, H, HEADDIM}, q.options()); // (B, Sq, H, D) BMHK

    auto scores = torch::empty({L, Sq * Sk}, q.options());                 // S scratch (operand dtype)
    auto fopts = q.options().dtype(torch::kFloat32);
    auto mi = torch::empty({L, Sq}, fopts);                                // running max
    auto sprime = torch::empty({L, Sq}, fopts);                            // running denominator

    const float scale = static_cast<float>(softmax_scale * kLog2e);
    cute_fmha::fmhaForwardDevice<Elem, float, HEADDIM>(Sq, Sk, H, B, as_const<Elem>(q), as_const<Elem>(k), as<Elem>(v),
                                                       as<Elem>(scores), as<Elem>(o), mi.data_ptr<float>(), sprime.data_ptr<float>(),
                                                       scale, current_stream());
    return o;
}

// Raw kernel-layout entry: q=(B,Sq,H,D), k/v=(B,Sk,H,D) contiguous BMHK; returns (B,Sq,H,D) BMHK.
// No transpose/copy, so a benchmark can time the kernel itself rather than the layout conversion.
torch::Tensor fmha_forward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, c10::optional<double> scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "cute_kernels: fmha inputs must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "cute_kernels: fmha expects (B, S, H, D) tensors");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "cute_kernels: q/k/v dtype mismatch");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "cute_kernels: fmha requires contiguous BMHK tensors");
    const int D = q.size(3);
    const double sc = scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(D));

    torch::Tensor out;
    CUTE_DISPATCH_HALF_BF16(q.scalar_type(), Elem, {
        switch (D)
        {
        case 64: out = run<Elem, 64>(q, k, v, sc); break;
        case 128: out = run<Elem, 128>(q, k, v, sc); break;
        case 256: out = run<Elem, 256>(q, k, v, sc); break;
        default: TORCH_CHECK(false, "cute_kernels: fmha head dim must be 64, 128 or 256, got ", D);
        }
    });
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(cute_kernels, m)
{
    m.def("fmha_forward(Tensor q, Tensor k, Tensor v, float? scale) -> Tensor", &fmha_forward);
}
