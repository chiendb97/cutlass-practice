// PyTorch binding for the FlashAttention-style fused multi-head attention forward
// (csrc/cute/fmha/hopper/fmha_forward.cuh).
//
// Op:  fmha_forward(q, k, v, scale?) -> o
//   q : (B, H, Sq, D) float16/bfloat16, contiguous
//   k : (B, H, Sk, D) float16/bfloat16, contiguous
//   v : (B, H, Sk, D) float16/bfloat16, contiguous
//   scale : optional softmax scale (default 1/sqrt(D))
//   o : (B, H, Sq, D), same dtype as q
//
// Head dim D must be 64, 128, or 256 (compile-time template). The device routine consumes
// scale already multiplied by log2(e), since it uses exp2 internally for the online softmax.

#include <cmath>

#include "dispatch.cuh"
#include "fmha_forward.cuh"

namespace
{
using namespace cute_bindings;

constexpr double kLog2e = 1.4426950408889634;

template <class Elem, int HEADDIM>
torch::Tensor run(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double softmax_scale)
{
    const int B = q.size(0);
    const int H = q.size(1);
    const int Sq = q.size(2);
    const int Sk = k.size(2);
    const int L = B * H;

    auto o = torch::empty_like(q);
    auto scores = torch::empty({L, Sq * Sk}, q.options());                 // S scratch (operand dtype)
    auto fopts = q.options().dtype(torch::kFloat32);
    auto mi = torch::empty({L, Sq}, fopts);                                // running max
    auto sprime = torch::empty({L, Sq}, fopts);                            // running denominator

    const float scale = static_cast<float>(softmax_scale * kLog2e);
    cute_fmha::fmhaForwardDevice<Elem, float, HEADDIM>(Sq, Sk, H, B, as_const<Elem>(q), as_const<Elem>(k), as<Elem>(v), as<Elem>(scores),
                                                       as<Elem>(o), mi.data_ptr<float>(), sprime.data_ptr<float>(), /*iterations=*/1, scale,
                                                       current_stream());
    return o;
}

torch::Tensor fmha_forward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, c10::optional<double> scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "cute_kernels: fmha inputs must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "cute_kernels: fmha expects (B, H, S, D) tensors");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "cute_kernels: q/k/v dtype mismatch");
    const auto qc = q.contiguous();
    const auto kc = k.contiguous();
    const auto vc = v.contiguous();
    const int D = qc.size(3);
    const double sc = scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(D));

    torch::Tensor out;
    CUTE_DISPATCH_HALF_BF16(qc.scalar_type(), Elem, {
        switch (D)
        {
        case 64: out = run<Elem, 64>(qc, kc, vc, sc); break;
        case 128: out = run<Elem, 128>(qc, kc, vc, sc); break;
        case 256: out = run<Elem, 256>(qc, kc, vc, sc); break;
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
