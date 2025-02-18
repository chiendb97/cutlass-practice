//
// Created by root on 2/12/25.
//

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"

using namespace cute;

template <typename T>
void CheckResult(const T* result, const T* target, const int size, float eps = 1e-5)
{
    for (int i = 0; i < size; ++i)
    {
        std::cout << result[i] << " " << target[i] << std::endl;

        if (fabs(result[i] - target[i]) > eps)
        {
            std::cout << "Error" << std::endl;
        }
    }
}

template <typename T, typename TC>
__global__
void GemmReferenceKernel(const int M,
                         const int N,
                         const int K,
                         float alpha,
                         const T* A,
                         const T* B,
                         float beta,
                         TC* C)
{
    auto m = blockIdx.x * blockDim.x + threadIdx.x;
    auto n = blockIdx.y * blockDim.y + threadIdx.y;

    TC accum = 0;
    for (int k = 0; k < K; ++k)
    {
        accum += static_cast<TC>(A[m + k * M] * B[n + k * N]);
    }

    C[m + n * M] = beta * C[m + n * M] + alpha * accum;
}

template <typename T, typename TC>
void GemmReference(const int M,
                   const int N,
                   const int K,
                   float alpha,
                   const T* A,
                   const T* B,
                   float beta,
                   TC* C)
{
    dim3 block(16, 16);
    dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
    GemmReferenceKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

template <
    class OperatorClass,
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementAccumulator>
struct GemmConfiguration
{
    static_assert(sizeof(ElementA) == 0, "No valid GemmConfiguration configuration exists.");
};

template <typename Element, typename Layout, int Alignment, int SizeK>
struct Gemm_TensorOpSm80_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct Gemm_TensorOpSm80_OperandB;

// FP16: 128-by-128-by-64

// Operand A - Row-major (K-major)
template <>
struct Gemm_TensorOpSm80_OperandA<cutlass::half_t, cutlass::layout::RowMajor, 8, 64>
{
    // Smem
    using SmemLayoutAtom = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_8, _64>,
                           Stride<_64, _1>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::half_t>;

    // Gmem
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<
                            SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                            cutlass::half_t>(),
                        Layout<
                            Shape<_16, _8>,
                            Stride<_8, _1>>{},
                        Layout<
                            Shape<_1, _8>>{}
        )
    );
};

// Operand A - Column-major (M-major)
template <int SizeK>
struct Gemm_TensorOpSm80_OperandA<cutlass::half_t, cutlass::layout::ColumnMajor, 8, SizeK>
{
    // Smem
    using SmemLayoutAtom = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_64, _8>,
                           Stride<_1, _64>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, cutlass::half_t>;

    // Gmem
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<
                            SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                            cutlass::half_t>(),
                        Layout<
                            Shape<_16, _8>,
                            Stride<_1, _16>>{},
                        Layout<
                            Shape<_8, _1>>{}
        )
    );
};

// F16: 128-by-128-by-32 (small k-block)

// Operand A - Row-major (K-major)
template <>
struct Gemm_TensorOpSm80_OperandA<cutlass::half_t, cutlass::layout::RowMajor, 8, 32>
{
    // Smem
    using SmemLayoutAtom = decltype(
        composition(Swizzle<2, 3, 3>{},
                    Layout<Shape<_8, _32>,
                           Stride<_32, _1>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::half_t>;

    // Gmem
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<
                            SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                            cutlass::half_t>(),
                        Layout<
                            Shape<_32, _4>,
                            Stride<_4, _1>>{},
                        Layout<
                            Shape<_1, _8>>{}
        )
    );
};

// Operand B - Column-major (K-major)
template <int Alignment, int SizeK>
struct Gemm_TensorOpSm80_OperandB<cutlass::half_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
    : Gemm_TensorOpSm80_OperandA<cutlass::half_t, cutlass::layout::RowMajor, Alignment, SizeK>
{
};

// Operand B - Row-major (N-major)
template <int Alignment, int SizeK>
struct Gemm_TensorOpSm80_OperandB<cutlass::half_t, cutlass::layout::RowMajor, Alignment, SizeK>
    : Gemm_TensorOpSm80_OperandA<cutlass::half_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
{
};


template <typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::half_t, LayoutA,
        cutlass::half_t, LayoutB,
        float, LayoutC,
        float>
{
    using ElementA = cutlass::half_t;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = cutlass::half_t;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementC = float;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using TileShape = Shape<_128, _128, _32>;
    using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<3>;

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_2, _2, _1>>, // 2x2x1 thread group
        Tile<_32, _32, _16> // 32x32x16 MMA for LDSM, 1x2x1 value group`
    >;

    // A
    using OperandA = Gemm_TensorOpSm80_OperandA<ElementA, LayoutA, AlignmentA, 32>;
    using SmemLayoutAtomA = typename OperandA::SmemLayoutAtom;
    using SmemCopyAtomA = typename OperandA::SmemCopyAtom;
    using GmemTiledCopyA = typename OperandA::GmemTiledCopy;

    // B
    using OperandB = Gemm_TensorOpSm80_OperandB<ElementB, LayoutB, AlignmentB, 32>;
    using SmemLayoutAtomB = typename OperandB::SmemLayoutAtom;
    using SmemCopyAtomB = typename OperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename OperandB::GmemTiledCopy;

    // Mainloop
    using CollectiveMainLoop = cutlass::gemm::collective::CollectiveMma<
        DispatchPolicy,
        TileShape,
        ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
        ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
        TiledMma,
        GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, identity,
        GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, identity
    >;

    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
        ElementC,
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementCompute>,
        cutlass::gemm::EpilogueDefault
    >;
};

void GemmCutlass(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const cutlass::half_t* A, const uint64_t lda,
                 const cutlass::half_t* B, const uint64_t ldb,
                 const float beta,
                 float* C, const uint64_t ldc)
{
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Config = GemmConfiguration<
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::half_t, LayoutA,
        cutlass::half_t, LayoutB,
        float, LayoutC,
        float>;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        Config::CollectiveMainLoop,
        Config::CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    auto dA = Gemm::CollectiveMainloop::StrideA(lda, cute::Int<1>{}, _);
    auto dB = Gemm::CollectiveMainloop::StrideB(ldb, cute::Int<1>{}, _);
    auto dC = Gemm::CollectiveEpilogue::StrideC(ldc, cute::Int<1>{}, _);

    Gemm gemm;

    Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {
            A, dA,
            B, dB
        },
        {
            {alpha, beta},
            C, dC,
            C, dC
        }
    };

    auto status = gemm(arguments);

    if (status == cutlass::Status::kSuccess)
    {
        std::cout << "Success" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}


int main()
{
    int M = 512;
    int N = 512;
    int K = 512;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;

    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A(cutlass::MatrixCoord(M, K));
    cutlass::HostTensor<ElementB, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C_cutlass(cutlass::MatrixCoord(M, N));
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C_reference(cutlass::MatrixCoord(M, N));

    uint64_t seed = 42;

    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stddev = 5.0_hf;

    int bits_less_than_one = 0;

    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(),
        seed * 2,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(),
        seed * 3,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
        C_cutlass.device_view(),
        seed * 5,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::device_memory::copy_device_to_device(
        C_reference.device_data(),
        C_cutlass.device_data(),
        C_cutlass.capacity()
    );

    C_cutlass.sync_host();
    C_reference.sync_host();

    GemmCutlass(
        M, N, K,
        1.0f,
        A.device_data(), A.stride(0),
        B.device_data(), B.stride(0),
        0.0f,
        C_cutlass.device_data(), C_cutlass.stride(0)
    );

    cudaDeviceSynchronize();

    GemmReference(
        M, N, K,
        1.0f,
        A.device_data(),
        B.device_data(),
        0.0f,
        C_reference.device_data()
    );

    C_cutlass.sync_host();
    C_reference.sync_host();

    CheckResult(C_reference.host_data(), C_reference.host_data(), M * N);
    return 0;
}
