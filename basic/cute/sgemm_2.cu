#include <cstdio>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


template<class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyA,
        class TB, class BStride, class BSmemLayout, class TiledCopyB,
        class TC, class CStride, class CSmemLayout, class TiledMma,
        class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copyA,
            TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copyB,
            TC *C, CStride dC, CSmemLayout, TiledMma mma,
            Alpha alpha, Beta beta) {
    CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) == cute::Int<3>{});

    CUTE_STATIC_ASSERT_V(size(copyA) == size(mma));
    CUTE_STATIC_ASSERT_V(size(copyB) == size(mma));

    static_assert(cute::is_static<ASmemLayout>::value);
    static_assert(cute::is_static<BSmemLayout>::value);
    static_assert(cute::is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(cute::size<0>(ASmemLayout{}) == cute::size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(cute::size<0>(CSmemLayout{}) == cute::size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(cute::size<0>(BSmemLayout{}) == cute::size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(CSmemLayout{}) == cute::size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(ASmemLayout{}) == cute::size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(cute::size<1>(BSmemLayout{}) == cute::size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(cute::select<0, 2>(shape_MNK), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(cute::select<1, 2>(shape_MNK), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(cute::select<0, 1>(shape_MNK), dC));         // dC strides for shape MN

    cute::Tensor mA = cute::make_tensor(cute::make_gmem_ptr(A), cute::select<0, 2>(shape_MNK), dA);
    cute::Tensor mB = cute::make_tensor(cute::make_gmem_ptr(B), cute::select<1, 2>(shape_MNK), dB);
    cute::Tensor mC = cute::make_tensor(cute::make_gmem_ptr(C), cute::select<0, 1>(shape_MNK), dC);

    auto cta_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);
    cute::Tensor gA = cute::local_tile(mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
    cute::Tensor gB = cute::local_tile(mA, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
    cute::Tensor gC = cute::local_tile(mC, cta_tiler, cta_coord, cute::Step<cute::_1, cute::_1, cute::X>{});

    __shared__ TA smemA[cute::cosize_v<ASmemLayout>];
    __shared__ TB smemB[cute::cosize_v<BSmemLayout>];

    cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(smemA), sA_layout);
    cute::Tensor sB = cute::make_tensor(cute::make_smem_ptr(smemB), sB_layout);

    auto thr_copyA = copyA.get_slice(threadIdx.x);
    cute::Tensor tAgA = thr_copyA.partition_S(gA);
    cute::Tensor tAsA = thr_copyA.partition_D(sA);
    cute::Tensor tArA = cute::make_fragment_like(tAsA);

    auto thr_copyB = copyB.get_slice(threadIdx.x);
    cute::Tensor tBgB = thr_copyB.partition_S(gB);
    cute::Tensor tBsB = thr_copyB.partition_D(sB);
    cute::Tensor tBrB = cute::make_fragment_like(tBsB);

    CUTE_STATIC_ASSERT_V(cute::size<1>(tAgA) == cute::size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(tAgA) == cute::size<1>(tArA));                // CPY_M
    CUTE_STATIC_ASSERT_V(cute::size<2>(tAgA) == cute::size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<2>(tAgA) == cute::size<2>(tArA));                // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<1>(tBgB) == cute::size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(tBgB) == cute::size<1>(tBrB));                // CPY_N
    CUTE_STATIC_ASSERT_V(cute::size<2>(tBgB) == cute::size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(cute::size<2>(tBgB) == cute::size<2>(tBrB));                // CPY_K

    cute::copy(copyA, tAgA(cute::_, cute::_, cute::_, 0), tArA);
    cute::copy(copyB, tBgB(cute::_, cute::_, cute::_, 0), tBrB);

    auto thr_mma = mma.get_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCgC = thr_mma.partition_C(gC);

    cute::Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    CUTE_STATIC_ASSERT_V(cute::shape(tCrC) == cute::shape(tCgC));
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCrC) == cute::size<1>(tCsA));
    CUTE_STATIC_ASSERT_V(cute::size<2>(tCrC) == cute::size<1>(tCsB));
    CUTE_STATIC_ASSERT_V(cute::size<2>(tCsA) == cute::size<2>(tCsB));

    cute::clear(tCrC);

#if 0
    if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
    if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
    if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1
    auto K_TILE_MAX = cute::size<3>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copyA, tAgA(cute::_, cute::_, cute::_, k_tile_next), tArA);
        copy(copyB, tBgB(cute::_, cute::_, cute::_, k_tile_next), tBrB);

        cute::gemm(mma, tCsA, tCsB, tCrC);
    }
#endif

    cute::axpby(alpha, tCrC, beta, tCgC);
}

template<class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const *A, int ldA,
        TB const *B, int ldB,
        Beta beta,
        TC *C, int ldC,
        cudaStream_t stream = 0) {
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = cute::make_shape(M, N, K);

    auto dA = cute::make_stride(cute::Int<1>{}, ldA);
    auto dB = cute::make_stride(cute::Int<1>{}, ldB);
    auto dC = cute::make_stride(cute::Int<1>{}, ldC);

    auto bM = cute::Int<128>{};
    auto bN = cute::Int<128>{};
    auto bK = cute::Int<8>{};
    auto cta_tiler = cute::make_shape(bM, bN, bK);

    auto sA = cute::make_layout(cute::make_shape(bM, bK));
    auto sB = cute::make_layout(cute::make_shape(bN, bK));
    auto sC = cute::make_layout(cute::make_shape(bM, bN));

    auto copyA = cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, TA>{},
    cute::Layout < cute::Shape<cute::_32, cute::_8>>
    {},
    cute::Layout < cute::Shape<cute::_4, cute::_1>>
    {});

    auto copyB = cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, TB>{},
    cute::Layout < cute::Shape<cute::_32, cute::_8>>
    {},
    cute::Layout < cute::Shape<cute::_4, cute::_1>>
    {});

    auto mma = cute::make_tiled_mma(cute::UniversalFMA<TC, TA, TB>{},
    cute::Layout < cute::Shape<cute::_16, cute::_16, cute::_1>>
    {});

#if 0
    print(copyA);
    print(copyB);
    print(mma);
#endif

#if 0
    print_latex(copyA);
    print_latex(copyB);
    print_latex(mma);
#endif

    dim3 dimBlock(cute::size(mma));
    dim3 dimGrid(cute::size(ceil_div(M, bM)),
                 cute::size(ceil_div(N, bN)));

    gemm_device<<<dimGrid, dimBlock, 0, stream>>>
            (prob_shape, cta_tiler,
             A, dA, sA, copyA,
             B, dB, sB, copyB,
             C, dC, sC, mma,
             alpha, beta);
}

template<class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const *A, int ldA,
        TB const *B, int ldB,
        Beta beta,
        TC *C, int ldC,
        cudaStream_t stream = 0) {
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = cute::make_shape(M, N, K);

    auto dA = cute::make_stride(ldA, cute::Int<1>{});
    auto dB = cute::make_stride(ldB, cute::Int<1>{});
    auto dC = cute::make_stride(cute::Int<1>{}, ldC);

    auto bM = cute::Int<128>{};
    auto bN = cute::Int<128>{};
    auto bK = cute::Int<8>{};
    auto cta_tiler = cute::make_shape(bM, bN, bK);

    auto sA = cute::make_layout(cute::make_shape(bM, bK),
                                cute::make_stride(cute::Int<1>{}, bM + cute::Int<1>{}));
    auto sB = cute::make_layout(cute::make_shape(bN, bK),
                                cute::make_stride(cute::Int<1>{}, bN + cute::Int<1>{}));
    auto sC = cute::make_layout(cute::make_shape(bM, bN));

    auto copyA = cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<TA>, TA>{},
    cute::Layout < cute::Shape<cute::_32, cute::_8>, cute::Stride<cute::_8, cute::_1>>
    {},
    cute::Layout < cute::Shape<cute::_1, cute::_1>>
    {});

    auto copyB = cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<TA>, TB>{},
    cute::Layout < cute::Shape<cute::_32, cute::_8>, cute::Stride<cute::_8, cute::_1>>
    {},
    cute::Layout < cute::Shape<cute::_1, cute::_1>>
    {});

    auto mma = cute::make_tiled_mma(cute::UniversalFMA<TC, TA, TB>{},
    cute::Layout < cute::Shape<cute::_16, cute::_16, cute::_1>>
    {});

#if 0
    print(copyA);
    print(copyB);
    print(mma);
#endif

#if 0
    print_latex(copyA);
    print_latex(copyB);
    print_latex(mma);
#endif

    dim3 dimBlock(cute::size(mma));
    dim3 dimGrid(cute::size(ceil_div(M, bM)),
                 cute::size(ceil_div(N, bN)));

    gemm_device<<<dimGrid, dimBlock, 0, stream>>>
            (prob_shape, cta_tiler,
             A, dA, sA, copyA,
             B, dB, sB, copyB,
             C, dC, sC, mma,
             alpha, beta);
}


template<class TA, class TB, class TC, class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const *A, int ldA,
     TB const *B, int ldB,
     Beta beta,
     TC *C, int ldC,
     cudaStream_t stream = 0) {
    if (transA == 'N' && transB == 'T') {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    } else if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not implemented");
}


int main(int argc, char **argv) {
    int m = 5120;
    if (argc >= 2) {
        sscanf(argv[1], "%d", &m);
    }

    int n = 5120;
    if (argc >= 3) {
        sscanf(argv[2], "%d", &n);
    }

    int k = 4096;
    if (argc >= 4) {
        sscanf(argv[3], "%d", &k);
    }

    char transA = 'N';
    if (argc >= 5) {
        sscanf(argv[4], "%c", &transA);
    }

    char transB = 'T';
    if (argc >= 6) {
        sscanf(argv[5], "%c", &transB);
    }

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    TI alpha = 1.0;
    TI beta = 0.0;

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    cute::device_init(0);

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    for (int i = 0; i < m * k; ++i) {
        h_A[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }

    for (int i = 0; i < n * k; ++i) {
        h_B[i] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
    }

    for (int i = 0; i < m * n; ++i) {
        h_C[i] = static_cast<TC>(-1);
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    } else if (transA == 'T') {
        ldA = k;
    } else {
        assert(false);
    }

    if (transB == 'T') {
        ldB = n;
    } else if (transB == 'N') {
        ldB = k;
    } else {
        assert(false);
    }

    d_C = h_C;
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;

    // Timing iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k,
             alpha,
             d_A.data().get(), ldA,
             d_B.data().get(), ldB,
             beta,
             d_C.data().get(), ldC);
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time * 1000);

    return 0;
}