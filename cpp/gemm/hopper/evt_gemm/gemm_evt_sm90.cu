#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>

#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace cute;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const& tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;

    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    // Tensor out = make_tensor_like<To_type>(out);
    // copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()), out);
    // return outl
}

template <class ElementA, class ElementB,
          class SmemLayoutA, // bM, bK, bP
          class SmemLayoutB> // bN, bK, bP>
struct SharedStorage
{
    cutlass::array_aligned<ElementA, cosize_v<SmemLayoutA>> smemA;

    cutlass::array_aligned<ElementB, cosize_v<SmemLayoutB>> smemB;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})]; // bP
    uint64_t mma_barrier[size<2>(SmemLayoutA{})]; // bP
};

template <class ProblemShape, class CtaTiler, class TA, class SmemLayoutA, class TmaA, class TB, class SmemLayoutB, class TmaB, class TC,
          class CStride, class TD, class DStride, class TiledMma, class Alpha, class Beta, class EVT>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                                                                        TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                                                                        TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                                                                        TC* C, CStride dC, TD* D, DStride dD, TiledMma mma,
                                                                                        Alpha alpha, Beta beta,
                                                                                        typename EVT::Params const epi_params)
{
    EVT evt(epi_params, {});

    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (bM, bN, bK)

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler)); // bM
    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler)); // bN
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler)); // bK
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler)); // bK

    // CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

    // Full and tiled tensors

    // Represent the full tensors
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));              // (M, K) TMA Tensor
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));              // (N, K) TMA Tensor
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC); // (M, N)
    Tensor mD = make_tensor(make_gmem_ptr(D), make_shape(M, N), dD); // (M, N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m, n, k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (bM, bK, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (bN, bK, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (bM, bN)
    Tensor gD = local_tile(mD, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (bM, bN)

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), SmemLayoutA{}); // (bM, bK, bP)
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), SmemLayoutB{}); // (bN, bK, bP)

    // Partition the copying of A and B tiles
    auto [tAgA, tAsA] =
        tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA)); // (TMA, k) and (TMA, bP)

    auto [tBgB, tBsB] =
        tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB)); // (TMA, k) and (TMA, bP)

    constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) + CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);

    // PREFETCH

    auto K_PIPE_MAX = size<1>(tAsA); // bP

    // Total count of tiles
    int k_tile_count = size<1>(tAgA); // k

    // Current tile index in gmem to read from
    int k_tile = 0;

    // Initialize barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], size(mma));
        }
    }

    // Ensure barrier init is complete on all CTAs
    cluster_sync();

    // Start async load for all pipes
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            // Set expected Tx byte after each reset/init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    // Define A/B partitioning and C acculators
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tDsA = thr_mma.partition_A(sA);
    Tensor tDsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tDgD = thr_mma.partition_C(gD);

    // Allocate accumulators and clear them
    Tensor tDrAcc = partition_fragment_C(mma, select<0, 1>(cta_tiler));
    clear(tDrAcc);

    Tensor tDrAcc_ep = partition_fragment_C(mma, select<0, 1>(cta_tiler));
    clear(tDrAcc_ep);

    // Allocate fragments
    Tensor tDrA = thr_mma.make_fragment_A(tDsA);
    Tensor tDrB = thr_mma.make_fragment_B(tDsB);

    // Pipelined mainloop

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>(); // TMA writes
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();  // MMA reads

    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
        // Wait for producer to complete
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        // MMAs to cover 1 K_TILE
        warpgroup_arrive();
        gemm(mma, tDrA(_, _, _, read_pipe), tDrB(_, _, _, read_pipe), tDrAcc); // (V, M) x (V, N) => (V, M, N)
        warpgroup_commit_batch();

        // Wait for all MMAs in a K_TILE to complete
        warpgroup_wait<0>();

        // Notify that consumption is done
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if ((warp_idx == 0) && lane_predicate)
        {
            int pipe = write_state.index();
            // Wait for consumer to complete consumption
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            // Set expected tx bytes after each reset/init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    // Epilogue (unpredicated)
    // axpby(alpha, tCrC, beta, tCgC);

    // Make sure all warpgroups have finished mma
    cutlass::arch::NamedBarrier::sync(size(mma), 0);

    using SmemCopyAtomD = Copy_Atom<SM90_U32x4_STSM_N, TC>;

    auto smem_tiled_copy_D = make_tiled_copy_C(SmemCopyAtomD{}, mma);
    auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(threadIdx.x);

    // EVT: Set up callbacks
    constexpr bool RefSrc = true; // Register tensors reference R2S copy src layout
    auto shape_MNKL = append<4>(shape_MNK, _1{});
    auto tile_coord_MNKL = append<4>(cta_coord, _0{});

    Tensor tCrC = partition_fragment_C(mma, select<0, 1>(cta_tiler));
    clear(tCrC);

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M, N));
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(cta_tiler), make_coord(blockIdx.x, blockIdx.y)); // (CTA_M, CTA_N)
    Tensor tRS_cD_mn = smem_thr_copy_D.partition_S(flat_divide(cD_mn, take<0, 2>(cta_tiler)));    // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)

    // Relative coordinate tensors (static)
    Tensor cD = make_counting_tensor(cD_mn.layout());         // (CTA_M, CTA_N)
    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout()); // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner
    // to get the max relative coordinate
    auto residue_cD = make_coord(M, N) - cD_mn(_0{});      // (m, n)
    auto residue_tRS_cD = make_coord(M, N) - tRS_cD(_0{}); // (m, n)

    auto cst_args =
        cutlass::epilogue::fusion::detail::ConsumerStoreArgs(shape_MNKL, cta_tiler, tile_coord_MNKL, mma, take<0, 2>(cta_tiler),
                                                             smem_tiled_copy_D, cD, residue_cD, tRS_cD, residue_tRS_cD, tCrC, threadIdx.x);

    auto cst_callbacks = evt.get_consumer_store_callbacks<RefSrc>(cst_args);

    if (evt.is_C_load_needed())
    {
        copy(tCgC, tCrC);
    }

    // Vectorize fragment view
    constexpr int FragmentSize = 32;
    Tensor tDrAcc_frg = recast<cutlass::Array<float, FragmentSize>>(tDrAcc);
    Tensor tDrAcc_ep_frg = recast<cutlass::Array<float, FragmentSize>>(tDrAcc_ep);

    for (int epi_v = 0; epi_v < size(tDrAcc_frg); ++epi_v)
    {
        tDrAcc_ep_frg(epi_v) = cst_callbacks.visit(tDrAcc_frg(epi_v), epi_v, 0, 0);
    }

    // Include this convert utility
    Tensor tDrAcc_out = convert_type<TC>(tDrAcc_ep);
    copy(tDrAcc_out, tDgD);

    cst_callbacks.end();
}

template <class EVT, class TA, class TB, class TC, class TD, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, TD* D, int ldD,
             typename EVT::Params& epi_params, cudaStream_t stream = 0)
{

    // Define shapes (dynamic)
    auto M = m;
    auto N = n;
    auto K = k;
    auto prob_shape = make_shape(M, N, K);

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);
    auto dD = make_stride(Int<1>{}, ldD);

    // Define CTA tile sizes (static)
    // The MxNxK of the MMA atom needs to divide into that of the operand and
    // accumulator tiles
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};

    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;

    // Define the smem layouts (static)
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // Define the MMA
    auto tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{}, AtomLayoutMNK{});

    // Define the TMAs
    // Create global memory tensor for TMA inspection
    auto mA = make_tensor(A, make_shape(M, K), dA);
    auto mB = make_tensor(B, make_shape(N, K), dB);

    // Create TMA atoms with the desired copy operation on the source and
    // destination
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    // Setup and Launch

    int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
    dim3 block_dim(size(tiled_mma));
    dim3 cluster_dim(1, 1, 1);
    dim3 grid_dim(round_up(size(ceil_div(m, bM)), cluster_dim.x), round_up(size(ceil_div(n, bN)), cluster_dim.y));

    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
        &gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(sA), decltype(tmaA), TB, decltype(sB), decltype(tmaB), TC,
                     decltype(dC), TD, decltype(dD), decltype(tiled_mma), decltype(alpha), decltype(beta), EVT>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Kernel launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr, prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC, D, dD,
                                                               tiled_mma, alpha, beta, epi_params);

    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "Error: Failed at kernel launch" << std::endl;
    }
}

template <class EVT, class TA, class TB, class TC, class TD, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, TD* D, int ldD,
             typename EVT::Params& epi_params, cudaStream_t stream = 0)
{

    // Define shapes (dynamic)
    auto M = m;
    auto N = n;
    auto K = k;
    auto prob_shape = make_shape(M, N, K);

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);
    auto dD = make_stride(Int<1>{}, ldD);

    // Define CTA tile sizes (static)
    // The MxNxK of the MMA atom needs to divide into that of the operand and
    // accumulator tiles
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};

    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;

    // Define the smem layouts (static)
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // Define the MMA
    auto tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}, AtomLayoutMNK{});

    // Define the TMAs
    // Create global memory tensor for TMA inspection
    auto mA = make_tensor(A, make_shape(M, K), dA);
    auto mB = make_tensor(B, make_shape(N, K), dB);

    // Create TMA atoms with the desired copy operation on the source and
    // destination
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    // Setup and Launch

    int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
    dim3 block_dim(size(tiled_mma));
    dim3 cluster_dim(1, 1, 1);
    dim3 grid_dim(round_up(size(ceil_div(m, bM)), cluster_dim.x), round_up(size(ceil_div(n, bN)), cluster_dim.y));

    cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
        &gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(sA), decltype(tmaA), TB, decltype(sB), decltype(tmaB), TC,
                     decltype(dC), TD, decltype(dD), decltype(tiled_mma), decltype(alpha), decltype(beta), EVT>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Kernel launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr, prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC, D, dD,
                                                               tiled_mma, alpha, beta, epi_params);

    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "Error: Failed at kernel launch" << std::endl;
    }
}

template <class EVT, class TA, class TB, class TC, class TD, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC,
          TD* D, int ldD, typename EVT::Params& epi_params, cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T')
    {
        gemm_nt<EVT>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, D, ldD, epi_params, stream);
    }
    else if (transA == 'T' && transB == 'N')
    {
        gemm_tn<EVT>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, D, ldD, epi_params, stream);
    }
    else
    {
        std::cerr << "Error: Do not support for " << transA << transB << std::endl;
    }
}

void cutlass_gemm(char transA, char transB, int m, int n, int k, float alpha, cutlass::half_t const* A, int ldA, cutlass::half_t const* B,
                  int ldB, float beta, float* C, int ldC, float* D, int ldD, cudaStream_t stream = 0)
{

    if (transA == 'N' && transB == 'T')
    {
        using LayoutA = cutlass::layout::ColumnMajor;

        using LayoutB = cutlass::layout::RowMajor;

        using LayoutC = cutlass::layout::ColumnMajor;

        using LayoutD = cutlass::layout::ColumnMajor;

        using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, float, LayoutC>;

        CutlassGemm gemm;

        CutlassGemm::Arguments args({m, n, k}, {A, ldA}, {B, ldB}, {C, ldC}, {D, ldD}, {alpha, beta});

        cutlass::Status status = gemm(args);
        CUTE_CHECK_LAST();

        if (status != cutlass::Status::kSuccess)
        {
            std::cerr << "Error: Failed at kernel launch" << std::endl;
        }
    }
    else if (transA == 'T' && transB == 'N')
    {
        using LayoutA = cutlass::layout::RowMajor;

        using LayoutB = cutlass::layout::ColumnMajor;

        using LayoutC = cutlass::layout::ColumnMajor;

        using LayoutD = cutlass::layout::ColumnMajor;

        using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, float, LayoutC>;

        CutlassGemm gemm;

        CutlassGemm::Arguments args({m, n, k}, {A, ldA}, {B, ldB}, {C, ldC}, {D, ldD}, {alpha, beta});

        cutlass::Status status = gemm(args);
        CUTE_CHECK_LAST();

        if (status != cutlass::Status::kSuccess)
        {
            std::cerr << "Error: Failed at kernel launch" << std::endl;
        }
    }
    else
    {
        std::cerr << "Error: Do not support for " << transA << transB << std::endl;
    }
}

void cublas_gemm(char transA, char transB, int m, int n, int k, float alpha, cutlass::half_t const* A, int ldA, cutlass::half_t const* B,
                 int ldB, float beta, float* C, int ldC, float* D, int ldD, cudaStream_t stream = 0)
{
    cudaMemcpy(D, C, m * n * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    if (transA == 'N' && transB == 'T')
    {
        ldA = m;
        ldB = n;
        ldC = m;
        ldD = m;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, CUDA_R_16F, ldA, B, CUDA_R_16F, ldB, &beta, D, CUDA_R_32F, ldD,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }
    else
    {
        ldA = k;
        ldB = k;
        ldC = m;
        ldD = m;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_16F, ldA, B, CUDA_R_16F, ldB, &beta, D, CUDA_R_32F, ldD,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }
}

template <typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
void run_gemm(char transA, char transB, int m, int n, int k, int seed = 42)
{
    using TA = cutlass::half_t;
    using TB = cutlass::half_t;
    using TC = float;
    using TI = float;

    TI alpha = TI(2.0f);
    TI beta = TI(1.0f);

    float mean = 0.0f;
    float stddev = 2.0f;
    int bits_less_than_one = 0;

    cutlass::HostTensor<TA, LayoutA> A(cutlass::MatrixCoord(m, k));
    cutlass::HostTensor<TB, LayoutB> B(cutlass::MatrixCoord(k, n));
    cutlass::HostTensor<TC, LayoutC> C(cutlass::MatrixCoord(m, n));
    cutlass::HostTensor<TC, LayoutD> D(cutlass::MatrixCoord(m, n));
    cutlass::HostTensor<TC, LayoutD> D_cutlass(cutlass::MatrixCoord(m, n));
    cutlass::HostTensor<TC, LayoutD> D_cublas(cutlass::MatrixCoord(m, n));

    cutlass::reference::device::TensorFillRandomGaussian(A.device_view(), 2 * seed, (TA)mean, (TA)stddev, bits_less_than_one);
    cutlass::reference::device::TensorFillRandomGaussian(B.device_view(), 3 * seed, (TB)mean, (TB)stddev, bits_less_than_one);
    cutlass::reference::device::TensorFillRandomGaussian(C.device_view(), 5 * seed, (TC)mean, (TC)stddev, bits_less_than_one);

    int ldA, ldB, ldC, ldD;
    if (transA == 'N' && transB == 'T')
    {
        ldA = m;
        ldB = n;
        ldC = m;
        ldD = m;
    }
    else if (transA == 'T' && transB == 'N')
    {
        ldA = k;
        ldB = k;
        ldC = m;
        ldD = m;
    }

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using EVTType = float;

    using namespace cutlass::epilogue::fusion;
    using CustomEVT = // alpha * acc + beta * C
        Sm90EVT<Sm90Compute<cutlass::homogeneous_multiply_add, TC, EVTType,
                            RoundStyle>,                                                // beta * C + (alpha * acc)
                Sm90ScalarBroadcast<EVTType>,                                           // beta
                Sm90SrcFetch<EVTType>,                                                  // C
                Sm90EVT<Sm90Compute<cutlass::multiplies, EVTType, EVTType, RoundStyle>, // alpha * acc
                        Sm90ScalarBroadcast<EVTType>,                                   // alpha
                        Sm90AccFetch                                                    // acc
                        >>;

    CustomEVT::Arguments epi_args{
        // ternary op : beta * C + (alpha * acc)
        {{EVTType(beta)}}, // leaf op+args : beta
        {},                // leaf op+args : C
        {
            // binary op : alpha * acc
            {{EVTType(alpha)}}, // leaf op+args : alpha
            {},                 // leaf op+args : acc
            {}                  // binary args : multiplies
        },                      // end binary op
        {}                      // ternary args : multiply_add
    }; // end

    typename CustomEVT::Params epi_params = CustomEVT::to_underlying_arguments(make_shape(m, n, k), epi_args, nullptr);

    double gflops = (2.0 * m * n * k) * 1e-9;
    const int n_iters = 1;
    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < n_iters; ++i)
    {
        gemm<CustomEVT>(transA, transB, m, n, k, alpha, A.device_data(), ldA, B.device_data(), ldB, beta, C.device_data(), ldC,
                        D.device_data(), ldD, epi_params);
    }

    CUTE_CHECK_LAST();

    double process_time = timer.seconds() / n_iters;

    std::cout << "CUTE GEMM: " << gflops / process_time << " gflop/s, " << process_time * 1000 << " ms" << std::endl;

    cutlass_gemm(transA, transB, m, n, k, alpha, A.device_data(), ldA, B.device_data(), ldB, beta, C.device_data(), ldC,
                 D_cutlass.device_data(), ldD);

    cublas_gemm(transA, transB, m, n, k, alpha, A.device_data(), ldA, B.device_data(), ldB, beta, C.device_data(), ldC,
                D_cublas.device_data(), ldD);

    D.sync_host();
    D_cutlass.sync_host();
    D_cublas.sync_host();

    if (cutlass::reference::host::TensorRelativelyEquals(D.host_view(), D_cutlass.host_view(), TC(1e-2), TC(1e-8)))
    {
        std::cout << "Success cutlass" << std::endl;
    }
    else
    {
        std::cout << "Fail cutlass" << std::endl;
    }

    if (cutlass::reference::host::TensorRelativelyEquals(D.host_view(), D_cublas.host_view(), TC(1e-2), TC(1e-8)))
    {
        std::cout << "Success cublas" << std::endl;
    }
    else
    {
        std::cout << "Fail cublas" << std::endl;
    }
}

int main(int argc, char** argv)
{
    int m = 512;
    if (argc >= 2)
    {
        sscanf(argv[1], "%d", &m);
    }

    int n = 512;
    if (argc >= 3)
    {
        sscanf(argv[2], "%d", &n);
    }

    int k = 512;
    if (argc >= 4)
    {
        sscanf(argv[3], "%d", &k);
    }

    char transA = 'T';
    if (argc >= 5)
    {
        sscanf(argv[4], "%c", &transA);
    }

    char transB = 'N';
    if (argc >= 6)
    {
        sscanf(argv[5], "%c", &transB);
    }

    int seed = 42;
    if (argc >= 7)
    {
        sscanf(argv[6], "%d", &seed);
    }

    if (transA == 'N' && transB == 'T')
    {
        run_gemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(
            transA, transB, m, n, k, seed);
    }
    else if (transA == 'T' && transB == 'N')
    {
        run_gemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(
            transA, transB, m, n, k, seed);
    }
    else
    {
        std::cerr << "Error: Do not support for " << transA << transB << std::endl;
    }

    return 0;
}