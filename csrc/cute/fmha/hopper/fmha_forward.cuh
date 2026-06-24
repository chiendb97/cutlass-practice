#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "copy_tensor.hpp"
#include "gemm_tensor.hpp"
#include "online_softmax.h"
namespace cute_fmha
{

#ifdef OBLKSIZE
constexpr int kQueriesPerBlock = OBLKSIZE;
#else
constexpr int kQueriesPerBlock = 64;
#endif

#ifdef KBLKSIZE
constexpr int kKeysPerBlock = KBLKSIZE;
#else
constexpr int kKeysPerBlock = 64;
#endif

// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutS, class SmemLayoutV>
struct SharedStorage
{
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smemQ;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smemK;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smemV;
#ifdef SINSMEM
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutS>> smemS;
#endif
    cute::uint64_t tma_load_mbar[8];
};

// Reshape Utility for converting the layout from accumulator of GEMM-I to Operand A of GEMM-II.
struct ReshapeTStoTP
{
    template <class FragmentC, class FragmentQ>
    __device__ auto operator()(FragmentC& tC, FragmentQ& tQ)
    {
        // Get the layout of one row of Q.
        auto layoutQRow = cute::make_ordered_layout(tQ(cute::_, 0, cute::_).layout().shape(), tQ(cute::_, 0, cute::_).layout().stride());
        // Get the layout of M dimension of C.
        auto layoutCM = cute::get<1>(tC.layout());
        return cute::make_layout(cute::get<0>(layoutQRow), layoutCM, cute::get<1>(layoutQRow));
    }
};

// Conversion Utility to convert RMEM from one type to another.
// Used for conversion from AccumType to PrecType.
template <typename To_type, typename From_type, typename Fragment>
CUTLASS_DEVICE auto convert_type(Fragment const& tensor)
{
    using namespace cute;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

void set_smem_size(int smem_size, void const* kernel)
{
    // account for dynamic smem capacity if needed
    if (smem_size >= (48 << 10))
    {
        cudaError_t result = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (cudaSuccess != result)
        {
            result = cudaGetLastError(); // to clear the error bit
            std::cout << "Shared Memory Allocation Failed " << std::endl
                      << " cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result) << std::endl;
        }
    }
}

template <class PrecType, class AccumType, class TiledMma0, class TiledMma1, class TiledCopyQ, class TileShapeQ, class SmemLayoutQ,
          class GmemLayoutQ, class TiledCopyK, class TileShapeK, class SmemLayoutK, class GmemLayoutK, class TileShapeS, class SmemLayoutS,
          class GmemLayoutS, class TiledCopyV, class TileShapeV, class SmemLayoutV, class GmemLayoutV, class SmemLayoutVt, class TiledCopyO,
          class TileShapeO, class GmemLayoutO, class GmemLayoutMi, class ClusterShape>
CUTLASS_GLOBAL void fmhaForward(PrecType const* Q, CUTE_GRID_CONSTANT TiledCopyQ const tmaLoadQ, TileShapeQ tileShapeQ,
                                SmemLayoutQ smemLayoutQ, GmemLayoutQ gmemLayoutQ, PrecType const* K,
                                CUTE_GRID_CONSTANT TiledCopyK const tmaLoadK, TileShapeK tileShapeK, SmemLayoutK smemLayoutK,
                                GmemLayoutK gmemLayoutK, PrecType* S, TileShapeS tileShapeS, SmemLayoutS smemLayoutS,
                                GmemLayoutS gmemLayoutS, int nTilesOfK, PrecType* V, CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV,
                                TileShapeV tileShapeV, SmemLayoutV smemLayoutV, GmemLayoutV gmemLayoutV, SmemLayoutVt smemLayoutVt,
                                PrecType* O, CUTE_GRID_CONSTANT TiledCopyO const tmaStoreO, TileShapeO tileShapeO, GmemLayoutO gmemLayoutO,
                                AccumType* mi, AccumType* sPrime, GmemLayoutMi gmemLayoutMi, float scale)
{
    using namespace cute;
    extern __shared__ char shared_memory[];

    using SharedStorage = SharedStorage<PrecType, SmemLayoutQ, SmemLayoutK, SmemLayoutS, SmemLayoutV>;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

    auto blockIdxX = uint64_t(blockIdx.x);
    auto blockIdxH = uint64_t(blockIdx.y);
    auto blockIdxB = uint64_t(blockIdx.z);

    auto sQ = make_tensor(make_smem_ptr(shared_storage.smemQ.data()), smemLayoutQ);
    auto sK = make_tensor(make_smem_ptr(shared_storage.smemK.data()), smemLayoutK);
#ifdef SINSMEM
    auto sS = make_tensor(make_smem_ptr(shared_storage.smemS.data()), smemLayoutS);
#else
    auto sS = make_tensor(make_smem_ptr(shared_storage.smemV.data()), smemLayoutS);
#endif
    auto sV = make_tensor(make_smem_ptr(shared_storage.smemV.data()), smemLayoutV);

    // Tensor Vtranspose; used in GEMM-II.
    auto sVt = make_tensor(make_smem_ptr(shared_storage.smemV.data()), smemLayoutVt);

    // Get the full un-partitioned tensors.
    auto mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));
    auto mK = tmaLoadK.get_tma_tensor(shape(gmemLayoutK));
    auto mV = tmaLoadV.get_tma_tensor(shape(gmemLayoutV));
    auto mO = tmaStoreO.get_tma_tensor(shape(gmemLayoutO));

    TiledMma0 tiledMma0;
    auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);

    TiledMma1 tiledMma1;
    auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

    auto block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr int32_t cluster_shape_x = get<0>(ClusterShape{});
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    uint16_t mcast_mask_a = 0;
    auto block_layout = Layout<ClusterShape>{}; // (m, n) -> block_id
    for (int n = 0; n < size(block_layout); ++n)
    {
        mcast_mask_a |= (uint16_t(1) << block_layout(n, 0, Int<0>{}));
    }

    // Prepare TMA_LOADS and TMA_STORE.
    auto cta_tmaQ = tmaLoadQ.get_slice(0);
    auto cta_tmaK = tmaLoadK.get_slice(cluster_local_block_id.x);
    auto cta_tmaV = tmaLoadV.get_slice(cluster_local_block_id.x);
    auto cta_tmaO = tmaStoreO.get_slice(0);

    // Get the block of Q for this CTA using the block coordinates.
    auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
    auto gQ = local_tile(mQ, tileShapeQ, blkCoordQ); // (bM, bN)

    // Partition the copying of source tiles for Q ammong threads.
    auto tQgQX = cta_tmaQ.partition_S(gQ);
    auto tQgQ = group_modes<1, rank(tQgQX)>(tQgQX); // (TMA, REST)

    auto kTiles = size<1>(tQgQ);
    assert(kTiles == 1);
    // assert(kTiles == size<2>(gQ));

    // Partition the copying of dest tiles for Q, K and V among threads.
    auto tQsQX = cta_tmaQ.partition_D(sQ);
    auto tQsQ = group_modes<1, rank(tQsQX)>(tQsQX);

    auto tKsKX = cta_tmaK.partition_D(sK);
    auto tKsK = group_modes<1, rank(tKsKX)>(tKsKX);

    auto tVsVX = cta_tmaV.partition_D(sV);
    auto tVsV = group_modes<1, rank(tVsVX)>(tVsVX);

    static_assert(size<1>(tQsQ) == 1);
    static_assert(size<1>(tKsK) == 1);

    // Allocate "fragments/descriptors" for GEMM-I.
    auto tSrQ = threadMma0.partition_fragment_A(sQ);
    auto tSrK = threadMma0.partition_fragment_B(sK);
    auto tSrS = partition_fragment_C(tiledMma0, tileShapeS);
    clear(tSrS);

    // Allocate "fragments/descriptors" for GEMM-II.
    auto tOrV = threadMma1.partition_fragment_B(sVt);
    auto tOrO = partition_fragment_C(tiledMma1, tileShapeO);
    clear(tOrO);

// Use this flag to store rsult of GEMM-I to SMEM, GEMM-II will also read from SMEM. By default, this flag is disable.
#ifdef SINSMEM
    auto tSsS = threadMma0.partition_C(sS);
    cute::fill(tSsS, PrecType(0.0));
    auto tOrP = threadMma1.partition_fragment_A(sS);
#else
    auto tOrS = threadMma1.partition_fragment_A(sS);
    auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);
    auto tOrP = make_tensor(tSrS.data(), tOrPLayout);
#endif

    // Allocate space for per-thread rowMax and rowSum in rmem.
    auto rowMax = make_tensor<AccumType>(Shape<Int<2 * size<1>(tSrS)>>{});
    auto rowSum = make_fragment_like(rowMax);

    cute::fill(rowMax, -cutlass::platform::numeric_limits<AccumType>::infinity());
    cute::fill(rowSum, AccumType(0.0));

    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    // Barrier 0-2 are transaction-bytes based. numthreads = 1 for them.
    // Barrier 0 is used for K copy, 1 for V copy, and 2 for Q copy.
    zai::barrierInit(tma_load_mbar[0], 1);
    zai::barrierInit(tma_load_mbar[1], 1);
    zai::barrierInit(tma_load_mbar[2], 1);

    // Copy Q tile from GMEM to SMEM.
    zai::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[2]);
    cute::wait_barrier(tma_load_mbar[2], 0); // This is required.

    // Prepare Tensors for first K copy on GMEM side.
    auto blkCoordK = make_coord(0, 0, blockIdxH, blockIdxB);
    auto gK = local_tile(mK, tileShapeK, blkCoordK);
    auto tKgKX = cta_tmaK.partition_S(gK);
    auto tKgK = group_modes<1, rank(tKgKX)>(tKgKX); // (TMA, REST)
    // assert(size<1>(tKgK) == size<2>(gK));
    assert(size<1>(tKgK) == kTiles);
    static_assert(size<1>(tKsK) == 1);

    // Copy first tile of K from GMEM to SMEM.
    zai::copy(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0], mcast_mask_a);

    // Initialize phase for barriers 0 and 1.
    int phase = 0;

#pragma unroll
    for (uint64_t blockIdxY = 0; blockIdxY < nTilesOfK; ++blockIdxY)
    {
        // Prepare Tensors for V copy on GMEM side.
        auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
        auto gV = local_tile(mV, tileShapeV, blkCoordV);
        auto tVgVX = cta_tmaV.partition_S(gV);
        auto tVgV = group_modes<1, rank(tVgVX)>(tVgVX); // (TMA, REST)
        // assert(size<1>(tVgV) == size<2>(gV));
        assert(size<1>(tVgV) == 1);

        // Copy current tile of V from GMEM to SMEM
        zai::syncCluster<ClusterShape>();
        zai::copy(tVgV(_, 0), tVsV(_, 0), tmaLoadV, tma_load_mbar[1], mcast_mask_a);
        clear(tSrS);

        // Compute GEMM-I.
        zai::gemm_ldbar(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0], phase);

// Required for verification ONLY.
#ifdef COPYOUTMM0
        auto mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
        auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
        auto gS = local_tile(mS, tileShapeS, blkCoordS);
        auto tSgS = threadMma0.partition_C(gS);
        copy(tSrS, tSgS);
#endif

        // Copy next tile of K from GMEM to SMEM.
        if (blockIdxY != (nTilesOfK - 1))
        {
            auto blkCoordK = make_coord(blockIdxY + 1, 0, blockIdxH, blockIdxB);
            auto gK = local_tile(mK, tileShapeK, blkCoordK);
            auto tKgKX = cta_tmaK.partition_S(gK);
            auto tKgK = group_modes<1, rank(tKgKX)>(tKgKX); // (TMA, REST)

            zai::syncCluster<ClusterShape>();
            zai::copy(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0], mcast_mask_a);
        }

        if (blockIdxY == 0) // Compute Online Softmax and NO Output Rescaling.
        {
            onlineSoftmaxAndRescale<true, AccumType>(rowMax, rowSum, tSrS, tOrO, scale);
        }
        else // Compute Online Softmax and Output Rescaling.
        {
            onlineSoftmaxAndRescale<false, AccumType>(rowMax, rowSum, tSrS, tOrO, scale);
        }

        // onlineSoftmaxAndRescale<blockIdxY == 0, AccumType>(rowMax, rowSum, tSrS, tOrO, scale);
        warpgroup_fence_operand(tSrS);

#ifdef SINSMEM
        // Compute GEMM-II with operandA from SMEM
        // Copy operandA from RMEM to SMEM before compute
        zai::copy(tSrS, tSsS);
        zai::gemm_ldbar(tiledMma1, tOrP, tOrV, tOrO, tma_load_mbar[1], phase);
#else
        // Compute GEMM-II with operandA from RMEM.
        // Convert operandA from AccumType(float) to PrecType(half_t) before compute.
        zai::gemm_ldbar(tiledMma1, convert_type<PrecType, AccumType>(tOrP), tOrV, tOrO, tma_load_mbar[1], phase);
#endif
        // Flip phase for barrier.
        phase = (phase + 1) % 2;
    }

    // Apply softmax normalizer before writing out to GMEM.
    applySoftmaxNormalizer<AccumType>(rowSum, tOrO);

    // Copy output tile O from RMEM to SMEM, overwriting sQ.
    auto tOsOAcc = threadMma1.partition_C(sQ);

    zai::copy(tOrO, tOsOAcc);

    // Prepare Tensors for writing out O on GMEM side.
    auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
    Tensor gO = local_tile(mO, tileShapeO, blkCoordO);
    Tensor tOgOX = cta_tmaO.partition_D(gO);
    Tensor tOgO = group_modes<1, rank(tOgOX)>(tOgOX);
    assert(size<1>(tOgO) == 1);

    // Partition the copying of SMEM tile for O among threads.
    Tensor tOsOX = cta_tmaO.partition_S(sQ);
    Tensor tOsO = group_modes<1, rank(tOsOX)>(tOsOX);
    static_assert(size<1>(tOsO) == 1);

    // Issue the TMA store.
    if (warp_idx == 0 and lane_predicate)
    {
        cute::copy(tmaStoreO, tOsO, tOgO);
    }

    // Wait for TMA store to complete.
    tma_store_wait<0>();

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
#ifdef CTA256
    const int NumMmaWarpGroups = 2;
#else
    const int NumMmaWarpGroups = 1;
#endif
    Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
    Tensor miGlobalOut = local_tile(miGlobal, make_shape(get<0>(tileShapeQ), 1, 1), make_coord(blockIdxX, blockIdxH, blockIdxB));
    Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
    Tensor sPrimeGlobalOut = local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeQ), 1, 1), make_coord(blockIdxX, blockIdxH, blockIdxB));
    if (threadIdx.x % 4 == 0)
    {
        auto mmaThreadLayoutC = TiledMma0{}.get_layoutC_TV();
        auto mmaShapeMNK = cute::tile_shape(TiledMma0{});
        auto mmaShapeMN = make_shape(shape<0>(mmaShapeMNK), shape<1>(mmaShapeMNK));
        auto flatCoord = cute::idx2crd(mmaThreadLayoutC(threadIdx.x, 0), mmaShapeMN);
        auto rowIdGlobal = get<0>(flatCoord); // starting rowId
        auto rowId = 0;
        for (int i = rowIdGlobal; i < kQueriesPerBlock; i += NumMmaWarpGroups * 64)
        {
            miGlobalOut(i) = rowMax(rowId);
            sPrimeGlobalOut(i) = rowSum(rowId);
            miGlobalOut(i + 8) = rowMax(rowId + 1);
            sPrimeGlobalOut(i + 8) = rowSum(rowId + 1);
            rowId += 2;
        }
    }
#endif

    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
    __syncthreads();
}

template <typename PrecType, typename AccumType, int HEADDIM>
void fmhaForwardDevice(int seqLen, int keyLen, int numHeads, int batchSize, PrecType const* tensorQ, PrecType const* tensorK,
                       PrecType* tensorV, PrecType* tensorS, PrecType* tensorO, AccumType* miOut, AccumType* sPrimeOut,
                       float scale, cudaStream_t stream = 0)
{
    using namespace cute;
    auto B = int(batchSize);
    auto H = int(numHeads);
    auto M = int(seqLen);
    auto N = int(keyLen);
    auto K = int(HEADDIM);

    using bM = Int<kQueriesPerBlock>;
    using bN = Int<kKeysPerBlock>;
    using bK = Int<HEADDIM>;

    using MmaA = PrecType;
    using MmaB = PrecType;
    using MmaC = AccumType;

#if defined(CLUSTERN) && CLUSTERN > 1
#define TMA_LOAD SM90_TMA_LOAD_MULTICAST
    using ClusterShape = Shape<Int<CLUSTERN>, _1, _1>;
#else
#define TMA_LOAD SM90_TMA_LOAD
    using ClusterShape = Shape<_1, _1, _1>;
#endif

    // All the tensors are stored in BMHK order, with K being the unit-1 dimension. For now, n=m.
    auto ptrQ = reinterpret_cast<MmaA const*>(tensorQ);
    auto ptrK = reinterpret_cast<MmaB const*>(tensorK);
    auto ptrV = reinterpret_cast<MmaB const*>(tensorV);

    auto tileShapeQ = make_shape(bM{}, bK{});
    auto smemLayoutQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaA>{}, tileShapeQ);
    auto gmemLayoutQ = make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));
    auto gQ = make_tensor(ptrQ, gmemLayoutQ);
    auto tmaQ = make_tma_copy(SM90_TMA_LOAD{}, gQ, smemLayoutQ, tileShapeQ, Int<1>{});

    auto tileShapeK = make_shape(bN{}, bK{});
    auto smemLayoutK = tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaB>{}, tileShapeK);
    auto gmemLayoutK = make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * N * K));
    auto gK = make_tensor(ptrK, gmemLayoutK);
    auto tmaK = make_tma_copy(TMA_LOAD{}, gK, smemLayoutK, tileShapeK, size<0>(ClusterShape{}));

    // Use only during debugging, direct writes to GMEM.
    auto tileShapeS = make_shape(bM{}, bN{});
    auto smemLayoutS = tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaA>{}, tileShapeS);
    auto gmemLayoutS = make_layout(make_shape(M, N, H, B), make_stride(N, 1, N * M, H * N * M));

    auto tileShapeV = make_shape(bN{}, bK{});
    auto smemLayoutV = tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaB>{}, tileShapeV);
    auto gmemLayoutV = make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * N * K));
    auto gV = make_tensor(ptrV, gmemLayoutV);
    auto tmaV = make_tma_copy(TMA_LOAD{}, gV, smemLayoutV, tileShapeV, size<0>(ClusterShape{}));

    // Layout Vtranspose. For use in GEMM-II
    auto tileShapeVt = make_shape(bK{}, bN{});
    auto smemLayoutVt = composition(smemLayoutV, make_layout(tileShapeVt, GenRowMajor{}));

    auto tileShapeO = make_shape(bM{}, bK{});
    auto smemLayoutO = tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaA>{}, tileShapeO);
    auto gmemLayoutO = make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));
    auto gO = make_tensor(tensorO, gmemLayoutO);
    auto tmaO = make_tma_copy(SM90_TMA_STORE{}, gO, smemLayoutO, tileShapeO, Int<1>{});

#ifdef CTA256
    using MmaTileShape = Layout<Shape<_2, _1, _1>>;
#else
    using MmaTileShape = Layout<Shape<_1, _1, _1>>;
#endif

    // Use SS version of GMMA for GEMM-I (Q*Kt): (M, K) * (K, N).
    using TiledMma0 = decltype(make_tiled_mma(GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bN, bK>>(), MmaTileShape{}));

#ifdef SINSMEM
    // Use SS version of GMMA for GEMM-II (O*V): (M, N) * (N, K).
    using TiledMma1 = decltype(make_tiled_mma(GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>, GMMA::Major::K, GMMA::Major::MN>(),
                                              MmaTileShape{}));
#else
    // Use RS version of GMMA for GEMM-II (O*V): (M, N) * (N, K).
    using TiledMma1 = decltype(make_tiled_mma(GMMA::rs_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>, GMMA::Major::K, GMMA::Major::MN>(),
                                              MmaTileShape{}));
#endif

    // Col-major for Mi and sPrime (used only for verification).
    auto gmemLayoutMi = make_layout(make_shape(M, H, B), GenColMajor{});

    void const* kernel =
        (void const*)fmhaForward<PrecType, AccumType, TiledMma0, TiledMma1, decltype(tmaQ), decltype(tileShapeQ), decltype(smemLayoutQ),
                                 decltype(gmemLayoutQ), decltype(tmaK), decltype(tileShapeK), decltype(smemLayoutK), decltype(gmemLayoutK),
                                 decltype(tileShapeS), decltype(smemLayoutS), decltype(gmemLayoutS), decltype(tmaV), decltype(tileShapeV),
                                 decltype(smemLayoutV), decltype(gmemLayoutV), decltype(smemLayoutVt), decltype(tmaO), decltype(tileShapeO),
                                 decltype(gmemLayoutO), decltype(gmemLayoutMi), ClusterShape>;

    auto smem_size =
        int(sizeof(SharedStorage<PrecType, decltype(smemLayoutQ), decltype(smemLayoutK), decltype(smemLayoutS), decltype(smemLayoutV)>));

    set_smem_size(smem_size, kernel);

    //
    // Define CUDA launch kernel parameters.
    //

    // Set the THREAD BLOCK (CTA) dimensions.
    // #threads in CTA = #threads in MMA (128 by default).
    dim3 block_dims(size(TiledMma0{}));

    // Set the GRID dimensions (3-D).
    // First dimension = # of blocks of Q.
    // Second dimension = # of heads.
    // Third dimension = # of batches.
    dim3 grid_dims(ceil_div(size(M), size(bM{})), H, B);

    dim3 cluster_dims(size<0>(ClusterShape{}), 1, 1);

    // Define the cluster launch parameter structure.
    cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims, smem_size, stream};

    auto nTilesOfK = ceil_div(size(N), size(bN{}));

    cutlass::launch_kernel_on_cluster(params, kernel, ptrQ, tmaQ, tileShapeQ, smemLayoutQ, gmemLayoutQ, ptrK, tmaK, tileShapeK, smemLayoutK,
                                      gmemLayoutK, tensorS, tileShapeS, smemLayoutS, gmemLayoutS, nTilesOfK, ptrV, tmaV, tileShapeV,
                                      smemLayoutV, gmemLayoutV, smemLayoutVt, tensorO, tmaO, tileShapeO, gmemLayoutO, miOut, sPrimeOut,
                                      gmemLayoutMi, scale);
}

}  // namespace cute_fmha
