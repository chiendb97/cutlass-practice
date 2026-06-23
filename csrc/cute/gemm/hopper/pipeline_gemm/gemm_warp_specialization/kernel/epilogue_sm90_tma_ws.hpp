#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "convert_util.h"

namespace zai
{
using namespace cute;
template <typename Ktraits>
struct CollectiveEpilogue
{
    using Element = typename Ktraits::OutputType;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;

    static constexpr int kNWarps = Ktraits::kNWarps;
    static constexpr int kNThreads = Ktraits::kNThreads;

    static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;

    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Stride<int32_t, _1>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using SmemLayoutC = typename Ktraits::SmemLayoutC;

    using SmemCopyAtomC = typename Ktraits::SmemCopyAtomC;

    using TMA_C = decltype(make_tma_copy(cute::SM90_TMA_STORE{},
                                         make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}),
                                         SmemLayoutC{}, select<0, 1>(TileShape_MNK{}), _1{}));

    struct Arguments
    {
        Element* ptr_c;
        LayoutT const layout_c;
    };

    struct Params
    {
        Element* ptr_c;
        LayoutT const layout_c;
        TMA_C tma_store;
    };

    static Params to_underlying_arguments(Arguments const& args)
    {
        Tensor mC = make_tensor(make_gmem_ptr(args.ptr_c), args.layout_c);
        TMA_C tma_store = make_tma_copy(cute::SM90_TMA_STORE{}, mC, SmemLayoutC{}, select<0, 1>(TileShape_MNK{}), _1{});

        return {args.ptr_c, args.layout_c, tma_store};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params)
    {
        cute::prefetch_tma_descriptor(epilogue_params.tma_store.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorC, typename TiledMma>
    CUTLASS_DEVICE void store(Params const& epilogue_params, FrgTensorC const& tCrC, SharedStorage& shared_storage, TiledMma tiled_mma,
                              int thread_idx, cute::tuple<int32_t, int32_t, int32_t> const& block_coord)
    {
        auto [m_block, n_block, bid_block] = block_coord;

        Tensor sC = make_tensor(make_smem_ptr(shared_storage.smem_c.data()), SmemLayoutC{});
        auto smem_tiled_copy_c = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
        auto smem_thr_copy_c = smem_tiled_copy_c.get_thread_slice(thread_idx);

        Tensor tCrC_out = convert_type<Element>(tCrC);
        Tensor taccCrC = smem_thr_copy_c.retile_S(tCrC_out);
        Tensor taccCsC = smem_thr_copy_c.partition_D(sC);

        copy(smem_tiled_copy_c, taccCrC, taccCsC);

        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        Tensor mC = epilogue_params.tma_store.get_tma_tensor(epilogue_params.layout_c.shape());
        Tensor gC = local_tile(mC, select<0, 1>(TileShape_MNK{}), make_coord(m_block, n_block));

        auto block_tma_store = epilogue_params.tma_store.get_slice(_0{});
        Tensor tCgC = block_tma_store.partition_D(gC);
        Tensor tCsC = block_tma_store.partition_S(sC);

        // TMA store: smem -> gmem
        int write_warp_idx = kNWarps - 1;
        int const warp_idx = cutlass::canonical_warp_idx_sync();
        int const lane_predicate = cute::elect_one_sync();

        if (warp_idx == write_warp_idx)
        {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                              cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        }

        if (warp_idx == write_warp_idx && lane_predicate)
        {
            cute::copy(epilogue_params.tma_store, tCsC, tCgC);
            tma_store_arrive();
        }
        // TODO: overlap epilogue with next CTA load in persistent kernel
        // tma_store_wait<0>();
    }

    CUTLASS_DEVICE void store_tail() { tma_store_wait<0>(); }
};
} // namespace zai
