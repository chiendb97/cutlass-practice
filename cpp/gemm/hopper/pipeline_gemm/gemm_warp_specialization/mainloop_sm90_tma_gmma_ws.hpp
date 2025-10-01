#pragma once

#include "cute/tensor.hpp"

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace zai
{
using namespace cute;

template <typename Ktraits>
struct CollectiveMainloop
{
    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;
    using BarrierType = typename Ktraits::BarrierType;

    static constexpr int kStages = Ktraits::kStages;

    using SmemLayoutA = typename Ktraits::SmemLayoutA;
    using SmemLayoutB = typename Ktraits::SmemLayoutB;

    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Stride<int32_t, _1>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using TMA_A =
        decltype(make_tma_copy(cute::SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}),
                               take<0, 2>(SmemLayoutA{}), select<0, 2>(TileShape_MNK{}), _1{}));

    using TMA_B =
        decltype(make_tma_copy(cute::SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}),
                               take<0, 2>(SmemLayoutB{}), select<1, 2>(TileShape_MNK{}), _1{}));

    static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    static constexpr uint32_t TmaTransactionBytesA =
        static_cast<uint32_t>(size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesB =
        static_cast<uint32_t>(size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<Element> / 8);

    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;

    // Host side kernel arguments
    struct Arguments
    {
        Element const* ptr_a;
        LayoutT layout_a;
        Element const* ptr_b;
        LayoutT layout_b;
    };

    // Device side kernel params
    struct Params
    {
        LayoutT layout_a;
        LayoutT layout_b;
        TMA_A tma_load_a;
        TMA_B tma_load_b;
    };

    static Params to_underlying_arguments(Arguments const& args)
    {
        Tensor mA = make_tensor(make_gmem_ptr(args.ptr_a), args.layout_a);
        Tensor mB = make_tensor(make_gmem_ptr(args.ptr_b), args.layout_b);

        TMA_A tma_load_a = make_tma_copy(cute::SM90_TMA_LOAD{}, mA, SmemLayoutA{}(_, _, _0{}), select<0, 2>(TileShape_MNK{}), _1{});
        TMA_B tma_load_b = make_tma_copy(cute::SM90_TMA_LOAD{}, mB, SmemLayoutB{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});

        return {args.layout_a, args.layout_b, tma_load_a, tma_load_b};
    }

    // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    // __forceinline__ __device__
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params)
    {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    }

    template <typename Scheduler, typename SharedStorage>
    // __forceinline__ __device__
    CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState& smem_pipe_write,
                             SharedStorage& shared_storage, Scheduler& scheduler, typename Scheduler::Params const& scheduler_params,
                             typename Scheduler::WorkTileInfo& work_tile_info, cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
                             int work_idx, int k_tile_count)
    {
        auto [m_block, n_block, bid_block] = block_coord;

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()), SmemLayoutB{});

        Tensor mA = mainloop_params.tma_load_a.get_tma_tensor(mainloop_params.layout_a.shape());
        Tensor mB = mainloop_params.tma_load_b.get_tma_tensor(mainloop_params.layout_b.shape());

        Tensor gA = local_tile(mA, select<0, 2>(TileShape_MNK{}), make_coord(blockIdx.x, _)); // (bM, bK, k)
        Tensor gB = local_tile(mB, select<1, 2>(TileShape_MNK{}), make_coord(blockIdx.y, _)); // (bN, bK, k)

        // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
        // Tensor gA = local_tile(mA, TileShape_MNK{}, cta_coord, Step<_1, X, _1>{});
        // Tensor gB = local_tile(mB, TileShape_MNK{}, cta_coord, Step<X, _1, _1>{});

        auto [tAgA, tAsA] = tma_partition(mainloop_params.tma_load_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                          group_modes<0, 2>(gA)); // (TMA, k), (TMA, PIPE)
        auto [tBgB, tBsB] = tma_partition(mainloop_params.tma_load_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                          group_modes<0, 2>(gB)); // (TMA, k), (TMA, PIPE)

        // VERSION 1: no prologue-epilogue overlapping
        int lane_pridicate = cute::elect_one_sync();
        if (lane_pridicate)
        {
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
            {
                pipeline.producer_acquire(smem_pipe_write);
                BarrierType* tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
                auto stage = smem_pipe_write.index();
                copy(mainloop_params.tma_load_a.with(*tmaBar, 0), tAgA(_, k_tile), tAsA(_, stage));
                copy(mainloop_params.tma_load_b.with(*tmaBar, 0), tBgB(_, k_tile), tBsB(_, stage));
                ++smem_pipe_write;
            }
        }

        // TODO VERSION 2: overlap prologue B load with epilogue
        // DO first TMA load A.
        // shared_storage.barrier_C.wait((work_idx + 1) % 2);
        // In mainloop, DO TMA load B then next TMA load A.
        // scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        // In last iteration, DO final TMA load B.
        // scheduler.broadcast_next_work(work_tile_info);
    }

    // Perform a producer epilogue to prevent early exit of blocks in a cluster
    CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState& smem_pipe_write)
    {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        if (warp_idx_in_warpgroup == 0 && lane_predicate)
        {
            /* This helps avoid early exit of blocks in Cluster
             * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
             * then would just be acquired since the phase was still inverted from make_producer_start_state
             */
            pipeline.producer_tail(smem_pipe_write);
        }
    }

    template <typename SharedStorage, typename FrgTensorC>
    CUTLASS_DEVICE void mma(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState& smem_pipe_read,
                            SharedStorage& shared_storage, FrgTensorC& tCrC, int thread_idx, int work_idx, int k_tile_count)
    {
        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()), SmemLayoutB{});

        typename Ktraits::TiledMMA tiled_mma;

        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

        Tensor tCsA = thr_mma.partition_A(sA); // MMA, MMA_M, MMA_K, PIPE
        Tensor tCsB = thr_mma.partition_B(sB); // MMA, MMA_N, MMA_K, PIPE

        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        // TODO VERSION 2
        // call arrive on barrier_C for prologue-epilogue overlapping
        // if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
        //     tma_store_wait<0>();
        //     shared_storage.barrier_C.arrive(0, lane_predicate);
        // }
        // int k_tile = k_tile_count-1;

        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile)
        {
            pipeline.consumer_wait(smem_pipe_read);
            auto stage = smem_pipe_read.index();
            warpgroup_arrive();
            gemm(tiled_mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrC);
            warpgroup_commit_batch();

            warpgroup_wait<0>();

            // Release stage of the pipeline for TMA
            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
        }

        // Make sure all warpgroups have finished mma
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, 0);
    }
};
} // namespace zai
