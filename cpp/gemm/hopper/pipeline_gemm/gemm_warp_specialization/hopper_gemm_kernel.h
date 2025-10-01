#pragma once

#include "cute/tensor.hpp"

#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "epilogue_sm90_tma_ws.hpp"
#include "kernel_traits.h"
#include "mainloop_sm90_tma_gmma_ws.hpp"
#include "tile_scheduler.hpp"

namespace zai
{
using namespace cute;
template <typename Ktraits, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNThreads, 1)
    hopper_gemm_ws(CUTE_GRID_CONSTANT typename CollectiveMainloop<Ktraits>::Params const mainloop_params,
                   CUTE_GRID_CONSTANT typename CollectiveEpilogue<Ktraits>::Params const epilogue_params,
                   CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params)
{
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMMA{});
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;

    using CollectiveMainloop = CollectiveMainloop<Ktraits>;
    using CollectiveEpilogue = CollectiveEpilogue<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];

    auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    if (warp_idx == 0 && lane_predicate)
    {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;

    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate)
    {
        shared_storage.barrier_c.init(size(ClusterShape{}));
    }

    MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    const int k_tile_count = cutlass::ceil_div(shape<1>(mainloop_params.layout_a), Ktraits::kBlockK);

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1)
    {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    }
    else
    {
        __syncthreads();
    }

    static_assert(Ktraits::kNWarps == 8 || Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16 || Ktraits::kNWarps == 20);
    if (warp_group_idx == 0)
    {
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 16 ? 32 : 24>(); // why ?

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0)
        {
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            TileScheduler scheduler(&shared_storage.tile_count_semaphore);

            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.template get_next_work<true>(scheduler_params, work_tile_info))
            {
                auto block_coord = work_tile_info.get_block_coord(scheduler_params);
                collective_mainloop.load(mainloop_params, pipeline, smem_pipe_write, shared_storage, scheduler, scheduler_params,
                                         work_tile_info, block_coord, work_idx, k_tile_count);
                ++work_idx;
            }
            collective_mainloop.load_tail(pipeline, smem_pipe_write);
        }
    }
    else
    {
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 8    ? 256
                                           : Ktraits::kNWarps == 12 ? 240
                                           : Ktraits::kNWarps == 16 ? 160
                                                                    : 112>(); // why ?

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);

        typename Ktraits::TiledMMA tiled_mma;

        PipelineState smem_pipe_read;

        int work_idx = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work<false>(scheduler_params, work_tile_info))
        {
            Tensor tCrC = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            clear(tCrC);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);

            collective_mainloop.mma(mainloop_params, pipeline, smem_pipe_read, shared_storage, tCrC, threadIdx.x - NumCopyThreads, work_idx,
                                    k_tile_count);
            collective_epilogue.store(epilogue_params, tCrC, shared_storage, tiled_mma, threadIdx.x - NumCopyThreads, block_coord);
            ++work_idx;
        }
        collective_epilogue.store_tail();
    }
}
} // namespace zai