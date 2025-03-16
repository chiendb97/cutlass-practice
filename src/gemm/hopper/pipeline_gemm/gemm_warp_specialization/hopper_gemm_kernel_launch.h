#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "epilogue_sm90_tma_ws.hpp"
#include "hopper_gemm_kernel.h"
#include "kernel_traits.h"
#include "mainloop_sm90_tma_gmma_ws.hpp"
#include "tile_scheduler.hpp"

using namespace cute;

template <bool validation = false, class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int M, int N, int K, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC,
             cudaStream_t stream = 0)
{
    // LEGEND:
    // kBlockM, kBlockN, kBlockK, kNWarps, kStages, ClusterM, ClusterN
    // operand dtype, output dtype, use fp32 accum

    // fp16 accum optimal sizes
    using Kernel_traits =
        std::conditional_t<!validation, Kernel_traits<256, 256, 128 - 32, 20, 2, /*ClusterM=*/1, /*ClusterN=*/2, TA, TC, false>,
                           // Kernel_traits<128 + 64, 256, 128, 12 + 4, 2, 1, 1, TA, TC, false>,
                           Kernel_traits<128, 256, 128, 12, 2, 1, 1, TA, TC, false> // validation params
                           >;

    // fp32 accum optimal sizes
    // using Kernel_traits = std::conditional_t<!validation, Kernel_traits<256, 192, 128, 12, 2, /*ClusterM=*/1, /*ClusterN=*/2, TA, TC>,
    //                                          Kernel_traits<128, 256, 128, 12, 2, 2, 1, TA, TC>
    //                                          // Kernel_traits<128, 256, 64, 12, 4, 2, 1, TA, TC>,
    //                                          // Kernel_traits<128, 256, 96, 12, 3, 2, 1, TA, TC>,
    //                                          // Kernel_traits<192, 192, 128, 16, 2, 1, 2, TA, TC>,
    //                                          >;

    // std::cout << "Num threads = " << Kernel_traits::kNThreads << std::endl;
    // std::cout << "Num Mma threads = " << Kernel_traits::NumMmaThreads << std::endl;
    // using TiledMMA = typename Kernel_traits::TiledMMA;
    // std::cout << "Size of tiled mma = " << size(TiledMMA{}) << std::endl;

    // auto smem_layout_a = typename Kernel_traits::SmemLayoutA{};
    // auto smem_layout_b = typename Kernel_traits::SmemLayoutB{};
    // auto smem_layout_c = typename Kernel_traits::SmemLayoutC{};
    // print("Smem Layout A: ");
    // print(smem_layout_a);
    // print("\n");
    // print("Smem Layout B: ");
    // print(smem_layout_b);
    // print("\n");
    // print("Smem Layout C: ");
    // print(smem_layout_c);
    // print("\n");

    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    using CollectiveMainloop = zai::CollectiveMainloop<Kernel_traits>;
    using CollectiveEpilogue = zai::CollectiveEpilogue<Kernel_traits>;
    // using Scheduler = zai::StaticPersistentTileScheduler;
    using Scheduler = zai::SingleTileScheduler;

    typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments({
        A, make_layout(make_shape(M, K), make_stride(ldA, Int<1>{})), // layout_A
        B, make_layout(make_shape(N, K), make_stride(ldB, Int<1>{})), // layout_B
    });

    // auto layout_a = mainloop_params.layout_a;
    // print(layout_a);
    // auto tma_a = mainloop_params.tma_load_a;
    // print(tma_a);
    // auto layout_b = mainloop_params.layout_b;
    // print(layout_b);
    // auto tma_b = mainloop_params.tma_load_b;
    // print(tma_b);

    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({C, make_layout(make_shape(M, N), make_stride(ldC, Int<1>{}))});

    // auto layout_c = epilogue_params.layout_c;
    // print(layout_c);
    // auto tma_store = epilogue_params.tma_store;
    // print(tma_store);

    int m_block = cutlass::ceil_div(M, Kernel_traits::kBlockM);
    int n_block = cutlass::ceil_div(N, Kernel_traits::kBlockN);
    // round if using clusters
    m_block = cutlass::ceil_div(m_block, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    n_block = cutlass::ceil_div(n_block, size<1>(ClusterShape{})) * size<1>(ClusterShape{});

    typename Scheduler::Arguments scheduler_args = {m_block, n_block, 1};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // Get the ptr to kernel function.
    void* kernel;
    kernel = (void*)zai::hopper_gemm_ws<Kernel_traits, Scheduler>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_a = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_a));
    // int smem_size_b = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_b));
    // int smem_size_c = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_c));
    // printf("smem_size = %d, A = %d, B = %d, C = %d.\n", smem_size, smem_size_a, smem_size_b, smem_size_c);
    if (smem_size >= 48 * 1024)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    // std::cout << grid_dims.x << " " << grid_dims.y << " " << grid_dims.z << std::endl;
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};

#if 1
    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, epilogue_params, scheduler_params);
#endif
}
