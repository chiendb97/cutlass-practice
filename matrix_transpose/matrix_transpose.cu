//
// Created by chiendb on 9/13/24.
//

#include <cstdio>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/detail/layout.hpp"


template<class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transposeKernelNaive(TensorS const S, TensorD const D,
                     ThreadLayoutS const tS, ThreadLayoutD const tD) {
    auto gS = S(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y);
    auto gD = D(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y);

    auto tSgS = cute::local_partition(gS, tS, threadIdx.x);
    auto tDgD = cute::local_partition(gD, tD, threadIdx.x);

    auto rmem = cute::make_tensor_like(tSgS);
    cute::copy(tSgS, rmem);
    cute::copy(rmem, tDgD);
}

template<typename T>
void transpose_naive(T *S, T *D, int M, int N) {
    auto gmem_layout_S = cute::make_layout(cute::make_shape(M, N), cute::LayoutRight{});
    auto gmem_layout_D = cute::make_layout(cute::make_shape(N, M), cute::LayoutLeft{});

    auto tensor_S = cute::make_tensor(cute::make_gmem_ptr(S), gmem_layout_S);
    auto tensor_D = cute::make_tensor(cute::make_gmem_ptr(D), gmem_layout_D);

    using bM = cute::Int<64>;
    using bN = cute::Int<64>;

    auto block_shape_S = cute::make_shape(bM{}, bN{});
    auto block_shape_D = cute::make_shape(bN{}, bM{});

    auto tiled_tensor_S = cute::tiled_divide(tensor_S, block_shape_S);
    auto tiled_tensor_D = cute::tiled_divide(tensor_D, block_shape_D);

    auto thread_layout_S = cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), cute::LayoutRight{});
    auto thread_layout_D = cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), cute::LayoutRight{});

    dim3 gridDim(cute::size<1>(tiled_tensor_S),
                 cute::size<2>(tiled_tensor_S));
    dim3 blockDim(cute::size(thread_layout_S));
    transposeKernelNaive<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D, thread_layout_S, thread_layout_D);
}

template<class Element, class SmemLayout>
struct SharedStorageTranspose {
    cute::array_aligned<Element, cute::cosize_v<SmemLayout>, cutlass::detail::alignment_for_swizzle(SmemLayout{})> smem;
};

template<class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS, class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transposeKernelSmem(TensorS const S, TensorD const D,
                    SmemLayoutS const smem_layout_S, ThreadLayoutS const tS,
                    SmemLayoutD const smem_layout_D, ThreadLayoutD const tD) {

    using Element = typename TensorS::value_type;
    extern __shared__ char shared_memory[];

    using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    auto sS = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem.data()), smem_layout_S);
    auto sD = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem.data()), smem_layout_D);

    auto gS = S(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y);
    auto gD = D(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y);

    auto tSsS = cute::local_partition(sS, tS, threadIdx.x);
    auto tSgS = cute::local_partition(gS, tS, threadIdx.x);
    auto tDsD = cute::local_partition(sD, tD, threadIdx.x);
    auto tDgD = cute::local_partition(gD, tD, threadIdx.x);

    cute::copy(tSgS, tSsS);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    cute::copy(tDsD, tDgD);
}

template<typename T, bool isSwizzled = true>
void transpose_smem(T *S, T *D, int M, int N) {
    auto gmem_layout_S = cute::make_layout(cute::make_shape(M, N), cute::LayoutRight{});
    auto gmem_layout_D = cute::make_layout(cute::make_shape(N, M), cute::LayoutRight{});

    auto tensor_S = cute::make_tensor(cute::make_gmem_ptr(S), gmem_layout_S);
    auto tensor_D = cute::make_tensor(cute::make_gmem_ptr(D), gmem_layout_D);

    using bM = cute::Int<64>;
    using bN = cute::Int<64>;

    auto block_shape_S = cute::make_shape(bM{}, bN{});
    auto block_shape_D = cute::make_shape(bN{}, bM{});

    auto tiled_tensor_S = cute::tiled_divide(tensor_S, block_shape_S);
    auto tiled_tensor_D = cute::tiled_divide(tensor_D, block_shape_D);

    auto block_layout_S = cute::make_layout(block_shape_S, cute::LayoutRight{});
    auto block_layout_D = cute::make_layout(block_shape_D, cute::LayoutRight{});

    auto smem_layout_S = block_layout_S;
    auto smem_layout_D = cute::composition(smem_layout_S, block_layout_D);

    auto smem_layout_S_swizzle = composition(cute::Swizzle<5, 0, 5>{}, block_layout_S);
    auto smem_layout_D_swizzle = composition(smem_layout_S_swizzle, block_layout_D);

    auto smem_size = int(sizeof(SharedStorageTranspose<T, decltype(smem_layout_S_swizzle)>));

    auto thread_layout_S = cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), cute::LayoutRight{});
    auto thread_layout_D = cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), cute::LayoutRight{});

    dim3 gridDim(cute::size<1>(tiled_tensor_S),
                 cute::size<2>(tiled_tensor_S));
    dim3 blockDim(cute::size(thread_layout_S));

    if constexpr (isSwizzled) {
        transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S, tiled_tensor_D,
                                                              smem_layout_S_swizzle, thread_layout_S,
                                                              smem_layout_D_swizzle, thread_layout_D);
    } else {
        transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S, tiled_tensor_D,
                                                              smem_layout_S, thread_layout_S,
                                                              smem_layout_D, thread_layout_D);
    }
}

template<typename T, bool isTranspose = true>
int
benchmark(void (*transpose)(T *S, T *D, int M, int N), int M, int N, int iterations = 10, bool verify = true) {
    auto tensor_shape_S = cute::make_shape(M, N);
    auto tensor_shape_D = isTranspose ? cute::make_shape(N, M) : cute::make_shape(M, N);
    thrust::host_vector<T> h_S(cute::size(tensor_shape_S));
    thrust::host_vector<T> h_D(cute::size(tensor_shape_D));

    for (auto i = 0; i < h_S.size(); ++i) {
        h_S[i] = static_cast<T>(i);
    }

    thrust::device_vector<T> d_S = h_S;
    thrust::device_vector<T> d_D = h_D;

    for (auto i = 0; i < iterations; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        transpose(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), M, N);
        auto result = cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();

        if (result != cudaSuccess) {
            std::cerr << "Cuda runtime error: " << cudaGetErrorString(result) << std::endl;
        }

        std::chrono::duration<double, std::milli> diff_time = (end_time - start_time);
        auto processed_time = diff_time.count();
        std::cout << "Trial " << i << " Completed in " << processed_time << "ms ("
                  << 2e-6 * M * N * sizeof(T) / processed_time << " GB/s)"
                  << std::endl;
    }
    if (verify) {
        h_D = d_D;
        int bad = 0;

        if constexpr (isTranspose) {
            auto transpose_function = cute::make_layout(tensor_shape_S, cute::LayoutRight{});
            for (auto i = 0; i < h_D.size(); ++i) {
                if (h_D[i] != h_S[transpose_function(i)]) {
                    ++bad;
                }
            }
        } else {
            for (auto i = 0; i < h_D.size(); ++i) {
                if (h_D[i] != h_S[i]) {
                    ++bad;
                }
            }
        }

        if (bad) {
            std::cout << "Validation failed, Incorrect values: " << bad << std::endl;
        } else {
            std::cout << "Validation Success." << std::endl;
        }

    }
}

int main(int argc, char **argv) {
    int M = (1 << 15), N = (1 << 15);
    using Element = float;
    cute::device_init(1);

    std::cout << "Naive transpose:\n";
    benchmark<Element, true>(transpose_naive, M, N, 10, false);
    std::cout << "\nSmem transpose:\n";
    benchmark<Element, true>(transpose_smem<Element, false>, M, N, 10, false);
    std::cout << "\nSmem swizzled transpose:\n";
    benchmark<Element, true>(transpose_smem<Element, true>, M, N, 10, false);

    return 0;
}