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

#include "cute/tensor.hpp"

#include "cutlass/arch/barrier.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "hopper_gemm_kernel_launch.h"

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC,
          cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T')
    {
        // gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
        std::cerr << "Error: Do not support for " << transA << transB << std::endl;
    }
    else if (transA == 'T' && transB == 'N')
    {
        gemm_tn<true>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

        CUTE_CHECK_LAST();
        cudaDeviceSynchronize();
    }
    else
    {
        std::cerr << "Error: Do not support for " << transA << transB << std::endl;
    }
}

void cutlass_gemm(char transA, char transB, int m, int n, int k, cutlass::half_t alpha, cutlass::half_t const* A, int ldA,
                  cutlass::half_t const* B, int ldB, cutlass::half_t beta, cutlass::half_t* C, int ldC, cudaStream_t stream = 0)
{

    if (transA == 'N' && transB == 'T')
    {
        using LayoutA = cutlass::layout::ColumnMajor;

        using LayoutB = cutlass::layout::RowMajor;

        using LayoutC = cutlass::layout::RowMajor;

        using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t, LayoutC>;

        CutlassGemm gemm;

        CutlassGemm::Arguments args({m, n, k}, {A, ldA}, {B, ldB}, {C, ldC}, {C, ldC}, {alpha, beta});

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

        using LayoutC = cutlass::layout::RowMajor;

        using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t, LayoutC>;

        CutlassGemm gemm;

        CutlassGemm::Arguments args({m, n, k}, {A, ldA}, {B, ldB}, {C, ldC}, {C, ldC}, {alpha, beta});

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

template <typename LayoutA, typename LayoutB, typename LayoutC>
void run_gemm(char transA, char transB, int m, int n, int k, int seed = 42)
{
    using TA = cutlass::half_t;
    using TB = cutlass::half_t;
    using TC = cutlass::half_t;
    using TI = cutlass::half_t;

    TI alpha = TI(1.0f);
    TI beta = TI(0.0f);

    float mean = 0.0f;
    float stddev = 2.0f;
    int bits_less_than_one = 0;

    cutlass::HostTensor<TA, LayoutA> A(cutlass::MatrixCoord(m, k));
    cutlass::HostTensor<TB, LayoutB> B(cutlass::MatrixCoord(k, n));
    cutlass::HostTensor<TC, LayoutC> C(cutlass::MatrixCoord(m, n));
    cutlass::HostTensor<TC, LayoutC> C_reference(cutlass::MatrixCoord(m, n));

    cutlass::reference::device::TensorFillRandomGaussian(A.device_view(), 2 * seed, (TA)mean, (TA)stddev, bits_less_than_one);
    cutlass::reference::device::TensorFillRandomGaussian(B.device_view(), 3 * seed, (TB)mean, (TB)stddev, bits_less_than_one);
    cutlass::reference::device::TensorFillRandomGaussian(C.device_view(), 5 * seed, (TC)mean, (TC)stddev, bits_less_than_one);

    cutlass::device_memory::copy_device_to_device(C_reference.device_data(), C.device_data(), C.capacity());

    int ldA, ldB, ldC;
    if (transA == 'N' && transB == 'T')
    {
        ldA = m;
        ldB = n;
        ldC = n;
    }
    else if (transA == 'T' && transB == 'N')
    {
        ldA = k;
        ldB = k;
        ldC = n;
    }

    double gflops = (2.0 * m * n * k) * 1e-9;
    const int n_iters = 1;
    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < n_iters; ++i)
    {
        gemm(transA, transB, m, n, k, alpha, A.device_data(), ldA, B.device_data(), ldB, beta, C.device_data(), ldC);
    }

    double process_time = timer.seconds() / n_iters;

    std::cout << "CUTE GEMM: " << gflops / process_time << " gflop/s, " << process_time * 1000 << " ms" << std::endl;

    cutlass_gemm(transA, transB, m, n, k, alpha, A.device_data(), ldA, B.device_data(), ldB, beta, C_reference.device_data(), ldC);

    C.sync_host();
    C_reference.sync_host();

    if (cutlass::reference::host::TensorRelativelyEquals(C.host_view(), C_reference.host_view(), TC(1e-2), TC(1e-8)))
    {
        std::cout << "Success" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
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
        run_gemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(transA, transB, m, n, k, seed);
    }
    else if (transA == 'T' && transB == 'N')
    {
        run_gemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(transA, transB, m, n, k, seed);
    }
    else
    {
        std::cerr << "Error: Do not support for " << transA << transB << std::endl;
    }

    return 0;
}