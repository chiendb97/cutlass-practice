#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/core_io.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

cudaError_t cutlass_hgemm_nn(int M, int N, int K, cutlass::half_t alpha,
                             cutlass::half_t const* A,
                             cutlass::layout::ColumnMajor::Stride::Index lda,
                             cutlass::half_t const* B,
                             cutlass::layout::ColumnMajor::Stride::Index ldb,
                             cutlass::half_t beta, cutlass::half_t* C,
                             cutlass::layout::ColumnMajor::Stride::Index ldc)
{

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor>;

    Gemm gemm_op;

    cutlass::Status status = gemm_op(
        {{M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}});

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t TestCutlassGemm(int M, int N, int K, cutlass::half_t alpha,
                            cutlass::half_t beta)
{
    cudaError_t result;

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(
        cutlass::MatrixCoord(M, K));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(
        cutlass::MatrixCoord(K, N));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>
        C_cutlass(cutlass::MatrixCoord(M, N));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>
        C_reference(cutlass::MatrixCoord(M, N));

    uint64_t seed = 1997;

    cutlass::half_t mean{0.0};
    cutlass::half_t stddev{5.0};

    int bits_less = 0;

    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(), seed + 1, mean, stddev, bits_less);

    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(), seed + 2, mean, stddev, bits_less);

    cutlass::reference::device::TensorFillRandomGaussian(
        C_cutlass.device_view(), seed + 3, mean, stddev, bits_less);

    cutlass::device_memory::copy_device_to_device(C_reference.device_data(),
                                                  C_cutlass.device_data(),
                                                  C_cutlass.capacity());

    C_reference.sync_host();

    result = cutlass_hgemm_nn(M, N, K, alpha, A.device_data(), A.stride(0),
                              B.device_data(), B.stride(0), beta,
                              C_cutlass.device_data(), C_cutlass.stride(0));

    if (result != cudaSuccess)
    {
        return result;
    }

    A.sync_host();
    B.sync_host();
    C_cutlass.sync_host();

    using GemmRef = cutlass::reference::host::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, cutlass::half_t,
        cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::half_t>;

    GemmRef gemm_ref;

    gemm_ref({M, N, K}, alpha, A.host_ref(), B.host_ref(), beta,
             C_reference.host_ref());

    if (!cutlass::reference::host::TensorEquals(C_reference.host_view(),
                                                C_cutlass.host_view()))
    {
        std::cerr << "Error - CUTLASS GEMM kernel differs from reference."
                  << std::endl;
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

int main(int argc, char* argv[])
{
    int matrix_dim[3] = {256, 256, 256};
    cutlass::half_t scalars[2] = {1.0_hf, 0.0_hf};

    for (int i = 1; i < argc && i < 4; ++i)
    {
        std::stringstream ss(argv[i]);
        ss >> matrix_dim[i - 1];
    }

    for (int i = 4; i < argc && i < 6; ++i)
    {
        std::stringstream ss(argv[i]);
        ss >> scalars[i - 4];
    }

    cudaError_t result = TestCutlassGemm(matrix_dim[0], matrix_dim[1],
                                         matrix_dim[2], scalars[0], scalars[1]);

    if (result == cudaSuccess)
    {
        std::cout << "Passed." << std::endl;
    }

    return result == cudaSuccess ? 0 : -1;
}
