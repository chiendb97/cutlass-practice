#include <iostream>
#include <vector>
#include <sstream>

#include "helper.h"

#include "cutlass/gemm/device/gemm.h"


__global__
void InitializeMatrix_kernel(float *matrix, int row, int column, int seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < row && j < column) {
    int offset = i + j * row;
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    matrix[offset] = value;
  }
}


cudaError_t InitializeMatrix(float *matrix, int row, int column, int seed) {

  dim3 block(16, 16);
  dim3 grid((row - 1) / block.x + 1, (column - 1) / block.y + 1);

  InitializeMatrix_kernel<<<grid, block>>>(matrix, row, column, seed);

  return cudaGetLastError();
}

cudaError_t AllocateMatrix(float **matrix, int row, int column, int seed) {
  cudaError_t result;
  size_t sizeof_matrix = sizeof(float) * row * column;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result) << std::endl;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix: " << cudaGetErrorString(result) << std::endl;
  }

  result = InitializeMatrix(*matrix, row, column, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: " << cudaGetErrorString(result) << std::endl;
  }

  return result;
}

__global__ void ReferenceGemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

cudaError_t ReferenceGemm(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {

  dim3 block(16, 16);
  dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<
      float,
      ColumnMajor,
      float,
      ColumnMajor,
      float,
      ColumnMajor>;

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M, N, K},
                              {A, lda},
                              {B, ldb},
                              {C, ldc},
                              {C, ldc},
                              {alpha, beta});

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}


cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  int lda = M;
  int ldb = K;
  int ldc = M;

  size_t sizeof_C = sizeof(float) * M * N;

  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  result = AllocateMatrix(&A, M, K, 1);

  if (result != cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 2);

  if (result != cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, K, N, 3);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, K, N, 4);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "Cutlass GEMM kernel failed: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  std::vector<float> host_cutlass(M * N, 0);
  std::vector<float> host_reference(M * N, 0);

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);
    return result;
  }

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  if (host_reference != host_cutlass) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}


int main(int argc, char *argv[]) {
  int matrix_dim[3] = {256, 256, 256};
  float scalars[2] = {1, 0};

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(argv[i]);
    ss >> matrix_dim[i - 1];
  }

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(argv[i]);
    ss >> scalars[i - 4];
  }

  cudaError_t result = TestCutlassGemm(
      matrix_dim[0],
      matrix_dim[1],
      matrix_dim[2],
      scalars[0],
      scalars[1]
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  return result == cudaSuccess ? 0 : -1;
}