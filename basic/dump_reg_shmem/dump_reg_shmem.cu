#include <iostream>

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#define EXAMPLE_MATRIX_ROW 64
#define EXAMPLE_MATRIX_COL 32


template <typename Element, typename GmemIterator, typename SmemIterator>
__global__ void kernel_dump(typename GmemIterator::Params params,
                            typename GmemIterator::TensorRef ref) {
  extern __shared__ Element shared_storage[];

  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  GmemIterator gmem_iterator(params, ref.data(),
                             {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL},
                             tb_thread_id);

  typename GmemIterator::Fragment frag;

  frag.clear();
  gmem_iterator.load(frag);

  // Call dump_fragment() with different parameters.
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nAll threads dump all the elements:\n");
  cutlass::debug::dump_fragment(frag);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps all the elements:\n");
  cutlass::debug::dump_fragment(frag, /*N = */ 1);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps first 16 elements:\n");
  cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
  cutlass::debug::dump_fragment(frag, /*N = */ 1, /*M = */ 16, /*S = */ 8);

  SmemIterator smem_iterator(
      typename SmemIterator::TensorRef(
          {shared_storage, SmemIterator::Layout::packed(
              {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL})}),
      tb_thread_id);

  smem_iterator.store(frag);

  if (threadIdx.x == 0 && blockIdx.x == 0) printf("\nDump all the elements:\n");
  cutlass::debug::dump_shmem(shared_storage,
                             EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nDump all the elements with a stride of 8:\n");
  cutlass::debug::dump_shmem(
      shared_storage, EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL, /*S = */ 8);
}

int main() {
  using Element = cutlass::half_t;
  using Layout = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<Element, Layout> matrix(
      {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL});
  cutlass::reference::host::BlockFillSequential(matrix.host_data(),
                                                matrix.capacity());

  std::cout << "Matrix:\n" << matrix.host_view() << "\n";

  matrix.sync_device();

  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>,
      32, cutlass::layout::PitchLinearShape<8, 4>, 8>;

  using GmemIterator =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, Element,
          Layout, 1, ThreadMap>;

  typename GmemIterator::Params params(matrix.layout());

  using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, Element,
      cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, 1,
      ThreadMap>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  int smem_size =
      int(sizeof(Element) * EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

  kernel_dump<Element, GmemIterator, SmemIterator>
  <<<grid, block, smem_size, 0>>>(params, matrix.device_ref());

  cudaError_t result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    std::cout << "Failed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);
}