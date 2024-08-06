#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/tensor_fill.h"

#pragma warning( disable : 4503)

template <typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<2> extent) {


    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment fragment;

    for(size_t i = 0; i < fragment.size(); ++i) {
      fragment[i] = 0;
    }

    src_iterator.load(fragment);
    dst_iterator.store(fragment);


    ++src_iterator;
    ++dst_iterator;

    for(; iterations > 1; --iterations) {

      src_iterator.load(fragment);
      dst_iterator.store(fragment);

      ++src_iterator;
      ++dst_iterator;
    }
}

cudaError_t TestTileIterator(int M, int K) {

    using Shape = cutlass::layout::PitchLinearShape<64, 4>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int const kThreads = 32;

    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

    using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
        Shape, Element, Layout, 1, ThreadMap>;


    cutlass::Coord<2> copy_extent = cutlass::make_Coord(M, K);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(M, K);

    cutlass::HostTensor<Element, Layout> src_tensor(alloc_extent);
    cutlass::HostTensor<Element, Layout> dst_tensor(alloc_extent);

    Element oob_value = Element(-1);

    cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
    cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

    dst_tensor.sync_device();
    src_tensor.sync_device();

    typename Iterator::Params dst_params(dst_tensor.layout());
    typename Iterator::Params src_params(src_tensor.layout());

    dim3 block(kThreads, 1);
    dim3 grid(1, 1);

    copy<Iterator><<< grid, block >>>(
            dst_params,
            dst_tensor.device_data(),
            src_params,
            src_tensor.device_data(),
            copy_extent
    );

    cudaError_t result = cudaGetLastError();
    if(result != cudaSuccess) {
      std::cerr << "Error - kernel failed." << std::endl;
      return result;
    }

    dst_tensor.sync_host();

    for(int s = 0; s < alloc_extent[1]; ++s) {
      for(int c = 0; c < alloc_extent[0]; ++c) {

          Element expected = Element(0);

          if(c < copy_extent[0] && s < copy_extent[1]) {
            expected = src_tensor.at({c, s});
          }
          else {
            expected = oob_value;
          }

          Element got = dst_tensor.at({c, s});
          bool equal = (expected == got);

          if(!equal) {
              std::cerr << "Error - source tile differs from destination tile." << std::endl;
            return cudaErrorUnknown;
          }
      }
    }

    return cudaSuccess;
}

int main(int argc, const char *arg[]) {

    cudaError_t result = TestTileIterator(57, 35);

    if(result == cudaSuccess) {
      std::cout << "Passed." << std::endl;  
    }

    // Exit
    return result == cudaSuccess ? 0 : -1;
}
