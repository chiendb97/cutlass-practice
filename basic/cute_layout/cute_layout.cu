#include "cute/layout.hpp"


int main(int argc, char *argv[]) {
  cudaError_t result;
  cute::Layout a = make_layout(make_shape (cute::Int<10>{}, cute::Int<2>{}),
                         make_stride(cute::Int<16>{}, cute::Int<4>{}));
  cute::Layout b = make_layout(make_shape (cute::Int< 5>{}, cute::Int<4>{}),
                         make_stride(cute::Int< 1>{}, cute::Int<5>{}));
  cute::Layout c = composition(a, b);
  print(c);
  return result == cudaSuccess ? 0 : -1;
}