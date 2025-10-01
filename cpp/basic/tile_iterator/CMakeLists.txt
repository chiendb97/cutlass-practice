add_executable(tile_iterator tile_iterator.cu)
target_include_directories(tile_iterator PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(tile_iterator PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(tile_iterator PRIVATE --expt-relaxed-constexpr)