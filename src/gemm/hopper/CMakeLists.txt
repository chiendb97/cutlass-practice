add_executable(gemm_sm90 gemm_sm90.cu)
target_include_directories(gemm_sm90 PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(gemm_sm90 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(gemm_sm90 PRIVATE --expt-relaxed-constexpr)

add_subdirectory(pipeline_gemm)