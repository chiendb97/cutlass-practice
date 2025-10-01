add_subdirectory(hopper)

add_executable(basic_gemm basic_gemm.cu)
target_include_directories(basic_gemm PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(basic_gemm PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(basic_gemm PRIVATE --expt-relaxed-constexpr)

add_executable(sgemm_1 sgemm_1.cu)
target_include_directories(sgemm_1 PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(sgemm_1 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_1 PRIVATE --expt-relaxed-constexpr)

add_executable(sgemm_2 sgemm_2.cu)
target_include_directories(sgemm_2 PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(sgemm_2 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_2 PRIVATE --expt-relaxed-constexpr)

add_executable(sgemm_sm80 sgemm_sm80.cu)
target_include_directories(sgemm_sm80 PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(sgemm_sm80 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_sm80 PRIVATE --expt-relaxed-constexpr)

add_executable(gemm_sm80 gemm_sm80.cu)
target_include_directories(gemm_sm80 PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(gemm_sm80 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(gemm_sm80 PRIVATE --expt-relaxed-constexpr)

