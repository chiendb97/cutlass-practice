add_executable(fmha_forward fmha_forward.cu)
target_include_directories(fmha_forward PRIVATE ${CUTLASS_INCLUDE_DIRS} lib/gemm)
set_target_properties(fmha_forward PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(fmha_forward PRIVATE --expt-relaxed-constexpr)