add_executable(dump_reg_shmem dump_reg_shmem.cu)
target_include_directories(dump_reg_shmem PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(dump_reg_shmem PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(dump_reg_shmem PRIVATE --expt-relaxed-constexpr)