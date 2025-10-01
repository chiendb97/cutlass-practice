add_executable(tutorial_layout tutorial_layout.cu)
target_include_directories(tutorial_layout PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(tutorial_layout PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(tutorial_layout PRIVATE --expt-relaxed-constexpr)

add_executable(tutorial_mma tutorial_mma.cu)
target_include_directories(tutorial_mma PRIVATE ${CUTLASS_INCLUDE_DIRS})
set_target_properties(tutorial_mma PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(tutorial_mma PRIVATE --expt-relaxed-constexpr)