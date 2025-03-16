cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(cutlass-practice VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUTLASS_INCLUDE_DIRS ${CUTLASS_SOURCE_DIR}/include ${CUTLASS_SOURCE_DIR}/tools/util/include ${CUTLASS_SOURCE_DIR}/examples/common CACHE INTERNAL "")

add_subdirectory(src)