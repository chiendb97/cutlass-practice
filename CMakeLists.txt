cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
include(FetchContent)

include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)
include(cmake/modules/resolve_dirs.cmake)
include(cmake/modules/parse_make_options.cmake)
include(cmake/modules/cuda.cmake)

project(cutlass_practice LANGUAGES CXX)

# Determine CUDA version before enabling the language extension
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    if(NOT WIN32) # Linux
        execute_process(
                COMMAND
                "bash" "-c"
                "${CMAKE_CUDA_COMPILER} --version | egrep -o 'V[0-9]+.[0-9]+.[0-9]+' | cut -c2-"
                RESULT_VARIABLE _BASH_SUCCESS
                OUTPUT_VARIABLE CMAKE_CUDA_COMPILER_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(NOT _BASH_SUCCESS EQUAL 0)
            message(FATAL_ERROR "Failed to determine CUDA version")
        endif()

    else() # Windows
        execute_process(
                COMMAND ${CMAKE_CUDA_COMPILER} --version
                OUTPUT_VARIABLE versionString
                RESULT_VARIABLE versionResult)

        if(versionResult EQUAL 0 AND versionString MATCHES
                "V[0-9]+\\.[0-9]+\\.[0-9]+")
            string(REGEX REPLACE "V" "" version ${CMAKE_MATCH_0})
            set(CMAKE_CUDA_COMPILER_VERSION "${version}")
        else()
            message(FATAL_ERROR "Failed to determine CUDA version")
        endif()
    endif()
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()

set(CUDA_REQUIRED_VERSION "11.2")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS CUDA_REQUIRED_VERSION)
    message(
            FATAL_ERROR
            "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} must be at least ${CUDA_REQUIRED_VERSION}"
    )
endif()

# Initialize CMAKE_CUDA_ARCHITECTURES before enabling CUDA
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real 89-real 90-real)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real)
    endif()
endif()

# Initialize CMAKE_CUDA_ARCHITECTURES before enabling CUDA
if(NOT DEFINED CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        set(CUDA_ARCHITECTURES 70 80 86 89 90)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    # Detect highest available compute capability
    set(OUTPUTFILE ${PROJECT_BINARY_DIR}/detect_cuda_arch)
    set(CUDAFILE ${CMAKE_SOURCE_DIR}/cmake/utils/detect_cuda_arch.cu)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o
            ${OUTPUTFILE})
    message(VERBOSE "Detecting native CUDA compute capability")
    execute_process(
            COMMAND ${OUTPUTFILE}
            RESULT_VARIABLE CUDA_RETURN_CODE
            OUTPUT_VARIABLE CUDA_ARCH_OUTPUT)
    if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
        message(WARNING "Detecting native CUDA compute capability - fail")
        message(
                WARNING "CUDA compute capability detection failed, compiling for 'all'")
        set(CMAKE_CUDA_ARCHITECTURES_ORIG "all")
    else()
        message(STATUS "Detecting native CUDA compute capability - done")
        set(CMAKE_CUDA_ARCHITECTURES_ORIG "${CUDA_ARCH_OUTPUT}")
    endif()
else()
    # Store CMAKE_CUDA_ARCHITECTURES for later use since torch sets this to "OFF"
    set(CMAKE_CUDA_ARCHITECTURES_ORIG "${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "GPU architectures: ${CMAKE_CUDA_ARCHITECTURES_ORIG}")

enable_language(C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

# Prevent CMake from creating a response file for CUDA compiler, so clangd can
# pick up on the includes
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

if(USE_SHARED_NVRTC)
    if(WIN32)
        message(FATAL_ERROR "Cannot use NVRTC shared library on Windows.")
    else()
        find_library(
                NVRTC_LIB nvrtc
                HINTS ${CUDAToolkit_LIBRARY_DIR}
                PATH_SUFFIXES lib64 lib lib/x64)
        find_library(
                NVRTC_BUILTINS_LIB nvrtc-builtins
                HINTS ${CUDAToolkit_LIBRARY_DIR}
                PATH_SUFFIXES lib64 lib lib/x64)
    endif()
else()
    if(WIN32)
        find_library(
                NVRTC_LIB nvrtc
                HINTS ${CUDAToolkit_LIBRARY_DIR}
                PATH_SUFFIXES lib64 lib lib/x64)
    else()
        find_library(
                NVRTC_LIB nvrtc_static
                HINTS ${CUDAToolkit_LIBRARY_DIR}
                PATH_SUFFIXES lib64 lib lib/x64)
        find_library(
                NVRTC_BUILTINS_LIB nvrtc-builtins_static
                HINTS ${CUDAToolkit_LIBRARY_DIR}
                PATH_SUFFIXES lib64 lib lib/x64)
        find_library(
                NVPTXCOMPILER_LIB nvptxcompiler_static
                HINTS ${CUDAToolkit_LIBRARY_DIR}
                PATH_SUFFIXES lib64 lib lib/x64)
    endif()
endif()

set(CUBLAS_LIB CUDA::cublas)
set(CUBLASLT_LIB CUDA::cublasLt)
set(CUDA_DRV_LIB CUDA::cuda_driver)
set(CUDA_NVML_LIB CUDA::nvml)
set(CUDA_RT_LIB CUDA::cudart_static)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_library(RT_LIB rt)

set_ifndef(ENABLE_MULTI_DEVICE 1)
if(ENABLE_MULTI_DEVICE EQUAL 1)
    # NCCL dependencies
    set_ifndef(NCCL_LIB_DIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/)
    set_ifndef(NCCL_INCLUDE_DIR /usr/include/)
    find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})
endif()

# Declare Cutlass as a dependency only once
FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v3.5.0
        CMAKE_ARGS -DCUTLASS_NVCC_ARCHS=${CUDA_ARCHITECTURES}
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
    add_subdirectory(${cutlass_SOURCE_DIR} ${cutlass_BINARY_DIR})
endif()

include_directories(
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDNN_ROOT_DIR}/include
        ${NCCL_INCLUDE_DIR}
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

set(CUTLASS_PRACTICE_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(cutlass_practice)


function(cutlass_apply_cuda_gencode_flags TARGET)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SM_ARCHS)
    cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (__SM_ARCHS)
        set(ARCHS_ENABLED ${__SM_ARCHS})
    else()
        set(ARCHS_ENABLED ${CUTLASS_NVCC_ARCHS_ENABLED})
    endif()

    set(NVCC_FLAGS)
    set(CLANG_FLAGS)
    set(__CMAKE_CUDA_ARCHS)
    foreach(ARCH ${ARCHS_ENABLED})
        list(APPEND CLANG_FLAGS --cuda-gpu-arch=sm_${ARCH})
        set(CODES)
        if(CUTLASS_NVCC_EMBED_CUBIN)
            list(APPEND CODES sm_${ARCH})
            list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-real)
        endif()
        if(CUTLASS_NVCC_EMBED_PTX)
            list(APPEND CODES compute_${ARCH})
            list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-virtual)
        endif()
        list(JOIN CODES "," CODES_STR)
        list(APPEND NVCC_FLAGS -gencode=arch=compute_${ARCH},code=[${CODES_STR}])
    endforeach()

    if (NOT __SM_ARCHS)
        if (CUDA_COMPILER MATCHES "[Cc]lang")
            target_compile_options(
                    ${TARGET}
                    PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:${CLANG_FLAGS}>
            )
        elseif(CMAKE_VERSION GREATER_EQUAL 3.18)
            set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES ${__CMAKE_CUDA_ARCHS})
        else()
            target_compile_options(
                    ${TARGET}
                    PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
            )
        endif()
    else()
        list(JOIN CLANG_FLAGS " " CLANG_FLAGS_STR)
        list(JOIN NVCC_FLAGS " " STR_NVCC_FLAGS)
        if (CUDA_COMPILER MATCHES "[Cc]lang")
            if(${TARGET} MATCHES ".*\.cpp")
                set_source_files_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${CLANG_FLAGS_STR})
            endif()
        elseif(CMAKE_VERSION GREATER_EQUAL 3.18)
            set_source_files_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES ${STR_NVCC_FLAGS})
        else()
            if(${TARGET} MATCHES ".*\.cu")
                set_source_files_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${STR_NVCC_FLAGS})
            endif()
        endif()
    endif()

endfunction()


function(cutlass_apply_standard_compile_options TARGET)

    if(CUDA_COMPILER MATCHES "[Cc]lang")
        set(CUDA_COMPILE_LANGUAGE CXX)
        set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_CLANG_FLAGS})
        set(_FLAGS_RELEASE ${__CUTLASS_CUDA_FLAGS_RELEASE} ${__CUTLASS_CUDA_CLANG_FLAGS_RELEASE})
        set(_FLAGS_RELWITHDEBINFO ${__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} ${__CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO})
        set(_FLAGS_DEBUG ${__CUTLASS_CUDA_FLAGS_DEBUG} ${__CUTLASS_CUDA_CLANG_FLAGS_DEBUG})
    else()
        set(CUDA_COMPILE_LANGUAGE CUDA)
        set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_NVCC_FLAGS})
        set(_FLAGS_RELEASE ${__CUTLASS_CUDA_FLAGS_RELEASE} ${__CUTLASS_CUDA_NVCC_FLAGS_RELEASE})
        set(_FLAGS_RELWITHDEBINFO ${__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} ${__CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO})
        set(_FLAGS_DEBUG ${__CUTLASS_CUDA_FLAGS_DEBUG} ${__CUTLASS_CUDA_NVCC_FLAGS_DEBUG})
    endif()

    target_link_libraries(${TARGET} PRIVATE cutlass_lib cutlass_tools_util_includes)

    target_compile_options(
            ${TARGET}
            PRIVATE
            $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:${_FLAGS}>
            $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELEASE>:${_FLAGS_RELEASE}>>
            $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELWITHDEBINFO>:${_FLAGS_RELWITHDEBINFO}>>
            $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:DEBUG>:${_FLAGS_DEBUG}>>
    )

endfunction()

function(cutlass_practice_add_executable NAME)

    set(options)
    set(oneValueArgs DISABLE_TESTS)
    set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
    cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED __DISABLE_TESTS)
        set(__DISABLE_TESTS OFF)
    endif()

    cutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS} BATCH_SOURCES OFF)

    add_dependencies(cutlass_practice ${NAME})

    target_link_libraries(
            ${NAME}
            PRIVATE
            cutlass_lib
            cutlass_tools_util_includes
            $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
            cuda
    )

    target_include_directories(
            ${NAME}
            PRIVATE
            ${CUTLASS_PRACTICE_COMMON_SOURCE_DIR}
            ${CUTLASS_PRACTICE_UTILS_DIR}
    )

    set_target_properties(${NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

    install(
            TARGETS ${NAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

endfunction()

add_subdirectory(basic)