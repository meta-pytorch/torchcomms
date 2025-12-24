# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)

file(GLOB_RECURSE CTRAN_SOURCES
    "*.cc"
    "*.cu"
)
list(FILTER CTRAN_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX ".*/benchmarks/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX ".*/examples/.*")
if (NOT DEFINED ENABLE_TCPDM OR NOT ENABLE_TCPDM)
    add_compile_definitions(CTRAN_DISABLE_TCPDM)
    list(FILTER CTRAN_SOURCES EXCLUDE REGEX ".*/ctran/backends/tcpdevmem/.*")
endif()

list(FILTER CMAKE_CUDA_FLAGS EXCLUDE REGEX ".*arch=compute_50,code=sm_50")
list(FILTER CMAKE_CUDA_FLAGS EXCLUDE REGEX ".*arch=compute_60,code=sm_60")
list(FILTER CMAKE_CUDA_FLAGS EXCLUDE REGEX ".*arch=compute_70,code=sm_70")
list(FILTER CMAKE_CUDA_FLAGS EXCLUDE REGEX ".*arch=compute_75,code=sm_75")

message(status "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# genctran.py generates source files (*.cu) for template instantiations across
# different algorithms, data types, and operations.
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/algos/genctran.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
file(GLOB CTRAN_GENSRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/gensrc/*.cu")

add_library(ctran OBJECT
  ${CTRAN_GENSRC_FILES}
  ${CTRAN_SOURCES}
)

target_compile_features(ctran PRIVATE cxx_std_20)

target_compile_options(ctran PRIVATE
    -fPIC
)
target_compile_options(ctran PRIVATE
    # copied from nccl
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all>
)
if (NOT DEFINED ENABLE_TCPDM OR NOT ENABLE_TCPDM)
    target_compile_options(ctran PRIVATE
        # copied from nccl
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options "-DCTRAN_DISABLE_TCPDM">
    )
endif()

set_target_properties(ctran PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ctran PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
