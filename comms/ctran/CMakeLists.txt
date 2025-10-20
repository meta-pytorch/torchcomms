# Copyright (c) Meta Platforms, Inc. and affiliates.

file(GLOB_RECURSE CTRAN_SOURCES
    "*.cc"
    "*.cu"
)
list(FILTER CTRAN_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER CTRAN_SOURCES EXCLUDE REGEX ".*/benchmarks/.*")

# genctran.py generates source files (*.cu) for template instantiations across different algorithms, data types, and operations.
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/algos/genctran.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
file(GLOB CTRAN_GENSRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/gensrc/*.cu")

add_library(ctran OBJECT
  ${CTRAN_GENSRC_FILES}
  ${CTRAN_SOURCES}
)

target_compile_options(ctran PRIVATE
    -fPIC
)
target_compile_options(ctran PRIVATE
    # copied from nccl
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all>
)

set_target_properties(ctran PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ctran PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE}
)
