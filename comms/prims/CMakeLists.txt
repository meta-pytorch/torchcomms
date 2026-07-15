# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(CUDA REQUIRED)

file(GLOB_RECURSE PRIMS_SOURCES
    "*.cc"
    "*.cu"
)

# tests / benchmarks are gtest/MPI-dependent and not part of the library.
list(FILTER PRIMS_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER PRIMS_SOURCES EXCLUDE REGEX ".*/benchmarks/.*")
# AMD/HIP backend: the CUDA-only OSS build never defines __HIP_PLATFORM_AMD__,
# so the transport/amd tree is excluded (it needs mlx5dv/bnxt direct-verbs
# headers unavailable here).
list(FILTER PRIMS_SOURCES EXCLUDE REGEX ".*/transport/amd/.*")
# moe_ep is a self-contained Python `_cpp` extension (needs torch + pybind11),
# not consumed by MCCL/ctran — build it separately if OSS ever needs MoE EP.
list(FILTER PRIMS_SOURCES EXCLUDE REGEX ".*/collectives/moe_ep/.*")
# triton is Python-only.
list(FILTER PRIMS_SOURCES EXCLUDE REGEX ".*/triton/.*")

add_library(prims OBJECT
    ${PRIMS_SOURCES}
)

target_compile_features(prims PRIVATE cxx_std_20)
target_compile_options(prims PRIVATE
    -fPIC
)
# CUDA device-compile flags, mirroring comms/ctran and comms/mccl (from nccl).
target_compile_options(prims PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
    $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=96>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin>
    $<$<COMPILE_LANGUAGE:CUDA>:-compress-all>
)

set_target_properties(prims PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ROOT (fbcode root) resolves comms/prims/..., comms/common/*.cuh (header-only
# bits: AtomicUtils/DeviceConstants/BitOps), and comms/ctran/ibverbx/... .
target_include_directories(prims PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)
