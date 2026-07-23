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

set(PRIMS_CUDA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
list(FILTER PRIMS_CUDA_ARCHS EXCLUDE REGEX "^(5[0-9]|6[0-9]|7[0-9]|8[0-9])")
if(NOT PRIMS_CUDA_ARCHS)
    set(PRIMS_CUDA_ARCHS OFF)
endif()
set_target_properties(prims PROPERTIES CUDA_ARCHITECTURES "${PRIMS_CUDA_ARCHS}")
message(STATUS "prims CUDA_ARCHITECTURES: ${PRIMS_CUDA_ARCHS} (from CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")

# ROOT (fbcode root) resolves comms/prims/..., comms/common/*.cuh (header-only
# bits: AtomicUtils/DeviceConstants/BitOps), and comms/ctran/ibverbx/... .
target_include_directories(prims PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
)

# prims is only compiled when the top level does add_subdirectory(comms/prims),
# so reaching this file means prims is enabled: set ENABLE_PRIMS unconditionally
# rather than referencing the higher-level MCCL_ENABLE_PRIMS option. PRIVATE:
# bakes the macro into prims' own objects. (Consumers that include ctran/mccl
# headers set ENABLE_PRIMS themselves for the ODR/export boundary; see the
# top-level CMakeLists.txt.)
target_compile_definitions(prims PRIVATE ENABLE_PRIMS=1)

# spdlog config for transport/rdma/NicDiscovery.cc (the only spdlog consumer in
# prims; no prims header includes spdlog). Consume spdlog header-only:
# SPDLOG_FMT_EXTERNAL uses the build's external fmt (not spdlog's bundled copy,
# an ODR hazard), while NOT defining SPDLOG_COMPILED_LIB inlines all spdlog code
# into NicDiscovery.o, so no libspdlog.a is linked. This sidesteps the static-
# archive ordering trap that otherwise left compiled-lib symbols (e.g.
# spdlog::logger::err_handler_) undefined in the shared module at dlopen.
target_compile_definitions(prims PRIVATE SPDLOG_FMT_EXTERNAL)

# Third-party link deps every consumer of the prims transports needs, declared
# here so they propagate automatically to any target that links the `prims`
# OBJECT library (PUBLIC = usage requirement). Resolved from CONDA_LIB (and
# THIRD_PARTY_LIB when set). doca_gpunetio: IBGDA; ibverbs/mlx5: IB verbs;
# dl: NVML dlopen. The CUDA driver (libcuda) is deliberately NOT linked: prims
# (platform/CudaDriverLazy.cc) and ctran (CudaWrap.cc) resolve driver entry
# points lazily via cudaGetDriverEntryPoint, so only CUDA::cudart is a link-time
# dep. Hard-linking cuda_driver would add a libcuda.so.1 NEEDED that breaks
# import on driverless build/verify hosts.
target_link_directories(prims PUBLIC ${CONDA_LIB})
if(DEFINED THIRD_PARTY_LIB)
    target_link_directories(prims PUBLIC ${THIRD_PARTY_LIB})
endif()
target_link_libraries(prims PUBLIC
    "-ldoca_gpunetio"
    "-libverbs"
    "-lmlx5"
    "-ldl"
)

# Install prims' public headers (ctran's installed headers #include
# "comms/prims/..." under ENABLE_PRIMS, so these must ship for the exported
# Ctran/Mccl packages to compile). Owned here so prims specifies what/where to
# install. The EXCLUDE list is an intentional lean-package boundary, kept
# co-located with the source glob above (lines ~11-21) so both are maintained in
# one place. DESTINATION must stay include/comms/prims to match those includes;
# file(REAL_PATH) resolves the symlinked OSS source tree.
file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" _PRIMS_SRC_DIR)
install(DIRECTORY ${_PRIMS_SRC_DIR}/ DESTINATION include/comms/prims
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
    PATTERN "tests" EXCLUDE PATTERN "benchmarks" EXCLUDE
    PATTERN "amd" EXCLUDE PATTERN "moe_ep" EXCLUDE PATTERN "triton" EXCLUDE)
