# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_ncclx
file(GLOB TORCHCOMMS_NCCLX_SOURCES
    "comms/torchcomms/ncclx/*.cpp"
)
file(GLOB TORCHCOMMS_CUDA_DEVICE_SOURCES
    "comms/torchcomms/device/cuda/*.cpp"
    "comms/torchcomms/device/cuda/*.cu"
)

enable_language(CUDA)
find_package(CUDA)

# NCCLX handling
if(USE_SYSTEM_LIBS)
    set(NCCLX_INCLUDE "${CONDA_INCLUDE}")
    set(NCCLX_STATIC_LIB "${CONDA_LIB}/libnccl_static.a")
    set(NCCLX_SHARED_LIB "${CONDA_LIB}/libnccl.so")
else()
    set(NCCLX_BUILD_DIR $ENV{BUILDDIR})
    if(NOT NCCLX_BUILD_DIR)
        set(NCCLX_BUILD_DIR "${ROOT}/build/ncclx")
    endif()
    set(NCCLX_INCLUDE "${NCCLX_BUILD_DIR}/include")
    set(NCCLX_STATIC_LIB "${NCCLX_BUILD_DIR}/lib/libnccl_static.a")
    set(NCCLX_SHARED_LIB "${NCCLX_BUILD_DIR}/lib/libnccl.so")
    # build_ncclx.sh is invoked by build_ncclx.cmake (included before other
    # third-party deps) so NCCLX_BUILD_DIR should already exist at this point.
endif()

# Get folly LDFLAGS using pkg-config
set(ENV{PKG_CONFIG_PATH} "${CONDA_LIB}/pkgconfig")
execute_process(
    COMMAND pkg-config --libs --static libfolly
    OUTPUT_VARIABLE FOLLY_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PKG_RESULT
)
if(NOT PKG_RESULT EQUAL 0)
    message(FATAL_ERROR "pkg-config for libfolly failed")
endif()
separate_arguments(FOLLY_LDFLAGS NATIVE_COMMAND "${FOLLY_LDFLAGS_RAW}")

# Remove glog/gflags/fmt from FOLLY_LDFLAGS — they're already linked
# statically via libtorchcomms.so from the third-party cmake modules.
# pkg-config may output -lglog, -l:libglog.a, or full paths like
# /path/to/libglog.a — the simple substring match handles all forms.
list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX "glog")
list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX "gflags")

execute_process(
    COMMAND pkg-config --libs --static absl_log absl_check
    OUTPUT_VARIABLE ABSL_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PKG_RESULT
)
if(NOT PKG_RESULT EQUAL 0)
    message(FATAL_ERROR "pkg-config for absl_log/absl_check failed")
endif()
separate_arguments(ABSL_LDFLAGS NATIVE_COMMAND "${ABSL_LDFLAGS_RAW}")

# Check NCCLX include exists
if(NOT EXISTS "${NCCLX_INCLUDE}")
    message(FATAL_ERROR "NCCLX include not found at ${NCCLX_INCLUDE}")
endif()

# Check if NCCL Device API headers are available (requires NCCLX 2.28+)
# This enables the device API for GPU-initiated networking (GIN) support
if(EXISTS "${NCCLX_INCLUDE}/nccl_device/core.h")
    message(STATUS
        "NCCL Device API headers found,"
        " building with device API support")
    set(TORCHCOMMS_HAS_NCCL_DEVICE_API ON)
    file(GLOB TORCHCOMMS_DEVICE_NCCLX_SOURCE
        "comms/torchcomms/device/ncclx/*.cpp")
else()
    message(STATUS
        "NCCL Device API headers NOT found"
        " (requires NCCLX 2.28+),"
        " building without device API support")
    set(TORCHCOMMS_HAS_NCCL_DEVICE_API OFF)
    set(TORCHCOMMS_DEVICE_NCCLX_SOURCE "")
endif()

# Check if ENABLE_PRIMS is requested via environment variable.
# When enabled, compiles prims device backend support into _comms_ncclx.
# Requires NCCL built with ENABLE_PRIMS=1 (prims transport in libnccl.so).
if("$ENV{ENABLE_PRIMS}" STREQUAL "1")
    set(TORCHCOMMS_ENABLE_PRIMS ON)
    message(STATUS "ENABLE_PRIMS=1 set, building with prims support")
    if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
        file(GLOB TORCHCOMMS_DEVICE_PIPES_SOURCE
            "comms/torchcomms/device/pipes/*.cpp")
    else()
        set(TORCHCOMMS_DEVICE_PIPES_SOURCE "")
    endif()
else()
    set(TORCHCOMMS_ENABLE_PRIMS OFF)
    set(TORCHCOMMS_DEVICE_PIPES_SOURCE "")
    message(STATUS "ENABLE_PRIMS not set, building without prims support")
endif()

add_library(torchcomms_comms_ncclx MODULE
    ${TORCHCOMMS_NCCLX_SOURCES}
    ${TORCHCOMMS_CUDA_DEVICE_SOURCES}
    ${TORCHCOMMS_DEVICE_NCCLX_SOURCE}
    ${TORCHCOMMS_DEVICE_PIPES_SOURCE}
    comms/utils/CudaRAII.cc
    comms/utils/GraphCaptureSideStream.cc
)
set_target_properties(torchcomms_comms_ncclx PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_ncclx"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms_ncclx PRIVATE
    ${ROOT}
    ${NCCLX_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
torchcomms_use_torch_pybind11(torchcomms_comms_ncclx)
target_compile_features(torchcomms_comms_ncclx PRIVATE cxx_std_20)

# Add device API compile definition if headers are available
if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
    target_compile_definitions(torchcomms_comms_ncclx
        PRIVATE TORCHCOMMS_HAS_NCCL_DEVICE_API=1)
endif()

# Add ENABLE_PRIMS compile definition for prims device backend
if(TORCHCOMMS_ENABLE_PRIMS)
    target_compile_definitions(torchcomms_comms_ncclx
        PRIVATE ENABLE_PRIMS=1)
endif()

target_link_directories(torchcomms_comms_ncclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_ncclx PRIVATE
    fmt::fmt
    torchcomms
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
)
if(USE_SYSTEM_LIBS)
    # Link libfmt explicitly: torchcomms compiles with FMT_HEADER_ONLY=1
    # (via fmt::fmt INTERFACE target) so most fmt symbols are inlined, but
    # a few (assert_fail, thousands_sep_impl) still emit out-of-line
    # references.
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_SHARED_LIB}
        ${FOLLY_LDFLAGS}
        "-lboost_program_options"
        "-lboost_filesystem"
        "-lfmt"
    )
else()
    set(TORCHCOMMS_NCCLX_PRIMS_LINK_LIBS "")
    if(TORCHCOMMS_ENABLE_PRIMS)
        list(APPEND TORCHCOMMS_NCCLX_PRIMS_LINK_LIBS
            "-ldoca_gpunetio"
            "-libverbs"
            "-lmlx5"
        )
    endif()
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_STATIC_LIB}
        ${TORCHCOMMS_NCCLX_PRIMS_LINK_LIBS}
        "-l:libfolly.a"
        "-l:libcrypto.a"
        "-l:libssl.a"
        "-l:libxxhash.a"
        "-l:libboost_filesystem.a"
        "-l:libboost_context.a"
        ${FOLLY_LDFLAGS}
    )
    # Rename NCCL symbols to avoid conflicting with OSS nccl* that is bundled
    # with PyTorch.
    add_custom_command(
        TARGET torchcomms_comms_ncclx
        PRE_LINK
        COMMAND "${ROOT}/rename_symbols.sh"
                '$<TARGET_OBJECTS:torchcomms_comms_ncclx>'
    )
    add_custom_command(
        TARGET torchcomms_comms_ncclx
        PRE_LINK
        COMMAND "${ROOT}/rename_symbols.sh" "${NCCLX_STATIC_LIB}"
    )
endif()

if(CUDA_FOUND)
    target_link_libraries(torchcomms_comms_ncclx PRIVATE CUDA::cudart)
endif()

# Link absl LAST — after the ncclx archive (NCCLX_STATIC_LIB) and the folly
# archives above. The static absl_log_internal_* members that satisfy
# ncclx's LOG / CHECK references (e.g. absl::log_internal::kCharNull) are only
# pulled in if an undefined reference is already pending when the linker reaches
# them; placed before the ncclx archive they'd be dropped, leaving the symbol
# undefined at import time. The --start-group wrap also absorbs the mutual
# references among the absl_log_* archives themselves.
target_link_libraries(torchcomms_comms_ncclx PRIVATE
    "-Wl,--start-group"
    ${ABSL_LDFLAGS}
    "-Wl,--end-group"
)

install(TARGETS torchcomms_comms_ncclx
    LIBRARY DESTINATION .
)
