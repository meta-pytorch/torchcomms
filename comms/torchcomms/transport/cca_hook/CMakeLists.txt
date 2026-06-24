# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._transport_cca_hook
#
# Minimal RDMA caching-allocator-hook extension. Unlike the full
# torchcomms._transport module it does NOT add_subdirectory(comms/ctran) or
# add_subdirectory(comms/utils) and compiles no .cu sources. The only ctran
# dependency (ctran::RegCache) is resolved by linking the prebuilt NCCLX
# static lib, exactly as comms/torchcomms/ncclx/CMakeLists.txt does for
# _comms_ncclx. This lets the hook build even when USE_TRANSPORT is OFF.

set(TORCHCOMMS_CCA_HOOK_SOURCES
    "${ROOT}/comms/torchcomms/transport/RdmaTransportCCA.cpp"
    "${ROOT}/comms/torchcomms/transport/cca_hook/_transport_cca_hook.cpp"
)

enable_language(CUDA)
find_package(CUDA)

# NCCLX handling — mirrors comms/torchcomms/ncclx/CMakeLists.txt so that
# ctran::RegCache symbols resolve from the prebuilt libnccl_static.a.
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

# Remove glog/gflags from FOLLY_LDFLAGS — they're already linked statically
# via libtorchcomms.so from the third-party cmake modules.
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

add_library(torchcomms_transport_cca_hook MODULE
    ${TORCHCOMMS_CCA_HOOK_SOURCES}
)
set_target_properties(torchcomms_transport_cca_hook PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_transport_cca_hook"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_transport_cca_hook PRIVATE
    ${ROOT}
    ${NCCLX_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
torchcomms_use_torch_pybind11(torchcomms_transport_cca_hook)
target_compile_features(torchcomms_transport_cca_hook PRIVATE cxx_std_20)

target_link_directories(torchcomms_transport_cca_hook PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_transport_cca_hook PRIVATE
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
    target_link_libraries(torchcomms_transport_cca_hook PRIVATE
        ${NCCLX_SHARED_LIB}
        ${FOLLY_LDFLAGS}
        "-lboost_program_options"
        "-lboost_filesystem"
        "-lfmt"
    )
else()
    target_link_libraries(torchcomms_transport_cca_hook PRIVATE
        ${NCCLX_STATIC_LIB}
        "-l:libcomms_tracing_service.a"
        "-Wl,--start-group"
        "-l:libasync.a"
        "-l:libconcurrency.a"
        "-l:libthrift-core.a"
        "-l:libthriftanyrep.a"
        "-l:libthriftcpp2.a"
        "-l:libthriftmetadata.a"
        "-l:libthriftprotocol.a"
        "-l:libthrifttype.a"
        "-l:libthrifttyperep.a"
        "-l:librpcmetadata.a"
        "-l:libruntime.a"
        "-l:libserverdbginfo.a"
        "-l:libtransport.a"
        "-l:libcommon.a"
        "-Wl,--end-group"
        "-l:libfolly.a"
        "-l:libwangle.a"
        "-l:libfizz.a"
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
        TARGET torchcomms_transport_cca_hook
        PRE_LINK
        COMMAND "${ROOT}/rename_symbols.sh"
                '$<TARGET_OBJECTS:torchcomms_transport_cca_hook>'
    )
    add_custom_command(
        TARGET torchcomms_transport_cca_hook
        PRE_LINK
        COMMAND "${ROOT}/rename_symbols.sh" "${NCCLX_STATIC_LIB}"
    )
endif()

if(CUDA_FOUND)
    target_link_libraries(torchcomms_transport_cca_hook PRIVATE CUDA::cudart)
endif()

# Link absl LAST — after the ncclx archive and the thrift / folly archives
# above, for the same ordering reasons documented in ncclx/CMakeLists.txt.
target_link_libraries(torchcomms_transport_cca_hook PRIVATE
    "-Wl,--start-group"
    ${ABSL_LDFLAGS}
    "-Wl,--end-group"
)

install(TARGETS torchcomms_transport_cca_hook
    LIBRARY DESTINATION .
)
