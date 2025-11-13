# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_ncclx
file(GLOB TORCHCOMMS_NCCLX_SOURCES
    "comms/torchcomms/ncclx/*.cpp"
    "comms/torchcomms/transport/*.cc"
)
file(GLOB TORCHCOMMS_CUDA_API_SOURCE "comms/torchcomms/device/CudaApi.cpp")

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
    set(FMT_INCLUDE "${ROOT}/third-party/fmt/include")

    if (NOT EXISTS "${NCCLX_BUILD_DIR}")
        execute_process(
            COMMAND ${ROOT}/build_ncclx.sh
            WORKING_DIRECTORY ${ROOT}
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output
        )
        if(result)
            message(FATAL_ERROR "NCCLX build failed: ${result}")
        endif()
    endif()
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

# Remove any -l:libglog.a from FOLLY_LDFLAGS
list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX ".*libglog.*")

# Check NCCLX include exists
if(NOT EXISTS "${NCCLX_INCLUDE}")
    message(FATAL_ERROR "NCCLX include not found at ${NCCLX_INCLUDE}")
endif()

add_library(torchcomms_comms_ncclx MODULE
    ${TORCHCOMMS_NCCLX_SOURCES}
    ${TORCHCOMMS_CUDA_API_SOURCE}
)
target_compile_definitions(torchcomms_comms_ncclx PRIVATE MOCK_SCUBA_DATA CTRAN_DISABLE_TCPDM)
set_target_properties(torchcomms_comms_ncclx PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_ncclx"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms_ncclx PRIVATE
    ${ROOT}
    ${NCCLX_INCLUDE}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_ncclx PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_ncclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_ncclx PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)
if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_SHARED_LIB}
        "-lfmt"
    )
else()
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
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
        "-l:libfmt.a"
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

install(TARGETS torchcomms_comms_ncclx
    LIBRARY DESTINATION .
)
