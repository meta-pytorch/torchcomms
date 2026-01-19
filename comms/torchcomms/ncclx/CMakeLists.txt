# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_ncclx

# NCCLX handling - paths are set by cmake/NCCLX.cmake or for system libs
# NCCLX requires Meta's extended NCCL headers, not standard NCCL
if(USE_SYSTEM_LIBS)
    # When using system libs, check if NCCLX headers are available in CONDA
    # NCCLX has types like ncclWinAccessType that don't exist in standard NCCL
    if(EXISTS "${CONDA_INCLUDE}/nccl.h")
        file(STRINGS "${CONDA_INCLUDE}/nccl.h" NCCLX_CHECK REGEX "ncclWinAccessType")
        if(NCCLX_CHECK)
            set(NCCLX_INCLUDE "${CONDA_INCLUDE}")
            set(NCCLX_SHARED_LIB "${CONDA_LIB}/libnccl.so")
            set(HAVE_NCCLX TRUE)
        else()
            message(STATUS "NCCL found in CONDA but lacks NCCLX extensions - skipping ncclx module")
            set(HAVE_NCCLX FALSE)
        endif()
    else()
        message(STATUS "NCCLX headers not found in CONDA - skipping ncclx module (standard NCCL from PyTorch doesn't have NCCLX extensions)")
        set(HAVE_NCCLX FALSE)
    endif()

    if(NOT HAVE_NCCLX)
        return()
    endif()
else()
    # NCCLX_INCLUDE, NCCLX_STATIC_LIB, NCCLX_SHARED_LIB are set by cmake/NCCLX.cmake
    # and will be built by ncclx_build target
    if(NOT DEFINED NCCLX_INCLUDE)
        set(NCCLX_BUILD_DIR $ENV{BUILDDIR})
        if(NOT NCCLX_BUILD_DIR)
            set(NCCLX_BUILD_DIR "${CMAKE_BINARY_DIR}/ncclx")
        endif()
        set(NCCLX_INCLUDE "${NCCLX_BUILD_DIR}/include")
        set(NCCLX_STATIC_LIB "${NCCLX_BUILD_DIR}/lib/libnccl_static.a")
        set(NCCLX_SHARED_LIB "${NCCLX_BUILD_DIR}/lib/libnccl.so")
    endif()
endif()
message(STATUS "NCCLX include: ${NCCLX_INCLUDE}")

file(GLOB TORCHCOMMS_NCCLX_SOURCES
    "comms/torchcomms/ncclx/*.cpp"
)
file(GLOB TORCHCOMMS_CUDA_API_SOURCE "comms/torchcomms/device/CudaApi.cpp")

find_package(CUDA)

# Get folly LDFLAGS using pkg-config (will fail during configure if deps not built,
# but that's okay - deps will be built first due to target dependencies)
set(ENV{PKG_CONFIG_PATH} "${CONDA_LIB}/pkgconfig")
execute_process(
    COMMAND pkg-config --libs --static libfolly
    OUTPUT_VARIABLE FOLLY_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PKG_RESULT
    ERROR_QUIET
)
if(PKG_RESULT EQUAL 0)
    separate_arguments(FOLLY_LDFLAGS NATIVE_COMMAND "${FOLLY_LDFLAGS_RAW}")
    # Remove any -l:libglog.a from FOLLY_LDFLAGS
    list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX ".*libglog.*")
else()
    # Deps not built yet, will be built before this target
    set(FOLLY_LDFLAGS "")
    message(STATUS "pkg-config for libfolly not available yet, deps will be built first")
endif()

add_library(torchcomms_comms_ncclx MODULE
    ${TORCHCOMMS_NCCLX_SOURCES}
    ${TORCHCOMMS_CUDA_API_SOURCE}
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
    # Add dependency on ncclx build target
    if(TARGET ncclx_build)
        add_dependencies(torchcomms_comms_ncclx ncclx_build)
    endif()

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
