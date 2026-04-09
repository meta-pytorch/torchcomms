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

    # ── CUDA setup ──────────────────────────────────────────────────────
    set(CUDA_HOME $ENV{CUDA_HOME})
    if(NOT CUDA_HOME)
        set(CUDA_HOME "/usr/local/cuda")
    endif()

    # ── NCCL_HOME ───────────────────────────────────────────────────────
    set(NCCL_HOME $ENV{NCCL_HOME})
    if(NOT NCCL_HOME)
        set(NCCL_HOME "${ROOT}/comms/ncclx/stable")
    endif()

    # ── CVars generation ────────────────────────────────────────────────
    set(CVARS_DIR "${ROOT}/comms/utils/cvars")
    add_custom_command(
        OUTPUT "${CVARS_DIR}/nccl_cvars.h" "${CVARS_DIR}/nccl_cvars.cc"
        COMMAND ${CMAKE_COMMAND} -E env "NCCL_CVARS_OUTPUT_DIR=${CVARS_DIR}"
                ${Python3_EXECUTABLE} "${CVARS_DIR}/extractcvars.py"
        DEPENDS "${CVARS_DIR}/nccl_cvars.yaml"
                "${CVARS_DIR}/nccl_cvars.cc.in"
                "${CVARS_DIR}/extractcvars.py"
        COMMENT "Generating nccl_cvars files"
    )
    add_custom_target(ncclx_cvars DEPENDS
        "${CVARS_DIR}/nccl_cvars.h"
        "${CVARS_DIR}/nccl_cvars.cc"
    )

    # ── Git version header ──────────────────────────────────────────────
    set(GIT_VERSION_DIR "${NCCLX_BUILD_DIR}/obj/include")
    set(GIT_VERSION_HEADER "${GIT_VERSION_DIR}/nccl_git_version.h")

    # Detect DEV_SIGNATURE
    set(DEV_SIGNATURE $ENV{DEV_SIGNATURE})
    if(NOT DEV_SIGNATURE)
        execute_process(
            COMMAND git rev-parse --short HEAD
            WORKING_DIRECTORY ${ROOT}
            OUTPUT_VARIABLE _git_short_hash
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _git_result
        )
        if(_git_result EQUAL 0 AND _git_short_hash)
            set(DEV_SIGNATURE "git-${_git_short_hash}")
        else()
            set(DEV_SIGNATURE "")
        endif()
    endif()

    file(MAKE_DIRECTORY "${GIT_VERSION_DIR}")
    set(_generate_script "${NCCL_HOME}/src/misc/generate_git_version.py")
    if(EXISTS "${_generate_script}")
        add_custom_command(
            OUTPUT "${GIT_VERSION_HEADER}"
            COMMAND ${Python3_EXECUTABLE} "${_generate_script}" "${GIT_VERSION_HEADER}"
            DEPENDS "${_generate_script}"
            COMMENT "Generating nccl_git_version.h"
        )
    else()
        file(WRITE "${GIT_VERSION_HEADER}"
"/* Auto-generated fallback. */
#ifndef NCCL_GIT_VERSION_H
#define NCCL_GIT_VERSION_H
#ifndef NCCL_GIT_BRANCH
#define NCCL_GIT_BRANCH \"unknown\"
#endif
#ifndef NCCL_GIT_COMMIT_HASH
#define NCCL_GIT_COMMIT_HASH \"${DEV_SIGNATURE}\"
#endif
#endif /* NCCL_GIT_VERSION_H */
")
    endif()
    add_custom_target(ncclx_git_version DEPENDS "${GIT_VERSION_HEADER}")

    # ── NVCC_GENCODE computation ────────────────────────────────────────
    set(NVCC_GENCODE $ENV{NVCC_GENCODE})
    if(NOT NVCC_GENCODE)
        set(NVCC_ARCH $ENV{NVCC_ARCH})
        if(NOT NVCC_ARCH)
            set(NVCC_ARCH "a100,h100")
        endif()

        # Auto-add b200 if CUDA 12.8+
        execute_process(
            COMMAND "${CUDA_HOME}/bin/nvcc" "--version"
            OUTPUT_VARIABLE _nvcc_version_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_nvcc_version_output MATCHES "release ([0-9]+)\\.([0-9]+)")
            set(_cuda_major "${CMAKE_MATCH_1}")
            set(_cuda_minor "${CMAKE_MATCH_2}")
            if((_cuda_major GREATER 12) OR (_cuda_major EQUAL 12 AND _cuda_minor GREATER_EQUAL 8))
                if(NOT NVCC_ARCH MATCHES "b200")
                    set(NVCC_ARCH "${NVCC_ARCH},b200")
                endif()
            endif()
        endif()

        # Map architecture names to gencode flags
        set(_gencode "")
        string(REPLACE "," ";" _arch_list "${NVCC_ARCH}")
        foreach(_arch ${_arch_list})
            if(_arch STREQUAL "p100")
                string(APPEND _gencode " -gencode=arch=compute_60,code=sm_60")
            elseif(_arch STREQUAL "v100")
                string(APPEND _gencode " -gencode=arch=compute_70,code=sm_70")
            elseif(_arch STREQUAL "a100")
                string(APPEND _gencode " -gencode=arch=compute_80,code=sm_80")
            elseif(_arch STREQUAL "h100")
                string(APPEND _gencode " -gencode=arch=compute_90,code=sm_90")
            elseif(_arch STREQUAL "b200")
                string(APPEND _gencode " -gencode=arch=compute_100,code=sm_100")
            elseif(_arch STREQUAL "b300")
                string(APPEND _gencode " -gencode=arch=compute_103a,code=sm_103a")
            endif()
        endforeach()
        string(STRIP "${_gencode}" NVCC_GENCODE)
    endif()
    message(STATUS "NVCC_GENCODE: ${NVCC_GENCODE}")

    # ── NCCLX include directory ─────────────────────────────────────────
    # The NCCLX make build uses -I$(CONDA_INCLUDE_DIR) for third-party
    # headers. With build_ncclx.sh all deps were installed into a single
    # prefix; with FetchContent they're scattered. Create a unified
    # include tree via symlinks at configure time.
    set(NCCLX_THIRD_PARTY_INCLUDE "${CMAKE_BINARY_DIR}/ncclx_include")
    file(MAKE_DIRECTORY "${NCCLX_THIRD_PARTY_INCLUDE}")

    # Helper: symlink a directory if the link doesn't exist yet
    macro(_ncclx_link_dir _name _src)
        if(NOT EXISTS "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}" AND EXISTS "${_src}")
            file(CREATE_LINK "${_src}" "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}" SYMBOLIC)
        endif()
    endmacro()

    # Helper: merge items from multiple source dirs into one target dir
    macro(_ncclx_merge_dir _name)
        file(MAKE_DIRECTORY "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}")
        foreach(_merge_src_dir ${ARGN})
            if(EXISTS "${_merge_src_dir}")
                file(GLOB _merge_items "${_merge_src_dir}/*")
                foreach(_item ${_merge_items})
                    get_filename_component(_base "${_item}" NAME)
                    if(NOT EXISTS "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}/${_base}")
                        file(CREATE_LINK "${_item}"
                            "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}/${_base}" SYMBOLIC)
                    endif()
                endforeach()
            endif()
        endforeach()
    endmacro()

    # Libraries with both source and generated headers need deep merging.
    # Use cp -rsn to create recursive symlink trees that overlay source
    # and build directories without conflicts.
    foreach(_pair
        "folly|${folly_SOURCE_DIR}/folly|${folly_BINARY_DIR}/folly"
        "thrift|${fbthrift_SOURCE_DIR}/thrift|${fbthrift_BINARY_DIR}/thrift"
        "fizz|${fizz_SOURCE_DIR}/fizz|${fizz_BINARY_DIR}/generated/fizz"
        "glog|${glog_SOURCE_DIR}/src/glog|${glog_BINARY_DIR}/glog"
        "event2|${libevent_SOURCE_DIR}/include/event2|${libevent_BINARY_DIR}/include/event2"
    )
        string(REPLACE "|" ";" _parts "${_pair}")
        list(GET _parts 0 _name)
        list(GET _parts 1 _src1)
        list(GET _parts 2 _src2)
        file(MAKE_DIRECTORY "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}")
        if(EXISTS "${_src1}")
            execute_process(COMMAND cp -rsn "${_src1}/." "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}/" ERROR_QUIET)
        endif()
        if(EXISTS "${_src2}")
            execute_process(COMMAND cp -rsn "${_src2}/." "${NCCLX_THIRD_PARTY_INCLUDE}/${_name}/" ERROR_QUIET)
        endif()
    endforeach()

    # Libraries with headers only in one location: symlink
    _ncclx_link_dir(fmt "${fmt_SOURCE_DIR}/include/fmt")
    _ncclx_link_dir(wangle "${wangle_SOURCE_DIR}/wangle")
    _ncclx_link_dir(gflags "${gflags_BINARY_DIR}/include/gflags")
    _ncclx_link_dir(double-conversion "${double-conversion_SOURCE_DIR}/double-conversion")

    # OpenSSL: symlink from wherever it was found (conda or system).
    # When built from source via ExternalProject, headers are copied at
    # build time in ncclx_build.sh instead (they don't exist yet here).
    _ncclx_link_dir(openssl "${OPENSSL_INCLUDE_DIR}/openssl")

    # Boost: each sub-library has libs/<name>/include/boost/
    # Use cp -rs to create recursive symlink trees (handles overlapping
    # subdirs like boost/detail/ that exist in multiple sub-libraries).
    file(GLOB _boost_lib_inc_dirs "${boost_SOURCE_DIR}/libs/*/include")
    foreach(_inc_dir ${_boost_lib_inc_dirs})
        if(IS_DIRECTORY "${_inc_dir}/boost")
            execute_process(
                COMMAND cp -rsn "${_inc_dir}/boost" "${NCCLX_THIRD_PARTY_INCLUDE}/"
                ERROR_QUIET
            )
        endif()
    endforeach()

    # libevent: folly includes <event.h> (root-level, not event2/)
    file(GLOB _libevent_root_headers "${libevent_SOURCE_DIR}/include/*.h")
    foreach(_hdr ${_libevent_root_headers})
        get_filename_component(_fname "${_hdr}" NAME)
        if(NOT EXISTS "${NCCLX_THIRD_PARTY_INCLUDE}/${_fname}")
            file(CREATE_LINK "${_hdr}"
                "${NCCLX_THIRD_PARTY_INCLUDE}/${_fname}" SYMBOLIC)
        endif()
    endforeach()

    # xxhash: individual header files
    if(DEFINED xxhash_SOURCE_DIR)
        foreach(_xxh xxhash.h xxh3.h xxh_x86dispatch.h)
            if(EXISTS "${xxhash_SOURCE_DIR}/${_xxh}")
                file(CREATE_LINK "${xxhash_SOURCE_DIR}/${_xxh}"
                    "${NCCLX_THIRD_PARTY_INCLUDE}/${_xxh}" SYMBOLIC)
            endif()
        endforeach()
    endif()

    # ── THIRD_PARTY_LDFLAGS from cmake targets ──────────────────────────
    # Use generator expressions to resolve library paths at generate time.
    # Wrap everything in --start-group / --end-group so the linker
    # resolves circular dependencies between static libs (folly↔wangle↔fizz).
    set(_ncclx_ldflags "-Wl,--start-group ")

    # comms_tracing_service
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:comms_tracing_service> ")

    # fbthrift libs
    foreach(_thrift_lib
        async concurrency thrift-core thriftanyrep thriftcpp2
        thriftmetadata thriftprotocol thrifttype thrifttyperep
        rpcmetadata runtime serverdbginfo transport common)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:${_thrift_lib}> ")
    endforeach()

    # folly, wangle, fizz, xxhash
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:folly> ")
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:wangle> ")
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:fizz> ")
    if(TARGET xxhash)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:xxhash> ")
    endif()

    # glog, gflags, boost_context, fmt, double-conversion, libevent
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:glog> ")
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:gflags> ")
    if(TARGET boost_context)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:boost_context> ")
    elseif(TARGET Boost::context)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:Boost::context> ")
    endif()
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:fmt> ")
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:double-conversion> ")
    if(TARGET event_core_static)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:event_core_static> ")
    endif()
    if(TARGET event_extra_static)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:event_extra_static> ")
    endif()

    # ssl, crypto
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:OpenSSL::SSL> ")
    string(APPEND _ncclx_ldflags "$<TARGET_FILE:OpenSSL::Crypto> ")

    # zlib
    if(TARGET zlibstatic)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:zlibstatic> ")
    elseif(TARGET ZLIB::ZLIB)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:ZLIB::ZLIB> ")
    endif()

    # zstd
    if(TARGET zstd::libzstd_static)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:zstd::libzstd_static> ")
    elseif(TARGET libzstd_static)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:libzstd_static> ")
    endif()

    # sodium (required by fizz)
    if(TARGET sodium)
        string(APPEND _ncclx_ldflags "$<TARGET_FILE:sodium> ")
    else()
        string(APPEND _ncclx_ldflags "-lsodium ")
    endif()

    string(APPEND _ncclx_ldflags "-Wl,--end-group ")

    # System libs — use -L to ensure conda/system libs are found
    string(APPEND _ncclx_ldflags "-L${CONDA_LIB} -lbz2 -llz4 -llzma -ldl -lpthread ")

    # Write the resolved flags to a file at generate time
    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/ncclx_third_party_ldflags.txt"
        CONTENT "${_ncclx_ldflags}"
    )

    # ── Build variables ─────────────────────────────────────────────────
    set(NCCL_FP8 $ENV{NCCL_FP8})
    if(NOT NCCL_FP8)
        set(NCCL_FP8 "1")
    endif()
    set(ENABLE_PIPES $ENV{ENABLE_PIPES})
    if(NOT ENABLE_PIPES)
        set(ENABLE_PIPES "0")
    endif()
    set(CUDARTLIB "cudart_static")

    # ── NCCLX make build via ExternalProject ────────────────────────────
    include(ExternalProject)
    include(ProcessorCount)
    ProcessorCount(_nproc)
    if(_nproc EQUAL 0)
        set(_nproc 8)
    endif()

    # Collect all cmake target dependencies so ExternalProject waits for them.
    set(_ncclx_deps ncclx_cvars ncclx_git_version comms_tracing_service)
    foreach(_dep folly wangle fizz fmt glog gflags openssl_external
                event_core_static event_extra_static
                double-conversion zlibstatic sodium_external)
        if(TARGET ${_dep})
            list(APPEND _ncclx_deps ${_dep})
        endif()
    endforeach()

    # Write a build script that reads THIRD_PARTY_LDFLAGS and INCFLAGS from
    # generated files and invokes make. This avoids shell quoting issues
    # in ExternalProject_Add.
    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/ncclx_build.sh"
        CONTENT "\
#!/bin/bash
set -e
# Merge build-time generated thrift headers into the include tree.
# These don't exist at cmake configure time, so we do it here.
cp -rsn ${fbthrift_BINARY_DIR}/thrift/. ${NCCLX_THIRD_PARTY_INCLUDE}/thrift/ 2>/dev/null || true
# Merge generated comms tracing service headers
cp -rsn ${CMAKE_BINARY_DIR}/comms/. ${NCCLX_THIRD_PARTY_INCLUDE}/comms/ 2>/dev/null || true
# Merge openssl headers (built via ExternalProject, not available at configure time)
if [ -d ${CMAKE_BINARY_DIR}/third_party/openssl-install/include/openssl ]; then
    cp -rsn ${CMAKE_BINARY_DIR}/third_party/openssl-install/include/openssl ${NCCLX_THIRD_PARTY_INCLUDE}/ 2>/dev/null || true
fi

THIRD_PARTY_LDFLAGS=\"$(cat ${CMAKE_BINARY_DIR}/ncclx_third_party_ldflags.txt)\"
exec make VERBOSE=1 -j${_nproc} src.build \
    BUILDDIR=${NCCLX_BUILD_DIR} \
    NVCC_GENCODE=\"${NVCC_GENCODE}\" \
    CUDA_HOME=${CUDA_HOME} \
    NCCL_HOME=${NCCL_HOME} \
    \"NCCL_SUFFIX=x-${DEV_SIGNATURE}\" \
    NCCL_FP8=${NCCL_FP8} \
    BASE_DIR=${ROOT} \
    CONDA_INCLUDE_DIR=${NCCLX_THIRD_PARTY_INCLUDE} \
    CONDA_LIB_DIR=${CONDA_LIB} \
    CUDARTLIB=${CUDARTLIB} \
    ENABLE_PIPES=${ENABLE_PIPES} \
    \"THIRD_PARTY_LDFLAGS=\$THIRD_PARTY_LDFLAGS\"
"
    )

    ExternalProject_Add(ncclx_build
        SOURCE_DIR "${NCCL_HOME}"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND bash "${CMAKE_BINARY_DIR}/ncclx_build.sh"
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE TRUE
        BUILD_BYPRODUCTS "${NCCLX_STATIC_LIB}" "${NCCLX_SHARED_LIB}"
        DEPENDS ${_ncclx_deps}
    )
endif()

# ── Check NCCLX include ────────────────────────────────────────────────

# For non-system-libs builds the include dir is created by the make build,
# so we skip the existence check at configure time and defer to build time.
if(USE_SYSTEM_LIBS AND NOT EXISTS "${NCCLX_INCLUDE}")
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

# Check if ENABLE_PIPES is requested via environment variable.
# When enabled, compiles pipes device backend support into _comms_ncclx.
# Requires NCCL built with ENABLE_PIPES=1 (pipes transport in libnccl.so).
if("$ENV{ENABLE_PIPES}" STREQUAL "1")
    set(TORCHCOMMS_ENABLE_PIPES ON)
    message(STATUS "ENABLE_PIPES=1 set, building with pipes support")
    if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
        file(GLOB TORCHCOMMS_DEVICE_PIPES_SOURCE
            "comms/torchcomms/device/pipes/*.cpp")
    else()
        set(TORCHCOMMS_DEVICE_PIPES_SOURCE "")
    endif()
else()
    set(TORCHCOMMS_ENABLE_PIPES OFF)
    set(TORCHCOMMS_DEVICE_PIPES_SOURCE "")
    message(STATUS "ENABLE_PIPES not set, building without pipes support")
endif()

add_library(torchcomms_comms_ncclx MODULE
    ${TORCHCOMMS_NCCLX_SOURCES}
    ${TORCHCOMMS_CUDA_DEVICE_SOURCES}
    ${TORCHCOMMS_DEVICE_NCCLX_SOURCE}
    ${TORCHCOMMS_DEVICE_PIPES_SOURCE}
    comms/utils/CudaRAII.cc
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
target_compile_features(torchcomms_comms_ncclx PRIVATE cxx_std_20)

# Add device API compile definition if headers are available
if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
    target_compile_definitions(torchcomms_comms_ncclx
        PRIVATE TORCHCOMMS_HAS_NCCL_DEVICE_API=1)
endif()

# Add ENABLE_PIPES compile definition for pipes device backend
if(TORCHCOMMS_ENABLE_PIPES)
    target_compile_definitions(torchcomms_comms_ncclx
        PRIVATE ENABLE_PIPES=1)
endif()

target_link_directories(torchcomms_comms_ncclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_ncclx PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)

if(USE_SYSTEM_LIBS)
    # Get folly LDFLAGS using pkg-config (system/conda libs path)
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

    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_SHARED_LIB}
        ${FOLLY_LDFLAGS}
        "-lfmt"
        "-lboost_program_options"
        "-lboost_filesystem"
    )
else()
    # Ensure the ExternalProject build completes before linking
    add_dependencies(torchcomms_comms_ncclx ncclx_build)

    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_STATIC_LIB}
        comms_tracing_service
        Folly::folly
        fizz::fizz
        OpenSSL::SSL
        OpenSSL::Crypto
        fmt::fmt
        glog::glog
        gflags
    )
    if(TARGET wangle::wangle)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE wangle::wangle)
    elseif(TARGET wangle)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE wangle)
    endif()
    if(TARGET xxHash::xxhash)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE xxHash::xxhash)
    elseif(TARGET xxhash)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE xxhash)
    endif()
    if(TARGET Boost::context)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE Boost::context)
    endif()
    if(TARGET Boost::filesystem)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE Boost::filesystem)
    endif()
    if(TARGET double-conversion)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE double-conversion)
    endif()
    if(TARGET event_core_static)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE event_core_static)
    endif()
    if(TARGET ZLIB::ZLIB)
        target_link_libraries(torchcomms_comms_ncclx PRIVATE ZLIB::ZLIB)
    endif()
    # Folly's static lib pulls in compression and system dependencies
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        -lbz2 -llz4 -llzma -lzstd -ldl -lpthread
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

_install(TARGETS torchcomms_comms_ncclx
    LIBRARY DESTINATION .
)
