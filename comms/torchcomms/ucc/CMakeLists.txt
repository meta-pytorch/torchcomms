# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_ucc

include(FetchContent)
include(ExternalProject)

# ---------------------------------------------------------------------------
# Build UCX and UCC as shared libraries via ExternalProject so we can
# control ./configure flags.
#
# UCC uses a plugin architecture (CL/TL/MC/EC components) that are loaded
# via dlopen at runtime.  These plugins link against libucc.so, so we must
# use shared libraries to avoid symbol duplication between a static copy
# in our extension and the shared copy loaded by plugins.
# ---------------------------------------------------------------------------

set(UCC_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/ucc_deps")

# Locate autotools (autoreconf, libtoolize) needed for building from git.
# Look in standard locations and conda environments.
find_program(AUTORECONF autoreconf
    HINTS ENV PATH
          "$ENV{HOME}/.conda/envs/autotools/bin"
          "$ENV{CONDA_PREFIX}/bin"
)
if(NOT AUTORECONF)
    message(FATAL_ERROR "autoreconf not found. Install autoconf/automake/libtool.")
endif()
get_filename_component(AUTOTOOLS_BIN_DIR "${AUTORECONF}" DIRECTORY)
message(STATUS "Found autoreconf: ${AUTORECONF}")

# --- UCX (required by UCC) ------------------------------------------------
# Use release tarball which ships with pre-generated configure script.
FetchContent_Declare(
  ucx
  URL https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
)
FetchContent_Populate(ucx)

ExternalProject_Add(ucx_build
  SOURCE_DIR     "${ucx_SOURCE_DIR}"
  PREFIX         "${UCC_PREFIX}/ucx"
  CONFIGURE_COMMAND
    "${ucx_SOURCE_DIR}/configure"
      --prefix=<INSTALL_DIR>
      --enable-shared
      --with-pic
      --without-cuda
      --without-rocm
      --without-java
      --without-go
      --disable-logging
      --disable-debug
      --disable-assertions
      --disable-params-check
  BUILD_COMMAND     make -j
  INSTALL_COMMAND   make install
  BUILD_IN_SOURCE   FALSE
  BUILD_BYPRODUCTS
    "<INSTALL_DIR>/lib/libucp.so"
    "<INSTALL_DIR>/lib/libuct.so"
    "<INSTALL_DIR>/lib/libucs.so"
    "<INSTALL_DIR>/lib/libucm.so"
)

ExternalProject_Get_Property(ucx_build INSTALL_DIR)
set(UCX_INSTALL_DIR "${INSTALL_DIR}")

# --- UCC ------------------------------------------------------------------
# Clone from git and run autoreconf to generate configure script.
FetchContent_Declare(
  ucc
  GIT_REPOSITORY https://github.com/openucx/ucc.git
  GIT_TAG        v1.7.0
  GIT_SHALLOW    TRUE
)
FetchContent_Populate(ucc)

# Generate configure script via autogen.sh (creates m4 files then runs autoreconf)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
    "PATH=${AUTOTOOLS_BIN_DIR}:$ENV{PATH}"
    /bin/bash "${ucc_SOURCE_DIR}/autogen.sh"
  WORKING_DIRECTORY "${ucc_SOURCE_DIR}"
  RESULT_VARIABLE _autogen_result
)
if(NOT _autogen_result EQUAL 0)
    message(FATAL_ERROR "autogen.sh failed for UCC (exit code ${_autogen_result})")
endif()

ExternalProject_Add(ucc_build
  SOURCE_DIR     "${ucc_SOURCE_DIR}"
  PREFIX         "${UCC_PREFIX}/ucc"
  DEPENDS        ucx_build
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env
      "PATH=${AUTOTOOLS_BIN_DIR}:$ENV{PATH}"
    "${ucc_SOURCE_DIR}/configure"
      --prefix=<INSTALL_DIR>
      --enable-shared
      --with-pic
      --with-ucx=${UCX_INSTALL_DIR}
      --without-cuda
      --without-rocm
      --without-nccl
  BUILD_COMMAND     make -j
  INSTALL_COMMAND   make install
  BUILD_IN_SOURCE   FALSE
  BUILD_BYPRODUCTS
    "<INSTALL_DIR>/lib/libucc.so"
)

ExternalProject_Get_Property(ucc_build INSTALL_DIR)
set(UCC_INSTALL_DIR "${INSTALL_DIR}")

# ---------------------------------------------------------------------------
# Python extension: _comms_ucc
# ---------------------------------------------------------------------------
file(GLOB TORCHCOMMS_UCC_SOURCES "comms/torchcomms/ucc/*.cpp")

add_library(torchcomms_comms_ucc MODULE ${TORCHCOMMS_UCC_SOURCES})
add_dependencies(torchcomms_comms_ucc ucx_build ucc_build)

set_target_properties(torchcomms_comms_ucc PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_ucc"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
    # RPATH: look in ucc/ subdirectory next to the extension for shared libs
    INSTALL_RPATH "$ORIGIN/ucc"
    BUILD_RPATH "${UCC_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib"
)
target_include_directories(torchcomms_comms_ucc PRIVATE
    ${ROOT}
    ${FMT_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    "${UCX_INSTALL_DIR}/include"
    "${UCC_INSTALL_DIR}/include"
)
target_compile_features(torchcomms_comms_ucc PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_ucc PRIVATE
    ${CONDA_LIB}
    "${UCC_INSTALL_DIR}/lib"
    "${UCX_INSTALL_DIR}/lib"
)

# Link against UCC/UCX shared libraries.
# UCC's plugin architecture requires shared libs to avoid symbol duplication.
target_link_libraries(torchcomms_comms_ucc PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
    "-lucc"
    "-lucp"
    "-luct"
    "-lucs"
    "-lucm"
    "-lpthread"
    "-ldl"
    "-lm"
    "-lrt"
)

if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_ucc PRIVATE "-lfmt")
else()
    target_link_libraries(torchcomms_comms_ucc PRIVATE "-l:libfmt.a")
endif()

install(TARGETS torchcomms_comms_ucc
    LIBRARY DESTINATION .
)

# Install UCC shared library
install(DIRECTORY "${UCC_INSTALL_DIR}/lib/"
    DESTINATION ucc
    FILES_MATCHING
      PATTERN "libucc.so*"
      PATTERN "pkgconfig" EXCLUDE
      PATTERN "cmake" EXCLUDE
      PATTERN "ucc" EXCLUDE
)

# Install UCC component plugin shared libraries (.so files that UCC
# dlopen's at runtime).  UCC finds plugins by doing:
#   dirname(libucc.so) → .../ucc/  then appending UCC_MODULE_SUBDIR="ucc"
# So plugins must go into ucc/ucc/ for the path to resolve correctly.
install(DIRECTORY "${UCC_INSTALL_DIR}/lib/ucc/"
    DESTINATION ucc/ucc
    FILES_MATCHING PATTERN "*.so*"
)

# Also install UCX shared libraries needed by UCC transport plugins
install(DIRECTORY "${UCX_INSTALL_DIR}/lib/"
    DESTINATION ucc
    FILES_MATCHING
      PATTERN "*.so*"
      PATTERN "ucx" EXCLUDE
      PATTERN "pkgconfig" EXCLUDE
      PATTERN "cmake" EXCLUDE
)
# And UCX transport modules
if(EXISTS "${UCX_INSTALL_DIR}/lib/ucx/")
    install(DIRECTORY "${UCX_INSTALL_DIR}/lib/ucx/"
        DESTINATION ucc/ucx
        FILES_MATCHING PATTERN "*.so*"
    )
endif()

# Fix RPATH on UCC plugins (in ucc/ucc/) so they find libucc.so and
# UCX libs in the parent directory (ucc/).
find_program(PATCHELF patchelf)
if(PATCHELF)
    install(CODE "
        file(GLOB _ucc_plugins \"\${CMAKE_INSTALL_PREFIX}/ucc/ucc/libucc_*.so.*\")
        file(GLOB _ucc_plugins2 \"\${CMAKE_INSTALL_PREFIX}/ucc/ucc/libucc_*.so\")
        list(APPEND _ucc_plugins \${_ucc_plugins2})
        list(REMOVE_DUPLICATES _ucc_plugins)
        foreach(_plugin \${_ucc_plugins})
            execute_process(
                COMMAND ${PATCHELF} --set-rpath \"\\$ORIGIN/..\" \"\${_plugin}\"
                RESULT_VARIABLE _result
            )
            if(NOT _result EQUAL 0)
                message(WARNING \"Failed to patch RPATH on \${_plugin}\")
            endif()
        endforeach()
    ")
endif()
