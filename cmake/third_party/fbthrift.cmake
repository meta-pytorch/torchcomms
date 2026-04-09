# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(FBThrift REQUIRED)
elseif(CONDA_PREFIX)
    find_package(FBThrift QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET FBThrift::thriftcpp2)
    include(FetchContent)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        fbthrift
        GIT_REPOSITORY https://github.com/facebook/fbthrift.git
        GIT_TAG ${TORCHCOMMS_FBTHRIFT_VERSION}
        GIT_SHALLOW TRUE
    )
    # Use Populate + add_subdirectory so we can patch fbthrift's Find modules:
    # - FindSodium unsets our pre-set cache vars via change-tracking guard
    # - FindLibEvent uses get_target_property(LOCATION) which fails on build targets
    FetchContent_Populate(fbthrift)
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindSodium.cmake"
         DESTINATION "${fbthrift_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindLibEvent.cmake"
         DESTINATION "${fbthrift_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindZstd.cmake"
         DESTINATION "${fbthrift_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindZstd.cmake"
         DESTINATION "${fbthrift_SOURCE_DIR}/thrift/cmake/")
    add_subdirectory(${fbthrift_SOURCE_DIR} ${fbthrift_BINARY_DIR} EXCLUDE_FROM_ALL)

    # fbthrift's CMakeLists.txt uses include_directories(.) which is
    # directory-scoped and doesn't propagate to consumers outside fbthrift's
    # add_subdirectory scope. Add explicit PUBLIC include dirs so downstream
    # targets (comms_tracing_service) get thrift headers.
    target_include_directories(thriftcpp2 PUBLIC
        "${fbthrift_SOURCE_DIR}"
        "${fbthrift_BINARY_DIR}"
    )

    # fbthrift only creates namespaced targets at install time (NAMESPACE FBThrift::).
    # add_fbthrift_cpp_library links against FBThrift::thriftcpp2, so create the alias.
    if(NOT TARGET FBThrift::thriftcpp2 AND TARGET thriftcpp2)
        add_library(FBThrift::thriftcpp2 ALIAS thriftcpp2)
    endif()

    # Set variables needed by add_fbthrift_cpp_library (normally set by
    # FBThriftConfig.cmake at install time).
    set(FBTHRIFT_COMPILER "$<TARGET_FILE:thrift1>" CACHE STRING "" FORCE)
    set(FBTHRIFT_INCLUDE_DIR "${fbthrift_SOURCE_DIR}" CACHE PATH "" FORCE)

    # Make FBThriftCppLibrary.cmake, FBThriftConfig.cmake, and
    # FBCMakeParseArgs.cmake available to downstream
    # add_fbthrift_cpp_library() calls (e.g. comms tracing service).
    list(APPEND CMAKE_MODULE_PATH
        "${fbthrift_SOURCE_DIR}/build/fbcode_builder/CMake"
    )
endif()
