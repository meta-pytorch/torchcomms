# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# Compiled libraries needed at link time.
set(_BOOST_COMPONENTS context date_time filesystem iostreams program_options regex system thread)

if(USE_SYSTEM_LIBS)
    find_package(Boost REQUIRED COMPONENTS ${_BOOST_COMPONENTS})
elseif(CONDA_PREFIX)
    find_package(Boost QUIET COMPONENTS ${_BOOST_COMPONENTS} HINTS ${CONDA_PREFIX})
endif()

if(NOT TARGET Boost::context)
    include(FetchContent)
    # Build all Boost libraries.  FB OSS deps (folly, fizz, wangle, fbthrift)
    # use a wide range of header-only boost sub-libraries (intrusive, algorithm,
    # container, preprocessor, etc.) and enumerating them is fragile.
    # Omitting BOOST_INCLUDE_LIBRARIES makes Boost configure everything.
    # This is slower but correct.
    set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG ${TORCHCOMMS_BOOST_VERSION}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(boost)

    # Collect ALL boost sub-library include directories.  Each sub-library
    # under libs/<name>/ has its own include/ dir.  Downstream FB OSS libs
    # (folly, wangle) reference ${Boost_INCLUDE_DIRS} directly in
    # target_include_directories, so we must populate this variable.
    file(GLOB _boost_lib_include_dirs "${boost_SOURCE_DIR}/libs/*/include")
    set(Boost_INCLUDE_DIRS "${_boost_lib_include_dirs}")
    set(Boost_INCLUDE_DIR "${_boost_lib_include_dirs}")

    # FetchContent creates Boost::headers but not the Boost::boost alias
    # that downstream libs (fbthrift, folly) expect.  Boost::headers only
    # provides libs/headers/include, not the per-library include dirs.
    # Create an INTERFACE library that aggregates all include paths and
    # the compiled libraries we need.
    if(NOT TARGET Boost::boost)
        add_library(_boost_all INTERFACE)
        target_include_directories(_boost_all INTERFACE ${_boost_lib_include_dirs})
        foreach(_comp IN LISTS _BOOST_COMPONENTS)
            if(TARGET Boost::${_comp})
                target_link_libraries(_boost_all INTERFACE Boost::${_comp})
            elseif(TARGET boost_${_comp})
                target_link_libraries(_boost_all INTERFACE boost_${_comp})
            endif()
        endforeach()
        add_library(Boost::boost ALIAS _boost_all)
    endif()

    # FetchContent creates Boost:: targets but downstream find_package(Boost)
    # won't find them. Generate a BoostConfig.cmake shim so CONFIG mode succeeds.
    # Convert the include dirs list to a semicolon-separated string for the shim.
    string(REPLACE ";" "\\\\;" _boost_inc_dirs_escaped "${_boost_lib_include_dirs}")
    set(_boost_config_dir "${CMAKE_BINARY_DIR}/cmake-shims")
    file(MAKE_DIRECTORY "${_boost_config_dir}")
    file(WRITE "${_boost_config_dir}/BoostConfig.cmake" "\
# Auto-generated shim: Boost was provided via FetchContent.
set(Boost_FOUND TRUE)
set(Boost_VERSION \"1.82.0\")
set(Boost_VERSION_MAJOR 1)
set(Boost_VERSION_MINOR 82)
set(Boost_VERSION_PATCH 0)
set(Boost_INCLUDE_DIR \"${_boost_inc_dirs_escaped}\")
set(Boost_INCLUDE_DIRS \"${_boost_inc_dirs_escaped}\")
set(Boost_LIBRARIES Boost::boost Boost::context Boost::date_time Boost::filesystem Boost::iostreams Boost::program_options Boost::regex Boost::system Boost::thread)
set(Boost_context_FOUND TRUE)
set(Boost_date_time_FOUND TRUE)
set(Boost_filesystem_FOUND TRUE)
set(Boost_iostreams_FOUND TRUE)
set(Boost_program_options_FOUND TRUE)
set(Boost_regex_FOUND TRUE)
set(Boost_system_FOUND TRUE)
set(Boost_thread_FOUND TRUE)
")
    file(WRITE "${_boost_config_dir}/BoostConfigVersion.cmake" "\
set(PACKAGE_VERSION \"1.82.0\")
if(\"\${PACKAGE_FIND_VERSION}\" VERSION_LESS_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(\"\${PACKAGE_FIND_VERSION}\" VERSION_EQUAL PACKAGE_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
")
    list(PREPEND CMAKE_PREFIX_PATH "${_boost_config_dir}")
    set(Boost_DIR "${_boost_config_dir}" CACHE PATH "" FORCE)
endif()
