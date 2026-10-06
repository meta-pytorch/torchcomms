# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# The uniflow._core extension. UniflowPy.cpp lives one level up next to the C++
# it wraps; this directory exists so the extension and the Python files that
# import it are staged together as an importable package.

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Without this pybind11 falls back to the legacy FindPythonLibs and can resolve a
# different interpreter than the one found above -- on a host with several
# Pythons that silently yields an extension tagged for the wrong ABI, which then
# fails to import. Set before pybind11 is resolved so both paths honour it.
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Let pybind11 reuse the Python found here" FORCE)
# PYBIND11_FINDPYTHON makes pybind11 call find_package(Python), whose cache
# variables are disjoint from FindPython3's, so the interpreter has to be named
# again here or pybind11 is free to resolve a different one.
set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
set(Python_ROOT_DIR "${Python3_ROOT_DIR}" CACHE PATH "" FORCE)

# pybind11 from the environment when present (a pip or system install exposes it
# through Python3), otherwise fetched, matching how spdlog is resolved. Setting
# UNIFLOW_PYBIND11_INCLUDE_DIR skips the search and uses a header tree, the same
# escape hatch spdlog and fmt have.
set(UNIFLOW_PYBIND11_INCLUDE_DIR "" CACHE PATH "Header-only pybind11 include directory")

if(NOT UNIFLOW_PYBIND11_INCLUDE_DIR AND NOT TARGET pybind11::module)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE UNIFLOW_PYBIND11_CMAKEDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE UNIFLOW_PYBIND11_PROBE
    )
    if(UNIFLOW_PYBIND11_PROBE EQUAL 0)
      find_package(pybind11 CONFIG REQUIRED PATHS "${UNIFLOW_PYBIND11_CMAKEDIR}" NO_DEFAULT_PATH)
    else()
      message(STATUS "pybind11 not found, fetching from GitHub...")
      include(FetchContent)
      FetchContent_Declare(
          pybind11
          GIT_REPOSITORY https://github.com/pybind/pybind11.git
          GIT_TAG v2.13.6
      )
      FetchContent_MakeAvailable(pybind11)
    endif()
  endif()
endif()

# The package directory is assembled in the build tree so the extension can be
# imported from there without installing: _core lands next to the __init__.py
# that re-exports it.
set(UNIFLOW_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/uniflow)

if(UNIFLOW_PYBIND11_INCLUDE_DIR)
  Python3_add_library(uniflow_core_extension MODULE WITH_SOABI
      ${CMAKE_CURRENT_SOURCE_DIR}/../UniflowPy.cpp
  )
  target_include_directories(uniflow_core_extension SYSTEM PRIVATE ${UNIFLOW_PYBIND11_INCLUDE_DIR})
  # pybind11 requires hidden visibility so that types registered by one module
  # cannot collide with another's.
  set_target_properties(uniflow_core_extension PROPERTIES CXX_VISIBILITY_PRESET hidden)
else()
  pybind11_add_module(uniflow_core_extension
      ${CMAKE_CURRENT_SOURCE_DIR}/../UniflowPy.cpp
  )
endif()

set_target_properties(uniflow_core_extension PROPERTIES
    OUTPUT_NAME "_core"
    LIBRARY_OUTPUT_DIRECTORY "${UNIFLOW_PYTHON_PACKAGE_DIR}"
)

target_compile_features(uniflow_core_extension PRIVATE cxx_std_20)

target_link_libraries(uniflow_core_extension PRIVATE uniflow)

# __init__.py names every symbol it re-exports, so a stale copy fails the import
# rather than silently exposing an older surface; _core.pyi rides along to keep
# the installed package typed.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../__init__.py
    ${UNIFLOW_PYTHON_PACKAGE_DIR}/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../_core.pyi
    ${UNIFLOW_PYTHON_PACKAGE_DIR}/_core.pyi COPYONLY)

# The extension is installed next to the library it links, and finds it there
# rather than through a system path, so the package is importable wherever it is
# unpacked. The build tree needs the same so the staged package imports without
# installing.
set_target_properties(uniflow_core_extension PROPERTIES
    BUILD_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS uniflow_core_extension uniflow LIBRARY DESTINATION uniflow)
set(_UNIFLOW_FETCHED_DEPENDENCY_TARGETS)
if(UNIFLOW_FETCHED_FMT)
  list(APPEND _UNIFLOW_FETCHED_DEPENDENCY_TARGETS fmt)
endif()
if(UNIFLOW_FETCHED_SPDLOG)
  list(APPEND _UNIFLOW_FETCHED_DEPENDENCY_TARGETS spdlog)
endif()
if(_UNIFLOW_FETCHED_DEPENDENCY_TARGETS AND
   (NOT DEFINED UNIFLOW_INSTALL_CPP OR NOT UNIFLOW_INSTALL_CPP))
  install(TARGETS ${_UNIFLOW_FETCHED_DEPENDENCY_TARGETS}
      LIBRARY DESTINATION uniflow
      RUNTIME DESTINATION uniflow
      PUBLIC_HEADER DESTINATION uniflow/include EXCLUDE_FROM_ALL)
endif()
unset(_UNIFLOW_FETCHED_DEPENDENCY_TARGETS)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../_core.pyi
    DESTINATION uniflow
)
