# ----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.8...3.20 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
project(PTL_Examples)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang|AppleClang)")
    string(APPEND CMAKE_CXX_FLAGS " -W -Wall")
endif()

find_package(PTL REQUIRED)

# add the common library Gperf and ittnotify options?
add_subdirectory(common)

# now the actual examples
add_subdirectory(minimal)
add_subdirectory(basic)
add_subdirectory(extended)

# Commenting as it can not currently compile add_subdirectory(gpu)
