# $Id: CMakeLists.txt 100707 2016-10-31 20:21:07Z ihrivnac $

#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(PTL_Examples LANGUAGES C CXX VERSION ${PTL_VERSION})

if(("${CMAKE_PROJECT_NAME}" STREQUAL "PTL" OR PTL_BUILD_EXAMPLES) AND
   CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang|AppleClang)")
    set(CMAKE_CXX_FLAGS "-W -Wall ${CMAKE_CXX_FLAGS}")
endif()

# add the common "library"
add_subdirectory(common)
list(APPEND EXTERNAL_LIBRARIES common)

add_subdirectory(minimal)
add_subdirectory(basic)
add_subdirectory(extended)
add_subdirectory(gpu)
