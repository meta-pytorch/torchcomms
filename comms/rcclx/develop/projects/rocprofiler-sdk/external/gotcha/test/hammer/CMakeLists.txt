#The test size is based around (XSIZE * YSIZE) change these variables to make
# the test smaller/larger.  
#
#Keep this test small for day-to-day builds.  But occasionally, before release
#or when performance testing, it's worth bumping this up to something big
#e.g, 64x64.  Note that this will take hours to build (but parallelizes well)
#and about a dozen gigs of disk space.
#
#Don't enable debug information at (64 * 64), do that at something like (4 * 4).
#Also, enabling optimizations in gcc currently breaks the test by eliminating
#needed symbols.  Don't do it.

set(XSIZE 1)
set(YSIZE 1)

set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -Wextra")

add_definitions(-DXSIZE=${XSIZE})
add_definitions(-DYSIZE=${YSIZE})
add_definitions(--std=c++11)
include_directories(${CMAKE_BINARY_DIR}/test/hammer)

add_library(math SHARED gen.cc)

add_library(wrap SHARED wrap.cc)
target_link_libraries(wrap gotcha dl)

add_custom_target(externals.h
        COMMAND ${CMAKE_SOURCE_DIR}/test/hammer/gen_externals_h.sh ${YSIZE} ${XSIZE})

add_executable(hammer hammer.cc)
target_link_libraries(hammer math wrap)

foreach (y RANGE ${YSIZE})
    foreach (x RANGE ${XSIZE})
        add_library(ref${y}_${x} SHARED multref.cc addref.cc)
        target_compile_definitions(ref${y}_${x} PRIVATE -DSTARTY=${y} PRIVATE -DSTARTX=${x})
        add_dependencies(ref${y}_${x} externals.h)
        target_link_libraries(ref${y}_${x} wrap)
        target_link_libraries(hammer ref${y}_${x})
    endforeach (x)
endforeach (y)

gotcha_add_test(hammer_time_test hammer)
environment_add(hammer_time_test TEST "GOTCHA_DEBUG=3")
set(preload_libs "")
foreach (y RANGE ${YSIZE})
    foreach (x RANGE ${XSIZE})
        string(CONCAT preload_libs ${preload_libs} ":${CMAKE_CURRENT_BINARY_DIR}/libref${y}_${x}.so")
    endforeach (x)
endforeach (y)
environment_add(hammer_time_test TEST "LD_PRELOAD=${preload_libs}")