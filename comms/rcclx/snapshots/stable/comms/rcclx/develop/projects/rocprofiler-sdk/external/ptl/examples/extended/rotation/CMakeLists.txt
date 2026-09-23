# ----------------------------------------------------------------------------
# OpenCV
#
set(OpenCV_COMPONENTS opencv_core opencv_imgproc opencv_highgui)
find_package(OpenCV QUIET COMPONENTS ${OpenCV_COMPONENTS})
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found! Skipping example 'extended/rotation'")
    return()
endif()

# ----------------------------------------------------------------------------
# Data files
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/data/cameraman.tif
               ${CMAKE_BINARY_DIR}/data/cameraman.tif COPYONLY)

# ----------------------------------------------------------------------------
# Build and link executable
#
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/source/*.hh)
file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/source/*.cc)

add_executable(rotation rotation.cc ${sources} ${headers})
target_include_directories(rotation PRIVATE "${CMAKE_CURRENT_LIST_DIR}/source")
target_link_libraries(rotation PUBLIC PTL::ptl ${OpenCV_LIBS} common)
