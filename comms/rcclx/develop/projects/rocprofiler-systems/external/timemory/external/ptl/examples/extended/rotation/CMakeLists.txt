#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(PTL_Rotation_Examples LANGUAGES C CXX)


#----------------------------------------------------------------------------
# Find PTL project
#
find_package(PTL REQUIRED)
list(APPEND EXTERNAL_LIBRARIES ${PTL_LIBRARIES})
list(APPEND TARGET_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/source ${PTL_INCLUDE_DIRS})


#----------------------------------------------------------------------------
# OpenCV
#
set(OpenCV_COMPONENTS opencv_core opencv_imgproc opencv_highgui)
find_package(OpenCV QUIET COMPONENTS ${OpenCV_COMPONENTS})
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found! Skipping example '${PROJECT_NAME}'...")
    return()
endif()
list(APPEND EXTERNAL_LIBRARIES ${OpenCV_LIBRARIES})


#----------------------------------------------------------------------------
# Data files
#
configure_file(${PROJECT_SOURCE_DIR}/data/cameraman.tif
    ${CMAKE_BINARY_DIR}/data/cameraman.tif COPYONLY)


#----------------------------------------------------------------------------
# Source files
#
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/source/*.hh)
file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/source/*.cc)

set(EXE_NAME rotation)

add_executable(${EXE_NAME} ${EXE_NAME}.cc
    ${sources} ${headers})

target_link_libraries(${EXE_NAME} PUBLIC
    ${EXTERNAL_LIBRARIES})

target_include_directories(${EXE_NAME} PRIVATE
    ${TARGET_INCLUDE_DIRECTORIES})

target_compile_options(${EXE_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)


#----------------------------------------------------------------------------
# installation
if(NOT DEFINED PTL_DEVELOPER_INSTALL OR PTL_DEVELOPER_INSTALL)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
