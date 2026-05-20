set(NAME timing)

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# split the source files evenly
list(LENGTH source_files NUM_SOURCES)
math(EXPR CMAKE_UNITY_BUILD_BATCH_SIZE "(${NUM_SOURCES} / 2) + (${NUM_SOURCES} % 2)")

# split again if greater than 6
if(CMAKE_UNITY_BUILD_BATCH_SIZE GREATER 6)
    math(EXPR CMAKE_UNITY_BUILD_BATCH_SIZE
         "(${CMAKE_UNITY_BUILD_BATCH_SIZE} / 2) + (${CMAKE_UNITY_BUILD_BATCH_SIZE} % 2)")
endif()

timemory_build_intermediate_library(
    USE_INTERFACE
    NAME ${NAME}
    TARGET ${NAME}-component
    CATEGORY COMPONENT
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    PROPERTY_DEPENDS GLOBAL)
