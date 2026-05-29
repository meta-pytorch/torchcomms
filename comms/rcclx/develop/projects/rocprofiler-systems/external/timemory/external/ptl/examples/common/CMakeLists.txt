
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/*.hh)
get_filename_component(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)

add_library(common INTERFACE)
target_sources(common INTERFACE ${headers})
target_include_directories(common INTERFACE ${COMMON_INCLUDE_DIR})
