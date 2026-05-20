add_library(common INTERFACE)
target_sources(common INTERFACE utils.hh)
target_include_directories(common INTERFACE ${PROJECT_SOURCE_DIR})
