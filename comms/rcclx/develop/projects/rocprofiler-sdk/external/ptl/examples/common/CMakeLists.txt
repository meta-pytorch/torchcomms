add_library(common STATIC utils.hh Backtrace.hh Timer.hh Timer.cc)
target_include_directories(common PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(common PUBLIC PTL::ptl)
