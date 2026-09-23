# ----------------------------------------------------------------------------
# minimal example
#
add_executable(ptl-minimal minimal.cc)
target_link_libraries(ptl-minimal PRIVATE PTL::ptl common)
