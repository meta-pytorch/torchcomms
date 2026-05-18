# ----------------------------------------------------------------------------
# standard example
#
add_executable(tasking tasking.cc)
target_link_libraries(tasking PRIVATE PTL::ptl common)

if(PTL_TBB_FOUND)
    add_executable(tbb_tasking tasking.cc)
    target_compile_definitions(tbb_tasking PRIVATE USE_TBB_TASKS)
    target_link_libraries(tbb_tasking PTL::ptl common)
endif()

# ----------------------------------------------------------------------------
# recursive example
#
add_executable(recursive_tasking recursive_tasking.cc)
target_link_libraries(recursive_tasking PRIVATE PTL::ptl common)

if(PTL_TBB_FOUND)
    add_executable(recursive_tbb_tasking recursive_tasking.cc)
    target_compile_definitions(recursive_tbb_tasking PRIVATE USE_TBB_TASKS)
    target_link_libraries(recursive_tbb_tasking PTL::ptl common)
endif()
