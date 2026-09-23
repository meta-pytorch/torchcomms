#
# remove the Init.cmake so that original Stages.cmake does not execute its workflow
#
execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_LIST_DIR}/Init.cmake
                OUTPUT_QUIET)

#
# execute the custom stages
#
include("${CMAKE_CURRENT_LIST_DIR}/CustomStages.cmake")
