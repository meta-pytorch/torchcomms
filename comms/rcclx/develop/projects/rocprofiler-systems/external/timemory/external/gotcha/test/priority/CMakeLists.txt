add_library(priority_lib1 SHARED lib1.c)
add_library(priority_tool1 SHARED tool1.c)
add_library(priority_tool2 SHARED tool2.c)
add_library(priority_tool3 SHARED tool3.c)
add_executable(priority_app app.c)
target_link_libraries(priority_app gotcha)
target_link_libraries(priority_tool1 gotcha)
target_link_libraries(priority_tool2 gotcha)
target_link_libraries(priority_tool3 gotcha)
target_link_libraries(priority_app priority_lib1 priority_tool1 priority_tool2 priority_tool3)
add_test(test_priority_123 priority_app)
set_property(TEST test_priority_123
        PROPERTY
        ENVIRONMENT "tool_one_priority=1"
)
set_property(TEST test_priority_123
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=2"
)
set_property(TEST test_priority_123
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=3"
)
add_test(test_priority_132 priority_app)
set_property(TEST test_priority_132
        PROPERTY
        ENVIRONMENT "tool_one_priority=1"
)
set_property(TEST test_priority_132
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=3"
)
set_property(TEST test_priority_132
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=2"
)
add_test(test_priority_213 priority_app)
set_property(TEST test_priority_213
        PROPERTY
        ENVIRONMENT "tool_one_priority=2"
)
set_property(TEST test_priority_213
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=1"
)
set_property(TEST test_priority_213
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=3"
)
add_test(test_priority_231 priority_app)
set_property(TEST test_priority_231
        PROPERTY
        ENVIRONMENT "tool_one_priority=2"
)
set_property(TEST test_priority_231
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=3"
)
set_property(TEST test_priority_231
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=1"
)
add_test(test_priority_312 priority_app)
set_property(TEST test_priority_312
        PROPERTY
        ENVIRONMENT "tool_one_priority=3"
)
set_property(TEST test_priority_312
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=1"
)
set_property(TEST test_priority_312
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=2"
)
add_test(test_priority_321 priority_app)
set_property(TEST test_priority_321
        PROPERTY
        ENVIRONMENT "tool_one_priority=3"
)
set_property(TEST test_priority_321
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=2"
)
set_property(TEST test_priority_321
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=1"
)

#add_test(test_priority_221 priority_app)
#set_property(TEST test_priority_221
#  PROPERTY
#  ENVIRONMENT "tool_one_priority=2"
#)
#set_property(TEST test_priority_221
#  APPEND
#  PROPERTY
#  ENVIRONMENT "tool_two_priority=2"
#)
#set_property(TEST test_priority_221
#  APPEND
#  PROPERTY
#  ENVIRONMENT "tool_three_priority=1"
#)
add_test(test_priority_212 priority_app)
set_property(TEST test_priority_212
        PROPERTY
        ENVIRONMENT "tool_one_priority=2"
)
set_property(TEST test_priority_212
        APPEND
        PROPERTY
        ENVIRONMENT "tool_two_priority=1"
)
set_property(TEST test_priority_212
        APPEND
        PROPERTY
        ENVIRONMENT "tool_three_priority=2"
)
#add_test(test_priority_122 priority_app)
#set_property(TEST test_priority_122
#  PROPERTY
#  ENVIRONMENT "tool_one_priority=1"
#)
#set_property(TEST test_priority_122
#  APPEND
#  PROPERTY
#  ENVIRONMENT "tool_two_priority=2"
#)
#set_property(TEST test_priority_122
#  APPEND
#  PROPERTY
#  ENVIRONMENT "tool_three_priority=2"
#)
