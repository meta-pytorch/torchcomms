# Auto-generated stub: fizz was provided via FetchContent.
set(fizz_FOUND TRUE)
if(NOT TARGET fizz::fizz AND TARGET fizz)
    add_library(fizz::fizz ALIAS fizz)
endif()
# Set FIZZ_LIBRARIES so downstream libs (wangle) that link ${FIZZ_LIBRARIES} work.
set(FIZZ_LIBRARIES fizz::fizz)
