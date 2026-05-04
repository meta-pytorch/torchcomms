#
# This file exists for the user to applications to create their own interface libraries
# that are exported along with the rest of the library
#
# See Packages.cmake for several examples on how this is done. In general, follow pattern
# of:
#
# timemory_add_interface_library(timemory-<SOME_PACKAGE>)
#
# find_package(<SOME_PACKAGE>) if(<SOME_PACKAGE>_FOUND) # populate target flags,
# definitions, etc. else() timemory_inform_empty_interface(timemory-<SOME_PACKAGE> "brief
# about <SOME_PACKAGE>") endif()
#

# include guard
include_guard(DIRECTORY)
