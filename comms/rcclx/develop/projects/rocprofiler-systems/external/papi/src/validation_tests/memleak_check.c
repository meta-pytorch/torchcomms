// This is a simple minimal test for memory leaks.
// This is automatically compiled with -g (including debug info).
// Execute: valgrind --leak-check=yes ./memleak_check 
// You can modify this program to check for leaks in specific 
// operations, as necessary. 

#include <papi.h>

int main(int argv, char **argc) {
   (void) argv;                                       // prevent warning for not-used.
   (void) argc;                                       // prevent warning for not-used.

   int retval = PAPI_library_init(PAPI_VER_CURRENT);  // This does several allocations.
   (void) retval;                                     // prevent warning for not-used.

   PAPI_shutdown();                                   // Shutdown should release them all. 
   return(0);
} // end main() 
