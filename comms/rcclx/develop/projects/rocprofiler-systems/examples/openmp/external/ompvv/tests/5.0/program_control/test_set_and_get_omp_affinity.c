//===--- test_set_and_get_omp_affinity.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the modification and retreival of the affinity-format-var 
// through the use of the omp_set_affinity_format() and omp_get_affinity_format() 
// routines. The omp_display_affinity() routine is used to display affinity format.
//
// Adopted from OpenMP 5.0 Example affinity_display.3.c
//===------------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include "ompvv.h"

#define FORMAT_STORE 80 

int main () {

   int errors;
   size_t nchars;
   size_t set_nchars;
   errors = 0;
    
   char default_format[FORMAT_STORE];
   char my_format[] = "host=%20H thrd_num=%0.4n binds_to=%A";

   // Display Default Affinity Format using omp_get_affinity_format and omp_display_affinity
   nchars = omp_get_affinity_format(default_format,(size_t)FORMAT_STORE);
   omp_display_affinity(default_format);
   OMPVV_TEST_AND_SET_VERBOSE(errors, nchars >= FORMAT_STORE);
   
   //Display Default Affinity Format after using omp_set_affinity_format using omp_display_affinity
   omp_set_affinity_format(my_format);
   set_nchars = omp_get_affinity_format(default_format,(size_t)FORMAT_STORE);
   omp_display_affinity(default_format);
   OMPVV_TEST_AND_SET_VERBOSE(errors, nchars == set_nchars);
   OMPVV_ERROR_IF(nchars == set_nchars, "Default affinity is the same number of characters as the set affinity, it is likely that omp_set_affinity() did not work properly.");
                                     
   OMPVV_REPORT_AND_RETURN(errors);
}
