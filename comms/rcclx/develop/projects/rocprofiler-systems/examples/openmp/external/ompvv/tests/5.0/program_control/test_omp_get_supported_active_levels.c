//===--- test_omp_get_supported_active_levels.c -----------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// Test for support of the omp_get_supported_active_levels() routine. 
// This routine returns the number of active levels of parallelism 
// supported by the implementation. This returned value must be greater
// than 0 and the max-active-levels-var ICV may not have a value that is greater
// than this number.
// 
///===-----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1028

int main() {

   int errors;
   int num_supp_active_levels;
   int max_active_levels;

   errors = 0;
 
   num_supp_active_levels = omp_get_supported_active_levels();
   max_active_levels = omp_get_max_active_levels();

   OMPVV_TEST_AND_SET_VERBOSE(errors, max_active_levels > num_supp_active_levels);
   OMPVV_TEST_AND_SET_VERBOSE(errors, num_supp_active_levels <= 0);

   OMPVV_REPORT_AND_RETURN(errors);

}
