//===--- test_task_depend_mutexinoutset.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//  
// This test verifies the working of the mutexinout on the depend clause.
// Here task T5 will be scheduled after tasks T1 and T3 are completed. Due to 
// the mutexinoutset dependence type on c, T4 and T5 may be scheduled
// in any order with respect to each other, but not at the same time. Tasks 
// T6 will be scheduled after both T4 and T5 are completed.
//
// Adapted from OpenMP Examples 5.0.
////===----------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int test_task_depend_mutexinoutset() {
  OMPVV_INFOMSG("test_task_mutexinoutset");
  int errors = 0;
  int a, b, c, d;

#pragma omp parallel
#pragma omp single
  {
#pragma omp task depend(out: c)
   c = 1; /* Task T1 */
 #pragma omp task depend(out: a)
   a = 2; /* Task T2 */
 #pragma omp task depend(out: b)
   b = 3; /* Task T3 */
 #pragma omp task depend(in: a) depend(mutexinoutset: c)
   c += a; /* Task T4 */
 #pragma omp task depend(in: b) depend(mutexinoutset: c)
   c += b; /* Task T5 */
 #pragma omp task depend(in: c)
   d = c; /* Task T6 */
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, d != 6);

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_depend_mutexinoutset());

  OMPVV_REPORT_AND_RETURN(errors);
}
