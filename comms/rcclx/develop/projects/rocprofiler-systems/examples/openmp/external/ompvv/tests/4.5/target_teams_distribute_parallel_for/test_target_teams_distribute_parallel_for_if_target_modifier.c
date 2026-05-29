//===---- test_target_teams_distribute_parallel_for_if_no_modifier.c --------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// In this test we want to try to check if the if clause is working 
// when used with the combined construct target teams distribute parallel for
// and the target modifier is specified.
// To do this we check if offloading is working, if it is not, it won't be
// possible for us to tell if the test passed or not, since it depends on 
// offloading capabilities. 
//
// The if has a directive-name-modifier that specifies to which directive the 
// if applies to (either both directives, to the target or to the parallel). 
// We have three tests: one for no directive, another for the target directive
// and another for the parallel directive 
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024

int checkPreconditions() {
  // We test if offloading is enable

  // Testing for offloading
  int isOffloading = 0;
  int i;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test if");

  return isOffloading;
}

int test_target_teams_distribute_if_target_modifier() {
  OMPVV_INFOMSG("test_target_teams_distribute_if_target_modifier");
  
  int a[N], warning[N];
  int attempt = 0;
  int errors = 0;
  int i;
  int isOffloading;

  isOffloading = checkPreconditions();

  for (i = 0; i < N; i++) {
    a[i] = 1;
    warning[i] = 0;
  }

  // We iterate NUM_ATTEMPTS times. When the iteration value is >= ATTEMPT_THRESHOLD the 
  // execution of the target region should happen in the device, and the number of threads
  // is expected to be greater than 1 (although if it is not, this is not an error, just warning)
  // If the iteration variable attempt  < ATTEMPT_THRESHOLD, then the if condition evaluates to false, 
  // and hence the execution should occur in the host, while the number of threads should not change
  // We check that when we are executing on the host, the value of the iteration variable is 
  // greather than ATTEMPT_THRESHOLD
  for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute parallel for if(target: attempt >= ATTEMPT_THRESHOLD)\
    map(tofrom: a) num_threads(OMPVV_NUM_THREADS_DEVICE)
    for (i = 0; i < N; i++) {
      warning[i] += (omp_get_num_threads() == 1) ? 1 : 0; // Ideally we should not change the number of threads at any point
      if (attempt >= ATTEMPT_THRESHOLD) {
        a[i] += (isOffloading && omp_is_initial_device() ? 10 : 0); // True condition, it should run on the device
      }
      else {
        a[i] += (omp_is_initial_device() ? 1 : 100);
      } 
    }
  }

  int raiseWarning = 0;
  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, a[i] != 1 + (ATTEMPT_THRESHOLD));
    if (warning[i] != 0) {
      raiseWarning = 1;
    }
  }

  OMPVV_WARNING_IF(raiseWarning != 0, "The number of threads was 1 even though we expected it to be more than 1. Not a compliance error in the specs");
  OMPVV_ERROR_IF(errors, "error in if(target: modifier). The execution was expected to occur in the device, but it happened in the host when if(false), or the other way around");

  return errors;
}

int main() {
 
  int errors = 0;

  //Offloading is checked in checkPreconditions() function
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if_target_modifier());

  OMPVV_REPORT_AND_RETURN(errors);
}
