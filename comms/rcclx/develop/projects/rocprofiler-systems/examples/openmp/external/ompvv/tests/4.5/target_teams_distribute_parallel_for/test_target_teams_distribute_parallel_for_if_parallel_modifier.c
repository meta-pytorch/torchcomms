//===---- test_target_teams_distribute_parallel_for_if_parallel_modifier.c ------===//
//
// OpenMP API Version 4.5 Nov 2015
// 
// In this test we want to try to check if the if clause is working 
// when used with the combined construct target teams distribute parallel for
// and the parallel modifier is specified.
// To do this we check if offloading is working, if it is not, it won't be
// possible for us to tell if the test passed or not, since it depends on 
// offloading capabilities. 
//
// The if has a directive-name-modifier that specifies to which directive the 
// if applies to (either both directives, to the target or to the parallel). 
// we create three test, one for no directive, another one for the target directive
// and another one for the parallel directive 
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024

void checkPreconditions() {
  // We test if offloading is enable, and if 
  // the number of threads is not 1. Having 
  // the number of threads equal to 1 is legal, but 
  // we won't be able to test if the if is 
  // affecting this or not

  // Testing for offloading
  int isOffloading = 0;
  int i;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test that if(parallel:) is not affecting the target offloading");

  // Testing for number of threads
  int init_num_threads_dev[N], init_num_threads_host[N];
  // init_num_threads_dev and init_num_threads_host arrays initialization
  for (i = 0; i < N; i++) {
    init_num_threads_dev[i] = 0;
    init_num_threads_host[i] = 0;
  }

  // Get the init_num_threads for host and device. If it is 1, then we 
  // cannot test the if parallel 
  // See section 2.5.1 of the ref manual
#pragma omp target teams distribute parallel for num_threads(OMPVV_NUM_THREADS_DEVICE)
  for (i = 0; i < N; i++) {
    init_num_threads_dev[i] = omp_get_num_threads();
  }

#pragma omp parallel for num_threads(OMPVV_NUM_THREADS_DEVICE)
  for (i = 0; i < N; i++) {
    init_num_threads_host[i] = omp_get_num_threads();
  }
  
  // We check that not all the values are 1. This would make the test of if parallel
  // not possible. 
  int raiseWarningDevice = 0, raiseWarningHost = 0;
  for (i = 0; i < N; i++) {
    if (init_num_threads_dev[i] > 1 ) {
      raiseWarningDevice +=  1;
    }
    if ( init_num_threads_host[i] > 1) {
      raiseWarningHost += 1;
    }
  }
  OMPVV_WARNING_IF(raiseWarningDevice == 0, "Initial number of threads in device was 1. It is not possible to test the if for the parallel directive");
  OMPVV_WARNING_IF(raiseWarningHost == 0, "Initial number of threads in host was 1. It is not possible to test the if for parallel");

}


int test_target_teams_distribute_if_parallel_modifier() {
  OMPVV_INFOMSG("test_target_teams_distribute_if_parallel_modifier");
  
  int a[N];
  int warning[N] ; // num_threads = 1 is not technically an error
  int attempt = 0;
  int errors = 0;
  int i;
  
  checkPreconditions();
  // Initialize a and warning
  for (i = 0; i < N; i++) {
    a[i] = 0;
    warning[i] = 0;
  }

  // We iterates NUM_ATTEMPTS times. When the iteration value is >= ATTEMPT_THRESHOLD the 
  // number of threads is expected to be greater than 1, (although if it is not, this is 
  // not an error, just warning)
  // If the iteration variable attempt < ATTEMPT_THRESHOLD, the if condition evaluates 
  // to false, and then the number of threads should be 1. Regardless of the value of 
  // the condition the execution should always occur in the device. 
  // We check that when we are never executing on the host, and we check that when the 
  // condition evaluates to false, the number of threads is always 1. We also raise a 
  // warning if the number of threads is 1 when the condition evaluates to true. 
  for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute parallel for if(parallel: attempt >= ATTEMPT_THRESHOLD)\
    map(tofrom: a, warning) num_threads(OMPVV_NUM_THREADS_DEVICE)
    for (i = 0; i < N; i++) {
      if (omp_is_initial_device())
        a[i] += 10; // This +10 should not happen

      if (attempt >= ATTEMPT_THRESHOLD) {
        // This is not an error, but we should mention it
        if (omp_get_num_threads() == 1) {
          warning[i] += 1;
        }
      } else {
        a[i] += (omp_get_num_threads() != 1) ? 10 : 1;  
      }
    }
  }

  int raiseWarning = 0;
  for (i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, a[i] != ATTEMPT_THRESHOLD);
    if (warning[i] != 0) {
      raiseWarning++;
    }
  }

  OMPVV_WARNING_IF(raiseWarning == N * (NUM_ATTEMPTS - ATTEMPT_THRESHOLD), "The number of threads was 1 when a number > 1 was expected. if(parallel:true). Not a compliance error in the specs");
  OMPVV_ERROR_IF(errors, "error in if(parallel: modifier). Possible causes are: the execution occurred in the host even though it should not affect the target region. The number of threads was > 1 when if(false).");

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if_parallel_modifier());

  OMPVV_REPORT_AND_RETURN(errors);
}
